"""Model-independent entropy screening by randomized image overlays."""

from pathlib import Path
from random import Random

from PIL import Image


class StripDetector:
    """Estimate prediction entropy under overlays from a clean reference set."""

    def __init__(self, model, transform, reference_files: list[Path], *,
                 overlays: int = 16, blend: float = 0.5, batch_size: int = 8,
                 device: str = "cpu"):
        if not reference_files or overlays < 1 or batch_size < 1 or not 0 < blend < 1:
            raise ValueError("Provide references, positive batch sizes, and blend in (0, 1)")
        self.model = model.eval()
        self.transform = transform
        self.reference_files = reference_files
        self.overlays = overlays
        self.blend = blend
        self.batch_size = batch_size
        self.device = device

    def score(self, image_path: str | Path, seed: int) -> float:
        import torch

        rng = Random(seed)
        with Image.open(image_path) as original:
            image = original.convert("RGB").copy()
        tensors = []
        for _ in range(self.overlays):
            reference_path = rng.choice(self.reference_files)
            with Image.open(reference_path) as original:
                reference = original.convert("RGB").resize(image.size)
            blended = Image.blend(image, reference, self.blend)
            tensors.append(self.transform(blended))
        entropies = []
        with torch.inference_mode():
            for start in range(0, len(tensors), self.batch_size):
                batch = torch.stack(tensors[start:start + self.batch_size]).to(self.device)
                logits = self.model(batch)
                probabilities = logits.softmax(dim=1)
                entropy = -(probabilities * logits.log_softmax(dim=1)).sum(dim=1)
                entropies.extend(entropy.cpu().tolist())
        return sum(entropies) / len(entropies)


def calibrate_threshold(clean_scores: list[float], false_positive_rate: float) -> float:
    """Low entropy is the suspicious side of the clean score distribution."""
    import numpy as np

    if not clean_scores or not 0 < false_positive_rate < 1:
        raise ValueError("Provide clean scores and false_positive_rate in (0, 1)")
    return float(np.quantile(clean_scores, false_positive_rate))


class NeuralCleanseDetector:
    """Estimate the smallest target-class mask on a fixed clean image batch."""

    def __init__(self, model, *, steps: int = 100, lr: float = 0.1,
                 regularization: float = 0.01, device: str = "cpu"):
        if steps < 1 or lr <= 0 or regularization < 0:
            raise ValueError("steps and lr must be positive; regularization cannot be negative")
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.steps = steps
        self.lr = lr
        self.regularization = regularization
        self.device = device

    def scan(self, images, class_count: int) -> dict:
        import numpy as np
        import torch
        import torch.nn.functional as F

        if class_count < 2 or images.ndim != 4 or len(images) == 0:
            raise ValueError("Expected a nonempty image batch and at least two classes")
        images = images.to(self.device)
        _, channels, height, width = images.shape
        results = []
        for target in range(class_count):
            mask_logits = torch.nn.Parameter(torch.zeros(1, 1, height, width, device=self.device))
            pattern_logits = torch.nn.Parameter(torch.zeros(1, channels, height, width, device=self.device))
            optimizer = torch.optim.Adam([mask_logits, pattern_logits], lr=self.lr)
            labels = torch.full((len(images),), target, dtype=torch.long, device=self.device)
            for _ in range(self.steps):
                mask = mask_logits.sigmoid()
                pattern = pattern_logits.tanh() * 3.0
                changed = images * (1 - mask) + pattern * mask
                logits = self.model(changed)
                loss = F.cross_entropy(logits, labels) + self.regularization * mask.mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            with torch.inference_mode():
                mask = mask_logits.sigmoid()
                pattern = pattern_logits.tanh() * 3.0
                predicted = self.model(images * (1 - mask) + pattern * mask).argmax(dim=1)
                results.append({"target": target, "mask_mean": float(mask.mean()),
                                "target_response_rate": float((predicted == target).float().mean())})
        norms = np.array([item["mask_mean"] for item in results])
        median = float(np.median(norms))
        mad = float(np.median(np.abs(norms - median)))
        for item in results:
            item["anomaly_index"] = float((median - item["mask_mean"]) / (1.4826 * mad + 1e-8))
        return {"targets": results, "median_mask_mean": median, "mad_mask_mean": mad,
                "steps": self.steps, "regularization": self.regularization,
                "samples": len(images)}


class FinePruner:
    """Rank low-activation channels at a named feature layer and mask them."""

    def __init__(self, model, layer_name: str, device: str = "cpu"):
        self.model = model.eval()
        self.layer_name = layer_name
        self.device = device
        modules = dict(model.named_modules())
        if layer_name not in modules:
            raise ValueError(f"Unknown layer {layer_name!r}; choose a name from model.named_modules()")
        self.layer = modules[layer_name]
        self.ranking = None
        self._mask_hook = None

    def rank(self, loader) -> list[int]:
        import torch

        totals = None
        samples = 0

        def capture(_module, _inputs, output):
            nonlocal totals, samples
            if not isinstance(output, torch.Tensor) or output.ndim != 4:
                raise ValueError("Fine pruning needs a feature layer with B,C,H,W output")
            activation = output.detach().abs().mean(dim=(0, 2, 3))
            totals = activation * len(output) if totals is None else totals + activation * len(output)
            samples += len(output)

        hook = self.layer.register_forward_hook(capture)
        try:
            with torch.inference_mode():
                for images, _ in loader:
                    self.model(images.to(self.device))
        finally:
            hook.remove()
        if totals is None or samples == 0:
            raise ValueError("Clean loader contained no images")
        self.ranking = torch.argsort(totals / samples).cpu().tolist()
        return self.ranking

    def apply(self, fraction: float) -> int:
        import torch

        if self.ranking is None:
            raise RuntimeError("Call rank before apply")
        if not 0 <= fraction < 1:
            raise ValueError("fraction must be in [0, 1)")
        self.remove()
        count = int(len(self.ranking) * fraction)
        channels = self.ranking[:count]

        def mask_output(_module, _inputs, output):
            mask = torch.ones(output.shape[1], device=output.device, dtype=output.dtype)
            mask[channels] = 0
            return output * mask.view(1, -1, 1, 1)

        self._mask_hook = self.layer.register_forward_hook(mask_output)
        return count

    def remove(self):
        if self._mask_hook is not None:
            self._mask_hook.remove()
            self._mask_hook = None


class CognitiveDistiller:
    """Find small image masks that preserve a classifier's original output."""

    def __init__(self, model, *, steps: int = 100, lr: float = 0.1,
                 gamma: float = 0.01, beta: float = 1.0, device: str = "cpu"):
        if steps < 1 or lr <= 0 or gamma < 0 or beta < 0:
            raise ValueError("Invalid optimization settings")
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.steps, self.lr, self.gamma, self.beta, self.device = steps, lr, gamma, beta, device

    def score_batch(self, images) -> list[float]:
        import torch
        import torch.nn.functional as F

        images = images.to(self.device)
        if images.ndim != 4 or not len(images):
            raise ValueError("Expected a nonempty B,C,H,W image batch")
        with torch.inference_mode():
            reference = self.model(images).detach()
        mask_logits = torch.nn.Parameter(torch.ones(len(images), 1, *images.shape[-2:], device=self.device))
        optimizer = torch.optim.Adam([mask_logits], lr=self.lr)
        baseline = F.avg_pool2d(images, kernel_size=7, stride=1, padding=3)
        for _ in range(self.steps):
            mask = mask_logits.sigmoid()
            changed = images * mask + baseline * (1 - mask)
            output = self.model(changed)
            preservation = (output - reference).abs().mean(dim=1)
            norm = mask.mean(dim=(1, 2, 3))
            tv = (mask[:, :, 1:, :] - mask[:, :, :-1, :]).abs().mean(dim=(1, 2, 3))
            tv += (mask[:, :, :, 1:] - mask[:, :, :, :-1]).abs().mean(dim=(1, 2, 3))
            loss = (preservation + self.gamma * norm + self.beta * tv).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return mask_logits.sigmoid().detach().mean(dim=(1, 2, 3)).cpu().tolist()


def spectral_high_frequency_score(image, cutoff: float = 0.25) -> float:
    """Fraction of non-DC Fourier energy outside a central frequency radius."""
    import numpy as np

    if not 0 < cutoff < 0.5:
        raise ValueError("cutoff must be in (0, 0.5)")
    pixels = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
    pixels = pixels - pixels.mean()
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(pixels))) ** 2
    height, width = pixels.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt(((yy - height // 2) / height) ** 2 + ((xx - width // 2) / width) ** 2)
    total = float(spectrum.sum())
    return float(spectrum[radius >= cutoff].sum() / total) if total > 0 else 0.0
