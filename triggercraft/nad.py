"""Neural Attention Distillation using named feature layers."""

from copy import deepcopy


def attention_map(features, power: float = 2.0):
    import torch

    if features.ndim != 4:
        raise ValueError("Attention transfer needs B,C,H,W feature maps")
    attention = features.abs().pow(power).sum(dim=1)
    return attention / torch.linalg.vector_norm(attention.flatten(1), dim=1).view(-1, 1, 1).clamp_min(1e-8)


class AttentionDistiller:
    """Fine-tune a teacher on clean data, then transfer its attention to a student."""

    def __init__(self, student, layer_names: list[str], *, beta: float = 1.0,
                 power: float = 2.0, device: str = "cpu"):
        if not layer_names or beta < 0 or power <= 0:
            raise ValueError("Provide feature layers, nonnegative beta, and positive power")
        modules = dict(student.named_modules())
        missing = [name for name in layer_names if name not in modules]
        if missing:
            raise ValueError(f"Unknown layers: {missing}")
        self.student = student.to(device)
        self.teacher = deepcopy(student).to(device)
        self.layer_names = layer_names
        self.beta = beta
        self.power = power
        self.device = device

    def fit(self, loader, *, teacher_epochs: int = 1, student_epochs: int = 1,
            lr: float = 0.001) -> list[dict]:
        import torch
        import torch.nn.functional as F

        if teacher_epochs < 1 or student_epochs < 1 or lr <= 0:
            raise ValueError("Epochs and learning rate must be positive")
        history = []
        teacher_optimizer = torch.optim.AdamW(self.teacher.parameters(), lr=lr)
        for epoch in range(teacher_epochs):
            self.teacher.train()
            total_loss = total = 0
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                loss = F.cross_entropy(self.teacher(images), labels)
                teacher_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                teacher_optimizer.step()
                total_loss += loss.item() * len(labels)
                total += len(labels)
            history.append({"stage": "teacher", "epoch": epoch + 1,
                            "loss": total_loss / total})

        self.teacher.eval().requires_grad_(False)
        student_features, teacher_features = {}, {}
        hooks = []
        for model, container in ((self.student, student_features), (self.teacher, teacher_features)):
            modules = dict(model.named_modules())
            for name in self.layer_names:
                hooks.append(modules[name].register_forward_hook(
                    lambda _module, _inputs, output, key=name, destination=container:
                    destination.__setitem__(key, output)))
        student_optimizer = torch.optim.AdamW(self.student.parameters(), lr=lr)
        try:
            for epoch in range(student_epochs):
                self.student.train()
                total_loss = total = 0
                for images, labels in loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    student_features.clear()
                    teacher_features.clear()
                    logits = self.student(images)
                    with torch.no_grad():
                        self.teacher(images)
                    transfer = sum(F.mse_loss(attention_map(student_features[name], self.power),
                                              attention_map(teacher_features[name], self.power))
                                   for name in self.layer_names) / len(self.layer_names)
                    loss = F.cross_entropy(logits, labels) + self.beta * transfer
                    student_optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    student_optimizer.step()
                    total_loss += loss.item() * len(labels)
                    total += len(labels)
                history.append({"stage": "student", "epoch": epoch + 1,
                                "loss": total_loss / total})
        finally:
            for hook in hooks:
                hook.remove()
        return history
