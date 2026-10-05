"""Train an ImageFolder classifier; keep the experiment loop visible."""

import argparse
import json
from pathlib import Path

from triggercraft.models import load_classifier
from triggercraft.utils import parse_experiment_args, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Root containing train/ and val/ class folders")
    parser.add_argument("--model", default="resnet18", help="Any timm classifier name")
    parser.add_argument("--backend", choices=["timm", "torchvision"], default="timm")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--schedule", choices=["none", "cosine"], default="none")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output", type=Path, default=Path("results/baseline"))
    args = parse_experiment_args(parser, "train")

    import torch
    from torchvision.datasets import ImageFolder
    from torch.utils.data import DataLoader

    seed_everything(args.seed)
    data = Path(args.data)
    train_root, val_root = data / "train", data / "val"
    classes = ImageFolder(train_root).classes
    classifier = load_classifier(args.model, len(classes), backend=args.backend,
                                 pretrained=args.pretrained, input_size=args.input_size,
                                 device=args.device)
    model = classifier.model
    train_data = ImageFolder(train_root, transform=classifier.train_transform)
    val_data = ImageFolder(val_root, transform=classifier.val_transform)
    if train_data.classes != val_data.classes:
        raise ValueError("train and val must contain the same class folders")
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
                 if args.schedule == "cosine" else None)
    args.output.mkdir(parents=True, exist_ok=True)
    best_accuracy = -1.0

    for epoch in range(args.epochs):
        epoch_lr = optimizer.param_groups[0]["lr"]
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)

        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                predicted = model(images.to(args.device)).argmax(dim=1).cpu()
                correct += (predicted == labels).sum().item()
                total += len(labels)
        val_accuracy = correct / total
        record = {"epoch": epoch + 1, "lr": epoch_lr,
                  "train_loss": total_loss / len(train_data),
                  "val_accuracy": val_accuracy}
        print(json.dumps(record), flush=True)
        if scheduler is not None:
            scheduler.step()
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), args.output / "best_model.pth")

    (args.output / "run.json").write_text(json.dumps({
        "model": args.model, "backend": args.backend, "classes": classes, "data": str(data.resolve()),
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "learning_rate": args.lr, "optimizer": args.optimizer,
        "momentum": args.momentum, "weight_decay": args.weight_decay,
        "schedule": args.schedule, "input_size": args.input_size,
        "best_val_accuracy": best_accuracy}, indent=2) + "\n")


if __name__ == "__main__":
    main()
