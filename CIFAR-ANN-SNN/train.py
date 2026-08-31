"""Train TinyCifarNet on CIFAR-10 (standard recipe, no heavy HPO)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from export import export_bundle
from local_cifar import LocalCifar10
from model import TinyCifarNet


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

DEFAULT_LOCAL_IMAGES = Path(__file__).resolve().parents[1] / "CIFAR" / "Workplace" / "CIFAR10.bin"
DEFAULT_LOCAL_LABELS = Path(__file__).resolve().parents[1] / "CIFAR" / "Workplace" / "CIFAR10.target.txt"


def build_loaders(
    batch_size: int,
    num_workers: int,
    data_dir: Path | None = None,
    local_images: Path | None = None,
    local_labels: Path | None = None,
    channel_order: str = "hwc",
):
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    test_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    images_path = local_images or DEFAULT_LOCAL_IMAGES
    labels_path = local_labels or DEFAULT_LOCAL_LABELS
    if images_path.is_file() and labels_path.is_file():
        train_set = LocalCifar10(images_path, labels_path, train=True, transform=train_tf, channel_order=channel_order)
        test_set = LocalCifar10(images_path, labels_path, train=False, transform=test_tf, channel_order=channel_order)
        print(f"using local CIFAR-10: {images_path}")
    else:
        if data_dir is None:
            data_dir = Path("data")
        train_set = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=train_tf)
        test_set = datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=test_tf)
        print(f"using torchvision CIFAR-10 under {data_dir}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        pred = logits.argmax(dim=1)
        correct += (pred == targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / total


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running = 0.0
    n = 0
    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running += loss.item() * targets.size(0)
        n += targets.size(0)
    return running / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--local-images", type=Path, default=DEFAULT_LOCAL_IMAGES)
    parser.add_argument("--local-labels", type=Path, default=DEFAULT_LOCAL_LABELS)
    parser.add_argument("--channel-order", choices=("hwc", "chw"), default="chw")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = TinyCifarNet().to(device)
    print(f"device={device}")
    print(f"trainable_params(with BN)={model.count_trainable_parameters()}")
    print(f"folded_params(estimate)={model.count_folded_parameters()}")
    if model.count_folded_parameters() > 30000:
        raise SystemExit("Folded parameter budget exceeded")

    train_loader, test_loader = build_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        data_dir=args.data_dir,
        local_images=args.local_images,
        local_labels=args.local_labels,
        channel_order=args.channel_order,
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_acc = -1.0
    best_path = args.out_dir / "best.pt"
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "loss": round(loss, 5),
            "test_acc": round(acc, 3),
            "lr": scheduler.get_last_lr()[0],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d}/{args.epochs}  loss={loss:.4f}  "
            f"test_acc={acc:.2f}%  lr={row['lr']:.5f}"
        )

        if acc > best_acc:
            best_acc = acc
            metrics = {
                "best_test_accuracy_pct": round(best_acc, 3),
                "best_epoch": epoch,
                "folded_params": model.count_folded_parameters(),
                "trainable_params_with_bn": model.count_trainable_parameters(),
                "epochs": args.epochs,
                "seed": args.seed,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "metrics": metrics,
                    "args": vars(args),
                },
                best_path,
            )

    elapsed = time.time() - t0
    (args.out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    final_metrics = {
        "best_test_accuracy_pct": round(best_acc, 3),
        "folded_params": model.count_folded_parameters(),
        "trainable_params_with_bn": model.count_trainable_parameters(),
        "epochs": args.epochs,
        "device": str(device),
        "seconds": round(elapsed, 1),
        "seed": args.seed,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    summary = export_bundle(best_path, args.out_dir, metrics=final_metrics)
    print("done:", json.dumps({**final_metrics, "export": summary}, indent=2))


if __name__ == "__main__":
    main()
