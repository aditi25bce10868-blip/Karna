
"""Controlled mono/stereo training for emergency audio classification.

Run from the repository root.

First run:
    python train_emergency_stereo.py --channels 1 --output artifacts/emergency_mono_control

Second run:
    python train_emergency_stereo.py --channels 2 --output artifacts/emergency_stereo

Both runs must use the same audio-path column, seed and training settings.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from datasets.emergency_audio_stereo import (
    EmergencyAudioDataset,
    build_records,
)

from models.small_cnn_stereo import SmallCNN


def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, class_names):

    model.eval()

    truth = []
    predictions = []

    with torch.no_grad():

        for x, y in loader:

            logits = model(x.to(device))

            predictions.extend(
                logits.argmax(1).cpu().tolist()
            )

            truth.extend(y.tolist())

    labels = list(range(len(class_names)))

    metrics = {

        "accuracy": accuracy_score(
            truth,
            predictions,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            truth,
            predictions,
        ),

        "macro_f1": f1_score(
            truth,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
    }

    report = classification_report(
        truth,
        predictions,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        digits=4,
    )

    matrix = confusion_matrix(
        truth,
        predictions,
        labels=labels,
    )

    return metrics, report, matrix


def make_loader(records, args, root, shuffle):

    dataset = EmergencyAudioDataset(
        records,
        str(root),
        target_sr=args.sample_rate,
        duration=args.duration,
        n_mels=args.n_mels,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        channels=args.channels,
    )

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default=".")

    parser.add_argument(
        "--manifest",
        default="emergency_datset/dataset_preparation/split_manifest.csv",
    )

    parser.add_argument(
        "--augmentation-manifest",
        default="emergency_datset/dataset_preparation/traffic_augmentation/traffic_augmentation_manifest.csv",
    )

    parser.add_argument(
        "--channels",
        type=int,
        choices=[1, 2],
        required=True,
    )

    parser.add_argument(
        "--audio-path-column",
        default="source_path",
        help="Column containing the original audio paths.",
    )

    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=8)

    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=3.0)

    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    set_seed(args.seed)

    root = Path(args.root).resolve()

    clean_path = root / args.manifest
    aug_path = root / args.augmentation_manifest

    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)

    if not clean_path.is_file():
        raise FileNotFoundError(
            f"Missing clean manifest: {clean_path}"
        )

    if not aug_path.is_file():
        raise FileNotFoundError(
            f"Missing augmentation manifest: {aug_path}"
        )

    clean = pd.read_csv(clean_path)
    aug = pd.read_csv(aug_path)

    required_clean = {
        args.audio_path_column,
        "class_name",
        "split",
        "include",
    }

    required_aug = {
        "augmented_path",
        "class_name",
        "split",
        "purpose",
    }

    if not required_clean.issubset(clean.columns):

        raise ValueError(
            "Required clean manifest columns missing.\n"
            f"Required: {sorted(required_clean)}\n"
            f"Found: {list(clean.columns)}"
        )

    if not required_aug.issubset(aug.columns):

        raise ValueError(
            "Required augmentation manifest columns missing.\n"
            f"Required: {sorted(required_aug)}\n"
            f"Found: {list(aug.columns)}"
        )

    clean = clean[
        clean["include"].astype(str).str.lower().isin(
            ["1", "true", "yes", "y"]
        )
    ]

    class_names = sorted(
        clean["class_name"].astype(str).unique().tolist()
    )

    if len(class_names) != 12:

        raise ValueError(
            f"Expected 12 classes, found {len(class_names)}"
        )

    class_to_index = {
        name: index
        for index, name in enumerate(class_names)
    }

    train_clean = clean[clean["split"] == "train"]
    val_clean = clean[clean["split"] == "validation"]
    test_clean = clean[clean["split"] == "test"]

    aug_train = aug[
        (aug["split"] == "train")
        & (aug["purpose"] == "training_augmentation")
    ]

    aug_val = aug[
        (aug["split"] == "validation")
        & (aug["purpose"] == "noisy_evaluation_only")
    ]

    aug_test = aug[
        (aug["split"] == "test")
        & (aug["purpose"] == "noisy_evaluation_only")
    ]

    train_records = (
        build_records(
            train_clean,
            args.audio_path_column,
            class_to_index,
        )
        +
        build_records(
            aug_train,
            "augmented_path",
            class_to_index,
        )
    )

    val_records = build_records(
        val_clean,
        args.audio_path_column,
        class_to_index,
    )

    test_records = build_records(
        test_clean,
        args.audio_path_column,
        class_to_index,
    )

    noisy_val_records = build_records(
        aug_val,
        "augmented_path",
        class_to_index,
    )

    noisy_test_records = build_records(
        aug_test,
        "augmented_path",
        class_to_index,
    )

    print("\n========== EXPERIMENT SUMMARY ==========")

    print("Input channels:", args.channels)
    print("Audio path column:", args.audio_path_column)
    print("Classes:", class_names)

    print("Training:", len(train_records))
    print("Clean validation:", len(val_records))
    print("Clean test:", len(test_records))
    print("Noisy validation:", len(noisy_val_records))
    print("Noisy test:", len(noisy_test_records))

    print(
        "\nMono recordings are duplicated into two channels "
        "during the stereo experiment."
    )

    if not train_records or not val_records or not test_records:

        raise ValueError(
            "Train, validation and test datasets must be non-empty."
        )

    train_loader = make_loader(
        train_records, args, root, True
    )

    val_loader = make_loader(
        val_records, args, root, False
    )

    test_loader = make_loader(
        test_records, args, root, False
    )

    noisy_val_loader = (
        make_loader(noisy_val_records, args, root, False)
        if noisy_val_records else None
    )

    noisy_test_loader = (
        make_loader(noisy_test_records, args, root, False)
        if noisy_test_records else None
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = SmallCNN(
        n_classes=len(class_names),
        in_channels=args.channels,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    criterion = torch.nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
    )

    best_f1 = -1.0
    stale_epochs = 0

    checkpoint_path = (
        output / f"best_emergency_{args.channels}ch_model.pt"
    )

    for epoch in range(1, args.epochs + 1):

        model.train()

        total_loss = 0.0
        correct = 0
        count = 0

        for x, y in train_loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)

            loss = criterion(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)

            correct += (
                logits.argmax(1) == y
            ).sum().item()

            count += y.size(0)

        val_metrics, _, _ = evaluate(
            model,
            val_loader,
            device,
            class_names,
        )

        scheduler.step(val_metrics["macro_f1"])

        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"| loss={total_loss/max(count, 1):.4f} "
            f"| train_acc={correct/max(count, 1):.4f} "
            f"| val_acc={val_metrics['accuracy']:.4f} "
            f"| val_bal_acc={val_metrics['balanced_accuracy']:.4f} "
            f"| val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_f1:

            best_f1 = val_metrics["macro_f1"]
            stale_epochs = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "smallcnn",
                    "input_channels": args.channels,
                    "class_names": class_names,
                    "sample_rate": args.sample_rate,
                    "duration": args.duration,
                    "n_mels": args.n_mels,
                    "n_fft": args.n_fft,
                    "hop_length": args.hop_length,
                    "audio_path_column": args.audio_path_column,
                    "best_validation_macro_f1": best_f1,
                },
                checkpoint_path,
            )

        else:

            stale_epochs += 1

            if stale_epochs >= args.patience:

                print(
                    "Early stopping: validation macro-F1 "
                    "did not improve."
                )

                break

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    with open(
        output / "class_map.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            {
                str(i): name
                for i, name in enumerate(class_names)
            },
            file,
            indent=2,
        )

    evaluations = [
        ("clean_validation", val_loader),
        ("clean_test", test_loader),
        ("noisy_validation", noisy_val_loader),
        ("noisy_test", noisy_test_loader),
    ]

    for name, loader in evaluations:

        if loader is None:
            continue

        metrics, report, matrix = evaluate(
            model,
            loader,
            device,
            class_names,
        )

        print(f"\n========== {name.upper()} ==========")

        for key, value in metrics.items():
            print(f"{key}: {value:.4f}")

        print(report)

        pd.DataFrame(
            matrix,
            index=class_names,
            columns=class_names,
        ).to_csv(
            output / f"{name}_confusion_matrix.csv"
        )

        with open(
            output / f"{name}_report.txt",
            "w",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(metrics, indent=2)
                + "\n\n"
                + report
            )

    print("\nBest checkpoint:", checkpoint_path)
    print("Training complete.")


if __name__ == "__main__":
    main()
