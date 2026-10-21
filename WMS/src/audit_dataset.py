"""Verify every pair through the project's preprocessing and export KM1 evidence.

Run from the repository root: python WMS/src/audit_dataset.py
Original files are copied into the same ignored split directories as prepareDataset.py.
Preprocessed tensors are checked in memory, as they are when loaded for training.
No model is loaded or trained.
"""

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "wms-matplotlib"))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import torch
import torchvision

from dataset import WMSDataset
from transforms import valTransforms


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "WMS/data/training"
OUT = ROOT / "Results/KM1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    names = sorted(p.name for p in (DATA / "images").glob("*.jpg"))
    mask_names = sorted(p.name for p in (DATA / "masks").glob("*.jpg"))
    require(names and names == mask_names, "Missing images or unmatched mask filenames")

    # Identical sorting, split proportions and seeds to prepareDataset.py.
    train, remaining = train_test_split(names, test_size=0.2, random_state=42)
    val, test = train_test_split(remaining, test_size=0.5, random_state=42)
    splits = {"train": train, "val": val, "test": test}
    require(len(set(train) | set(val) | set(test)) == len(names), "Incomplete split")
    require(not (set(train) & set(val) or set(train) & set(test) or set(val) & set(test)),
            "Overlapping split filenames")

    OUT.mkdir(parents=True, exist_ok=True)
    rows, examples, statistics = [], [], {}
    hashes = {}
    for split, filenames in splits.items():
        split_root = DATA / "temp" / split
        for kind in ("images", "masks"):
            destination = split_root / kind
            destination.mkdir(parents=True, exist_ok=True)
            extra = {p.name for p in destination.glob("*.jpg")} - set(filenames)
            require(not extra, f"Unexpected files in {destination}: {sorted(extra)}")
            for name in filenames:
                shutil.copy2(DATA / kind / name, destination / name)

        ordered = sorted(filenames)
        dataset = WMSDataset(
            [str(split_root / "images" / n) for n in ordered],
            [str(split_root / "masks" / n) for n in ordered],
            imageTransforms=valTransforms,
        )
        split_rows = []
        # One deterministic example from each split, without using model predictions.
        example_name = filenames[0]
        for index, name in enumerate(ordered):
            image_path, mask_path = DATA / "images" / name, DATA / "masks" / name
            original = cv2.imread(str(image_path))
            original_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            require(original is not None and original_mask is not None, f"Unreadable: {name}")
            require(original.shape[:2] == original_mask.shape, f"Source size mismatch: {name}")

            image, mask = dataset[index]
            require(image.shape == (3, 512, 512) and mask.shape == (1, 512, 512),
                    f"Invalid tensor shape: {name}")
            require(image.dtype == mask.dtype == torch.float32, f"Invalid dtype: {name}")
            pixels, labels = image.numpy(), mask.numpy()
            require(np.isfinite(pixels).all() and np.isfinite(labels).all(),
                    f"Non-finite values: {name}")
            require(0 <= pixels.min() <= pixels.max() <= 1, f"Image range error: {name}")
            require(np.isin(labels, [0, 1]).all(), f"Non-binary mask: {name}")
            foreground = int(labels.sum())
            source_hash = sha256(image_path)
            hashes.setdefault(source_hash, []).append((split, name))
            row = {
                "split": split, "filename": name,
                "source_width": original.shape[1], "source_height": original.shape[0],
                "foreground_pixels_512": foreground,
                "background_pixels_512": 512 * 512 - foreground,
                "foreground_percent_512": 100 * foreground / (512 * 512),
                "source_image_sha256": source_hash,
                "source_mask_sha256": sha256(mask_path),
                "processed_image_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "processed_mask_sha256": hashlib.sha256(labels.tobytes()).hexdigest(),
            }
            split_rows.append(row)
            if name == example_name:
                examples.append((split, name, pixels.transpose(1, 2, 0), labels[0]))
            if (index + 1) % 100 == 0:
                print(f"{split}: {index + 1}/{len(ordered)} verified", flush=True)

        rows.extend(split_rows)
        fg = sum(r["foreground_pixels_512"] for r in split_rows)
        total = len(ordered) * 512 * 512
        statistics[split] = {
            "pairs": len(ordered), "dataset_percent": 100 * len(ordered) / len(names),
            "foreground_pixels": fg, "background_pixels": total - fg,
            "foreground_percent": 100 * fg / total,
            "background_percent": 100 * (total - fg) / total,
            "empty_masks": sum(r["foreground_pixels_512"] == 0 for r in split_rows),
            "foreground_percent_min": min(r["foreground_percent_512"] for r in split_rows),
            "foreground_percent_max": max(r["foreground_percent_512"] for r in split_rows),
        }
        print(f"{split}: {len(ordered)} pairs OK; foreground {100 * fg / total:.4f}%", flush=True)

    duplicate_groups = [v for v in hashes.values() if len(v) > 1]
    cross_split_duplicates = [v for v in duplicate_groups if len({s for s, _ in v}) > 1]
    require(not cross_split_duplicates, f"Byte-identical images in different splits: {cross_split_duplicates}")
    summary = {
        "verified_pairs": len(rows), "failed_pairs": 0, "split_seed": 42,
        "preprocessing": "WMSDataset(imageTransforms=valTransforms), no random augmentation",
        "storage": "Source pairs in training/temp/{split}/{images,masks}; tensors processed on load",
        "image_shape": [3, 512, 512], "mask_shape": [1, 512, 512],
        "dtype": "float32", "image_range": [0, 1], "mask_values": [0, 1],
        "source_width_range": [min(r["source_width"] for r in rows), max(r["source_width"] for r in rows)],
        "source_height_range": [min(r["source_height"] for r in rows), max(r["source_height"] for r in rows)],
        "empty_masks": sum(s["empty_masks"] for s in statistics.values()),
        "byte_identical_image_groups": duplicate_groups,
        "cross_split_byte_identical_groups": cross_split_duplicates,
        "splits": statistics,
        "examples": [{"split": s, "filename": n} for s, n, _, _ in examples],
        "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__,
                     "numpy": np.__version__, "opencv": cv2.__version__, "sklearn": sklearn.__version__},
        "source_code_sha256": {n: sha256(Path(__file__).parent / n)
                               for n in ("dataset.py", "transforms.py", "prepareDataset.py", "audit_dataset.py")},
    }
    with (OUT / "split_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "dataset_statistics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    titles = {"train": "Treningowy", "val": "Walidacyjny", "test": "Testowy"}
    fig, axes = plt.subplots(3, 2, figsize=(7.0, 9.2), layout="constrained")
    for row_index, (split, name, image, mask) in enumerate(examples):
        axes[row_index, 0].imshow(image)
        axes[row_index, 1].imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axes[row_index, 0].set_title(f"{titles[split]} — obraz po preprocessingu\n{name}", fontsize=9)
        axes[row_index, 1].set_title("Odpowiadająca maska binarna", fontsize=9)
        for ax in axes[row_index]:
            ax.axis("off")
    fig.savefig(OUT / "preprocessing_examples.png", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
