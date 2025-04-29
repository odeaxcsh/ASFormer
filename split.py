import os
import argparse
import random
from pathlib import Path

def split_ground_truth(data_dir, train_ratio=0.9, split=0):
    gt_dir = Path(data_dir) / "features"
    split_dir = Path(data_dir) / "splits"
    split_dir.mkdir(exist_ok=True)

    all_files = sorted([f.name for f in gt_dir.glob("*.npy")])
    # replace npy extion with txt
    all_files = [f.replace(".npy", ".txt") for f in all_files]

    split_idx = int(len(all_files) * (1 - train_ratio))
    train_files = all_files[split_idx:]
    test_files = all_files[:split_idx]

    train_path = split_dir / f"train.split{split}.bundle"
    test_path = split_dir / "test.split{split}.bundle"

    with open(train_path, "w") as f:
        for name in train_files:
            f.write(name + "\n")

    with open(test_path, "w") as f:
        for name in test_files:
            f.write(name + "\n")

    print(f"[+] Wrote {len(train_files)} training files to {train_path}")
    print(f"[+] Wrote {len(test_files)} testing files to {test_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to dataset root (should contain groundTruth/)", default="data/Ours/")
    parser.add_argument("--ratio", type=float, default=0.9, help="Ratio of training data")
    parser.add_argument("--split", type=int, default="split", help="Split number")
    args = parser.parse_args()

    split_ground_truth(args.data, args.ratio, args.split)
