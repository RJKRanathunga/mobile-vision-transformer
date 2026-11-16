"""Inspect extracted GOT-10k train/val/test layout; no GPU required."""
from __future__ import annotations

import argparse

from mvt.data.got10k import inspect_split


def main():
    parser = argparse.ArgumentParser(description="Inspect GOT-10k split layout")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    for split in ("train", "val", "test"):
        root = f"{args.data_root}/{split}"
        try:
            info = inspect_split(root)
        except Exception as exc:
            print(f"{split}: ERROR: {exc}")
            continue
        print(f"{split}: {info}")
        if split in ("train", "val") and info.get("annotated", 0) == 0:
            print(f"  WARNING: No groundtruth.txt files in {root}. Check extraction and split structure.")
    print("Test split is held out and must not be used to fit or tune the network.")


if __name__ == "__main__":
    main()
