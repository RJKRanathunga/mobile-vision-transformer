
"""Evaluate the MVT object tracker on GOT-10k."""

import argparse
from pathlib import Path

import numpy as np
import torch

from got10k.trackers import Tracker
from got10k.datasets import GOT10k
from got10k.experiments import ExperimentGOT10k

from mvt.models.mvt import MVT
from mvt.evaluation.tracker import MVTTracker
from mvt.training.pretrained import load_training_weights


class MVTAdapter(Tracker):
    """Connect the local MVTTracker to the GOT-10k toolkit."""

    def __init__(self, engine, name="MVT"):
        super().__init__(
            name=name,
            is_deterministic=True,
        )
        self.engine = engine

    def init(self, image, box):
        """Initialize tracking using the first frame and GT box."""
        rgb = np.asarray(image.convert("RGB"))

        self.engine.initialize(
            rgb,
            [float(v) for v in box],
        )

    def update(self, image):
        """Predict the target bounding box in the next frame."""
        rgb = np.asarray(image.convert("RGB"))

        result = self.engine.track(rgb)

        return np.asarray(
            result["target_bbox"],
            dtype=np.float64,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MVT on the GOT-10k dataset"
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the model checkpoint",
    )

    parser.add_argument(
        "--subset",
        choices=["val", "test"],
        default="val",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
    )

    parser.add_argument(
        "--name",
        type=str,
        default="MVT",
    )

    parser.add_argument(
        "--list-file",
        type=str,
        default=None,
        help="Optional text file listing sequences to evaluate",
    )

    parser.add_argument(
        "--backbone-only",
        action="store_true",
        help="Load only pretrained MobileViT-S backbone weights",
    )

    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # 1. Select device
    # ---------------------------------------------------------

    if args.device == "auto":
        device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    else:
        device = args.device

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print(f"Using device: {device}")

    # ---------------------------------------------------------
    # 2. Verify checkpoint
    # ---------------------------------------------------------

    weights_path = Path(args.weights)

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}"
        )

    # ---------------------------------------------------------
    # 3. Build model and load weights
    # ---------------------------------------------------------

    print("Initializing MVT model...")

    model = MVT()

    print("Loading checkpoint...")

    report = load_training_weights(
        model,
        weights_path,
        backbone_only=args.backbone_only,
    )

    print(report)

    if args.backbone_only:
        print(
            "WARNING: Only backbone weights were loaded. "
            "The tracking head remains untrained, so "
            "tracking accuracy is not meaningful."
        )

    # ---------------------------------------------------------
    # 4. Create stateful tracker
    # ---------------------------------------------------------

    model = model.to(device)
    model.eval()

    engine = MVTTracker(
        model=model,
        device=device,
    )

    tracker = MVTAdapter(
        engine=engine,
        name=args.name,
    )

    # ---------------------------------------------------------
    # 5. Create GOT-10k experiment
    # ---------------------------------------------------------

    print(f"Dataset root: {args.data_root}")
    print(f"Dataset subset: {args.subset}")

    experiment = ExperimentGOT10k(
        args.data_root,
        subset=args.subset,
        result_dir="results",
        report_dir="reports",
    )

    # Filter sequences using the existing dataset instance.
    # Compatible with GOT-10k versions without list_file support.

    if args.list_file:
        list_path = Path(args.list_file)

        if not list_path.is_file():
            raise FileNotFoundError(
                f"Sequence list not found: {list_path}"
            )

        selected_names = [
            line.strip()
            for line in list_path.read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]

        dataset = experiment.dataset

        original_names = list(dataset.seq_names)

        selected_indices = [
            i
            for i, name in enumerate(original_names)
            if name in selected_names
        ]

        if not selected_indices:
            raise ValueError(
                f"No matching sequences found: {selected_names}"
            )

        missing = set(selected_names) - set(original_names)

        if missing:
            raise ValueError(
                f"Unknown sequence names: {sorted(missing)}"
            )

        # Filter the dataset's internal sequence lists.
        dataset.seq_names = [
            original_names[i]
            for i in selected_indices
        ]

        for attribute in ("seq_dirs", "anno_files"):
            values = getattr(dataset, attribute)
            setattr(
                dataset,
                attribute,
                [values[i] for i in selected_indices],
            )

        print(
            f"Selected {len(dataset)} sequence(s)"
        )

    # ---------------------------------------------------------
    # 6. Run tracking inference
    # ---------------------------------------------------------

    print("\nStarting GOT-10k evaluation...\n")

    experiment.run(
        tracker,
        visualize=False,
    )

    # ---------------------------------------------------------
    # 7. Calculate benchmark metrics
    # ---------------------------------------------------------

    print("\nGenerating evaluation report...\n")

    results = experiment.report([tracker.name])

    if args.subset == "val" and results is not None:
        overall = results[tracker.name]["overall"]

        print("\n" + "=" * 50)
        print("MVT TRACKING EVALUATION RESULTS")
        print("=" * 50)

        print(f"Model: {tracker.name}")
        print(f"Checkpoint: {weights_path}")
        print(f"Device: {device}")

        print(
            f"Average Overlap (AO): "
            f"{overall.get('ao', float('nan')):.4f}"
        )

        print(
            f"Success Rate (SR): "
            f"{overall.get('sr', {})}"
        )

        print(
            f"Tracking speed (FPS): "
            f"{overall.get('speed_fps', float('nan')):.2f}"
        )

        print("=" * 50)

    else:
        print(
            "Test predictions generated. "
            "Official test scoring requires submission "
            "to the GOT-10k evaluation server."
        )

    print("\nEvaluation completed successfully.")
    print("Tracking results: results/GOT-10k/")
    print("Evaluation reports: reports/GOT-10k/")


if __name__ == "__main__":
    main()
