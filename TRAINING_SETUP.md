# MVT tracking: GOT-10k local training package

This drop-in package adds the dataset/indexing, causal sampler, RGB pair augmentations, one-device trainer, CLI and checkpoint resume missing from the `main` branch of `RJKRanathunga/mobile-vision-transformer` as reviewed on 2026-11-16. Existing model/loss/optimizer/checkpoint code remains untouched.

## 1. Place the files

**Merge** the contents of this package into the root of your local `mobile-vision-transformer` checkout. Do **not** replace the entire `src/mvt/data` or `src/mvt/training` directories; only copy the named new files. See `FILE_MANIFEST.txt`.

Your current extracted data should be:

```text
mobile-vision-transformer/
  data/
    train/
      list.txt                         # optional if sequence folders can be discovered
      GOT-10k_Train_000001/
        00000001.jpg
        00000002.jpg
        groundtruth.txt               # comma-separated x,y,width,height per frame
        absence.label                 # optional; 1=absent
        cover.label                   # optional; >0=visible
        meta_info.ini                 # optional
      ...
    val/
      list.txt                         # optional
      GOT-10k_Val_000001/
        00000001.jpg
        groundtruth.txt
      ...
    test/                              # reserved for future tracking benchmark evaluation
```

If your dataset instead has extra nested subdirectories (`data/train/train/GOT...`), adjust the paths with `--train-root` etc. The split roots supplied to the CLI should contain **sequence folders directly**, not ZIP/TAR archives. If `list.txt` exists, it must list the sequence folder names. Frames are sorted by numeric filename.

The loader requires annotated train/val sequences and ignores frames with invalid boxes, `absence.label == 1`, or `cover.label == 0`. It samples a visible template frame before a later visible search frame within the configured max interval (200). Unlabelled GOT-10k test data **cannot** be used for supervised validation or training.

## 2. Install environment

Use Python 3.11 or newer from the repository root. For NVIDIA GPUs, first install a CUDA-enabled PyTorch build compatible with your NVIDIA driver, using https://pytorch.org/get-started/locally/ . Then:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/inspect_got10k.py --data-root data
python -m pytest -q tests/test_training_pipeline.py
```

**Windows PowerShell:** activate with `.venv\Scripts\Activate.ps1` if using a virtual environment. On Linux/macOS: `source .venv/bin/activate`. The GPU version of PyTorch is an environment choice, not provided by this ZIP.

## 3. Smoke test first

```bash
python train.py --smoke-test --batch-size 2 --num-workers 0 --device auto
```

This indexes both train and val and runs a few training/validation pairs. A CPU run can be slow; an NVIDIA GPU is recommended.

## 4. Start training

Memory-conscious single GPU (effective batch 128 via accumulation; the extra microbatches make this much slower than a real 128-image batch):

```bash
python train.py --batch-size 8 --grad-accum 16 --num-workers 4 --amp --device cuda
```

For the original experiment's actual batch setting, use `--batch-size 128 --grad-accum 1` only if your device(s) can fit it. **This implementation is single-device**; no DDP/multi-GPU code is included. The default config is `configs/train.yaml`: GOT-10k, 100 epochs, 60,000 train pairs and 10,000 validation pairs per epoch, AdamW 4e-4, backbone multiplier 0.1, cosine annealing. Validation runs every 10 epochs by default and at the final epoch. Turn on AMP with `--amp` when CUDA is available; config defaults to AMP off.

If you cannot fit batch 8, try `--batch-size 2 --grad-accum 64 --num-workers 0 --amp` instead. Accumulation approximates larger-batch optimization but does **not** reproduce batch-normalization statistics of an actual large batch.

### Optional pretrained initialization

Training with no flags starts with randomly initialized weights. The original MVT experiment initializes its backbone from a MobileViT-S classification checkpoint (`mobilevit_s.pt`) **that is not in your repository**. Obtain a checkpoint from the original MVT project authors or their documented upstream MobileViT/ml-cvnets source, verify its provenance and tensor shapes, place it at `pretrained_models/mobilevit_s.pt`, and use:

```bash
python train.py --pretrained-backbone pretrained_models/mobilevit_s.pt --batch-size 8 --grad-accum 16 --num-workers 4 --amp --device cuda
```

A complete native MVT checkpoint or a checkpoint from the audited pinned upstream MVT implementation can instead be used to **fine-tune** an existing tracker:

```bash
python train.py --pretrained-tracker /path/to/known-compatible-tracker.pth.tar --batch-size 8 --grad-accum 16 --amp --device cuda
```

Compatibility is strict: tensor count and shapes are validated. A random MobileViT checkpoint from another version may not work. The official upstream repo https://github.com/goutamyg/MVT links its released pretrained tracker weights in its README. Downloading datasets/checkpoints and license checks remain the user's responsibility.

## 5. Checkpoints, metrics and resume

```text
checkpoints/mvt_got10k/
  latest.pth       # model + optimizer + scheduler (+ AMP scaler) every epoch
  best.pth         # best validation loss among evaluated epochs
  epoch_0010.pth  # periodic state, every 10 epochs by default
  metrics.jsonl    # one JSON object per epoch
```

```bash
python train.py --resume checkpoints/mvt_got10k/latest.pth --batch-size 8 --grad-accum 16 --num-workers 4 --amp --device cuda
```

When resuming, keep the **same batch size, accumulation, schedule length, and dataset policy**; use the original `--epochs` count. Checkpoints produced by this package store state under `model`; checkpoints loaded for pretrained initialization can use upstream `net`. Only load checkpoints you trust (PyTorch's serialization may execute code).

The saved `best.pth` is selected using **pairwise validation loss**, not GOT-10k's official AO/SR tracking metrics. A full per-sequence benchmark evaluator remains to be implemented. `data/test` is therefore not used; do not train on it.

## 6. What is still not included

- GPU drivers, CUDA-enabled PyTorch wheels, dataset archives or third-party weight files.
- Exact upstream distributed training/DDP and official GOT-10k test/evaluation launchers.
- Benchmark AO/SR and full-sequence tracking metrics (the existing `MVTTracker` provides an inference primitive, not a complete benchmark harness).
- Tracking runs verified on your actual machine. The included unit tests rely on a **synthetic annotated dataset**; check real annotations with the inspection command and smoke test before full training.

### Troubleshooting

- **No usable GOT-10k sequences**: verify root and `groundtruth.txt` lines per sequence, boxes are `x,y,w,h`, and there are at least two visible frames within 200 frames.
- **CUDA out of memory**: reduce `--batch-size`, then increase `--grad-accum`; try AMP.
- **Wrong PyTorch CUDA build**: `python -c "import torch; print(torch.__version__,torch.cuda.is_available())"`.
- **Validation seems inconsistent**: the supplied validation pair sampling and crop jitter are deterministic by index, but validation loss is a proxy for sequence-level tracking quality.
- **Missing pretrained checkpoint**: optional; omit `--pretrained-backbone` to train from random initialization (not comparable with the original experiment).

No repository files were changed remotely by creating this package.
