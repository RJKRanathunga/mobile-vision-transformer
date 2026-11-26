# MVT virtual tripwire

This application adds a causal, single-target event detector after MVT inference. It does not alter training, model weights or benchmark code. Use trusted full tracker checkpoints only (the existing checkpoint loader uses PyTorch deserialization).

## Install and test

From the project root, in the existing working environment:

```sh
python -m pip install -e . --no-deps
python -m pytest tests/test_tripwire.py -v
```

## Quick algorithm-only demo

```sh
python demo_tripwire.py --synthetic
```

This uses synthetic ground-truth boxes, not MVT predictions. It should report two crossings with the default line and margin. Add `--browser-video` if ffmpeg with libx264 is available.

## Actual MVT demo (GOT-10k)

```sh
python -u demo_tripwire.py --weights pretrained_models/MobileViT_Track_ep0100_weights_only.pth --sequence data/test/GOT-10k_Test_000001 --device auto --line 0.5 0 0.5 1 --margin 10 --fps 15
```

The first row of the sequence's groundtruth.txt is used only for initialization. If unavailable, supply `--init-box X Y W H` (pixel XYWH) or `--select-roi` (desktop GUI). Add `--display` for desktop preview; Q stops the run. Image sequences are read from the sequence directory or its img/ subdirectory. Select a line before inference, and do not assume the target will cross it.

## Actual MVT demo (video)

```sh
python -u demo_tripwire.py --weights pretrained_models/MobileViT_Track_ep0100_weights_only.pth --video sample.mp4 --select-roi --display --device auto --line 0.5 0 0.5 1
```

Select the target on the first frame and press Enter. No automatic object detection or multi-object tracking is implemented. For headless environments use explicit `--init-box` and omit `--display` and `--select-roi`.

## Outputs

Each run writes to output/novelty_demo/<source-name>/ (or --output-dir). Rerunning the same source in the same output directory replaces its outputs. The MP4 contains the line, bounding box, trajectory and alerts; events.csv logs direction, confirmation frame and nominal time; predictions.txt stores XYWH boxes; summary.json records settings and measured timing. `--browser-video` additionally produces tripwire_browser.mp4 for notebook playback. Sequence FPS defaults to 15 (playback assumption), video FPS defaults to the source's reported FPS; nominal times are not exact timestamps for variable-frame-rate videos. Tracker FPS excludes frame acquisition, conversion and rendering; processing FPS includes these and encoding but excludes initialization and optional H.264 conversion. Synthetic tracker FPS is null.

## Detection and evaluation

The two normalized endpoints define an infinite line, clipped visually by the image. Signed distance is positive on one side and negative on the other. The dead band is ±margin pixels. The stable side changes only when the centroid moves beyond the opposite margin; an invalid box or frame gap resets the side. A confirmation may lag the physical crossing; a tracker jump can trigger a false alarm. No confidence gating or world-coordinate calibration is implemented. The line is fixed in image space, so camera motion can produce apparent crossings.

For research, use validation videos with independently annotated crossing events, predefine boundaries, and compare margin 0 to nonzero margins. Report one-to-one matched event precision/recall/F1, false alarms, latency, and computational overhead. The synthetic demo verifies behavior, not MVT accuracy. Full GOT-10k test trajectories are not publicly annotated.
