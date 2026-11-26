"""Normal MVT single-object tracking demo on GOT-10k validation sequences.

Run from the repository root. No tripwire module is required.
"""
import argparse
import csv
import math
import re
import time
from pathlib import Path

import cv2
import numpy as np

GT = (0, 255, 0)       # green (BGR)
PRED = (255, 140, 0)   # blue (BGR)
WINDOW = 'MVT - Normal Object Tracking'


def valid(box):
    box = np.asarray(box)
    return box.shape == (4,) and np.isfinite(box).all() and (box[2:] > 0).all()


def overlap(a, b):
    if not valid(b):
        return None
    if not valid(a):
        return 0.0
    a, b = np.asarray(a), np.asarray(b)
    intersection = np.maximum(0, np.minimum(a[:2] + a[2:], b[:2] + b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(intersection / (a[2:].prod() + b[2:].prod() - intersection))


def load_gt(path):
    rows = [[float(v) for v in re.split(r'[,\s]+', line.strip())]
            for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    boxes = np.asarray(rows, dtype=float)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(f'Expected Nx4 XYWH annotations: {path}')
    return boxes


def natural_key(path):
    return tuple((0, int(s)) if s.isdigit() else (1, s.lower())
                 for s in re.split(r'(\d+)', path.name))


def read_frame(path):
    frame = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f'Cannot read image: {path}')
    return frame


def draw_box(frame, box, color, label, thickness):
    if not valid(box):
        return
    x, y, w, h = box
    H, W = frame.shape[:2]
    x1, x2 = [int(round(np.clip(v, 0, W - 1))) for v in (x, x + w)]
    y1, y2 = [int(round(np.clip(v, 0, H - 1))) for v in (y, y + h)]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    ty = max(14, y1 - 5) if label == 'GT' else min(H - 4, y2 + 15)
    cv2.putText(frame, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1, cv2.LINE_AA)


def closed():
    try:
        return cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        return True


def dismiss(delay):
    return (cv2.waitKey(max(1, delay)) & 0xff) in (ord('q'), ord('Q'), 27) or closed()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, default=Path('pretrained_models/MobileViT_Track_ep0100_weights_only.pth'))
    p.add_argument('--sequence', type=Path, default=Path('data/val/GOT-10k_Val_000001'))
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--fps', type=float, default=15, help='Saved video playback FPS; actual live speed depends on inference')
    p.add_argument('--display', action='store_true', help='Show tracking and hold the final frame until Q/Esc/window close')
    p.add_argument('--output-dir', type=Path, default=Path('output/tracking_demo'))
    args = p.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0:
        p.error('--fps must be positive and finite')
    if not args.weights.is_file() or not args.sequence.is_dir():
        p.error('Check --weights and --sequence paths')
    root = args.sequence / 'img' if (args.sequence / 'img').is_dir() else args.sequence
    frames = sorted((f for f in root.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}), key=natural_key)
    gt = load_gt(args.sequence / 'groundtruth.txt')
    if len(frames) < 2 or len(frames) != len(gt) or not valid(gt[0]):
        raise ValueError(f'Invalid sequence: {len(frames)} images, {len(gt)} annotations; need matching counts and valid initialization')
    absent = np.zeros(len(frames), dtype=bool)
    absence_path = args.sequence / 'absence.label'
    if absence_path.is_file():
        flags = np.loadtxt(absence_path, ndmin=1).reshape(-1)
        if len(flags) != len(frames) or not np.isin(flags, [0, 1]).all() or flags[0]:
            raise ValueError('Invalid absence.label or absent target on initialization frame')
        absent = flags.astype(bool)

    import torch
    from mvt.models.mvt import MVT
    from mvt.evaluation.tracker import MVTTracker
    from mvt.training.pretrained import load_training_weights
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is unavailable; use --device cpu')
    model = MVT()
    print(load_training_weights(model, args.weights, backbone_only=False), flush=True)
    tracker = MVTTracker(model=model.to(device).eval(), device=device)
    first = read_frame(frames[0])
    H, W = first.shape[:2]
    tracker.initialize(cv2.cvtColor(first, cv2.COLOR_BGR2RGB), gt[0].tolist())

    out = args.output_dir / args.sequence.name
    out.mkdir(parents=True, exist_ok=True)
    video = out / 'MVT_tracking.mp4'
    # Add a separate information banner; do not obscure the tracked target.
    banner = 76
    ow = max(540, W); ow += ow % 2
    oh = H + banner; oh += oh % 2
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'mp4v'), args.fps, (ow, oh))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError(f'Cannot create video: {video}')
    predictions, rows = [], []
    total_iou = 0.0
    scored = 0
    stopped = False
    complete = False
    try:
        if args.display:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            scale = min(1.0, 1280 / ow, 800 / oh)
            cv2.resizeWindow(WINDOW, max(1, round(ow * scale)), max(1, round(oh * scale)))
        for i, path in enumerate(frames):
            start = time.perf_counter()
            frame = first.copy() if i == 0 else read_frame(path)
            if frame.shape[:2] != (H, W):
                raise ValueError(f'Frame dimensions changed: {path}')
            pred = gt[0].copy() if i == 0 else np.asarray(tracker.track(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))['target_bbox'], dtype=float)
            if pred.shape != (4,):
                raise ValueError('Tracker returned a non-XYWH bounding box')
            ref = gt[i] if not absent[i] else None
            score = overlap(pred, ref) if ref is not None else None
            if i > 0 and score is not None:
                total_iou += score
                scored += 1
            ao = total_iou / scored if scored else None
            if ref is not None:
                draw_box(frame, ref, GT, 'GT', 4)
            draw_box(frame, pred, PRED, 'MVT', 2)
            canvas = np.full((oh, ow, 3), (22, 25, 29), dtype=np.uint8)
            left = (ow - W) // 2
            canvas[banner:banner+H, left:left+W] = frame
            def text(s, x, y, color=(235,235,235)):
                cv2.putText(canvas, s, (x,y), cv2.FONT_HERSHEY_SIMPLEX, .49, color, 1, cv2.LINE_AA)
            text(f'{args.sequence.name} | Frame {i+1}/{len(frames)}', 10, 18)
            text('GREEN: Ground truth', 10, 38, GT)
            text('BLUE: MVT prediction', 250, 38, PRED)
            text(f'IoU: {score:.3f}' if score is not None else 'IoU: N/A', 10, 60)
            text(f'Running AO: {ao:.3f}' if ao is not None else 'Running AO: N/A (initialization)', 160, 60)
            writer.write(canvas)
            predictions.append(pred)
            rows.append((i+1, score, ao))
            if args.display:
                cv2.imshow(WINDOW, canvas)
                delay = round(1000 * max(0, 1/args.fps - (time.perf_counter()-start)))
                if dismiss(delay):
                    stopped = True
                    break
            if (i+1) % 50 == 0:
                print(f'Processed {i+1}/{len(frames)} frames', flush=True)
        complete = not stopped and len(rows) == len(frames)
    finally:
        writer.release()
        if args.display and not complete:
            cv2.destroyAllWindows()
    np.savetxt(out / 'predictions.txt', predictions, delimiter=',', fmt='%.6f')
    with (out / 'frame_metrics.csv').open('w', newline='', encoding='utf-8') as f:
        log = csv.writer(f)
        log.writerow(['frame_number', 'iou', 'running_ao_excluding_initialization'])
        log.writerows(rows)
    print(f'Video saved: {video.resolve()} | Frames: {len(rows)}', flush=True)
    if scored:
        print(f'Average overlap (excluding initialization): {total_iou/scored:.4f}', flush=True)
    if args.display and complete:
        try:
            print('Finished. Final frame stays open. Press Q or Esc to close.', flush=True)
            while not dismiss(50):
                pass
        finally:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
