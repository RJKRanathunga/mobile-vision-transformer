"""Validation demo: independent MVT predictions and reference boxes in one video.

Place this file in the patched repository root, next to demo_tripwire.py.
Run python demo_tripwire_gt.py --help for options.
"""
import argparse
import csv
import json
import math
import re
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from demo_tripwire import natural_key, read_image, load_engine
from mvt.applications.tripwire import VirtualTripwire, CrossingEvent

GT_COLOR = (0, 255, 0)       # BGR: green
PRED_COLOR = (255, 160, 0)   # BGR: blue
TRAIL_COLOR = (0, 255, 255)  # BGR: yellow
WIRE_COLOR = (0, 0, 255)     # BGR: red


def valid(box):
    b = np.asarray(box, dtype=float)
    return b.shape == (4,) and np.isfinite(b).all() and (b[2:] > 0).all()


def iou(a, b):
    if not valid(a) or not valid(b):
        return None
    a, b = np.asarray(a), np.asarray(b)
    inter_wh = np.maximum(0, np.minimum(a[:2]+a[2:], b[:2]+b[2:]) - np.maximum(a[:2], b[:2]))
    inter = float(np.prod(inter_wh))
    union = float(np.prod(a[2:]) + np.prod(b[2:]) - inter)
    return inter / union if union > 0 else None


def load_gt(path):
    rows = []
    for n, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        try:
            row = [float(v) for v in re.split(r'[,\s]+', line.strip())]
            if len(row) != 4:
                raise ValueError('expected four XYWH coordinates')
        except ValueError as exc:
            raise ValueError(f'{path}, row {n}: {exc}') from exc
        rows.append(row)
    return np.asarray(rows, dtype=float)


def draw_box(frame, box, color, label, label_below=False):
    if not valid(box):
        return
    x, y, w, h = map(float, box)
    height, width = frame.shape[:2]
    # Clip only drawing coordinates; metrics use original boxes.
    x1, y1, x2, y2 = [round(v) for v in (max(0, min(width-1, x)), max(0, min(height-1, y)), max(0, min(width-1, x+w)), max(0, min(height-1, y+h)))]
    cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2 if not label_below else 3)
    ty = min(height-4, y2+15) if label_below else max(12, y1-5)
    cv2.putText(frame, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, .45, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(frame, label, (x1, ty), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)


def write_events(path, wire):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(CrossingEvent.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(event) for event in wire.events)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--weights', type=Path, default=Path('pretrained_models/MobileViT_Track_ep0100_weights_only.pth'))
    p.add_argument('--sequence', type=Path, default=Path('data/val/GOT-10k_Val_000001'))
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--line', nargs=4, type=float, default=[.5,0,.5,1], metavar=('X1','Y1','X2','Y2'))
    p.add_argument('--margin', type=float, default=10)
    p.add_argument('--fps', type=float, default=15, help='Video playback FPS, not measured inference speed')
    p.add_argument('--trail', type=int, default=100)
    p.add_argument('--output-dir', type=Path, default=Path('output/gt_demo'))
    p.add_argument('--display', action='store_true', help='Open desktop preview; press Q or Esc to stop')
    p.add_argument('--no-tripwire', action='store_true', help='Only display GT, predictions and IoU')
    args = p.parse_args()
    if not args.weights.is_file():
        p.error(f'Checkpoint not found: {args.weights}')
    if not args.sequence.is_dir():
        p.error(f'Sequence not found: {args.sequence}')
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in args.line) or args.line[:2] == args.line[2:]:
        p.error('--line requires two distinct normalized XY endpoints')
    if not math.isfinite(args.margin) or args.margin < 0 or not math.isfinite(args.fps) or args.fps <= 0 or args.trail < 1:
        p.error('Margin must be nonnegative, FPS and trail must be positive')
    root = args.sequence / 'img' if (args.sequence / 'img').is_dir() else args.sequence
    frames = sorted((f for f in root.iterdir() if f.is_file() and f.suffix.lower() in {'.jpg','.jpeg','.png','.bmp'}), key=natural_key)
    gt_path = args.sequence / 'groundtruth.txt'
    if not gt_path.is_file():
        p.error(f'Ground-truth file missing: {gt_path}')
    gt = load_gt(gt_path)
    if len(frames) < 2 or len(frames) != len(gt):
        p.error(f'Need at least two frames and one annotation per frame; found {len(frames)} images and {len(gt)} annotations')
    # GOT-10k optionally marks target absence. Exclude absent reference frames
    # from IoU and reference events; do not give that information to MVT.
    absent_path = args.sequence / 'absence.label'
    absent = np.zeros(len(frames), dtype=bool)
    if absent_path.is_file():
        flags = np.loadtxt(absent_path, dtype=float, ndmin=1).reshape(-1)
        if len(flags) != len(frames) or not np.isin(flags, [0,1]).all():
            p.error('absence.label must contain one 0/1 value per frame')
        absent = flags.astype(bool)
    first = read_image(frames[0])
    h,w = first.shape[:2]
    if absent[0] or not valid(gt[0]):
        p.error('First frame must contain a valid visible target for initialization')
    a = (args.line[0]*(w-1), args.line[1]*(h-1))
    b = (args.line[2]*(w-1), args.line[3]*(h-1))
    pred_wire = VirtualTripwire(a,b,args.margin)
    gt_wire = VirtualTripwire(a,b,args.margin)
    v = (np.asarray(b)-a)/pred_wire.length
    reach = 2*math.hypot(w,h)
    draw_a = tuple(np.rint(np.asarray(a)-reach*v).astype(int))
    draw_b = tuple(np.rint(np.asarray(b)+reach*v).astype(int))
    engine, sync, device = load_engine(args)
    # This is the only time ground truth enters the tracker.
    engine.initialize(cv2.cvtColor(first,cv2.COLOR_BGR2RGB),gt[0].tolist())
    sync()
    out = args.output_dir / args.sequence.name
    out.mkdir(parents=True,exist_ok=True)
    # Add a HUD above the original image rather than obscuring the target.
    ow = max(480, w+w%2)
    hud = 100
    oh = h+hud+h%2
    video_path = out/'MVT_GT_comparison.mp4'
    writer = cv2.VideoWriter(str(video_path),cv2.VideoWriter_fourcc(*'mp4v'),args.fps,(ow,oh))
    if not writer.isOpened():
        writer.release()
        raise RuntimeError('Cannot create MP4 video; check OpenCV video codec support')
    trail = deque(maxlen=args.trail)
    gt_trail = deque(maxlen=args.trail)
    rows = []
    predictions = []
    scores = []
    inference_seconds = 0.0
    alert_until = 0
    gt_alert_until = 0
    stopped = False
    try:
        for n,path in enumerate(frames,1):
            frame = first.copy() if n == 1 else read_image(path)
            if frame.shape[:2] != (h,w):
                raise ValueError(f'Frame {n}: image size changed')
            if n == 1:
                pred = gt[0].copy()
            else:
                rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                sync(); start = time.perf_counter()
                pred = np.asarray(engine.track(rgb)['target_bbox'],dtype=float)
                sync(); inference_seconds += time.perf_counter()-start
            pred = pred if valid(pred) else np.full(4,np.nan)
            ref = gt[n-1] if not absent[n-1] and valid(gt[n-1]) else np.full(4,np.nan)
            score = iou(pred,ref)
            # Count invalid predictions as zero on valid reference frames.
            if valid(ref) and score is None:
                score = 0.0
            if n > 1 and score is not None:
                scores.append(score)
            mean = float(np.mean(scores)) if scores else None
            if not args.no_tripwire:
                pe = pred_wire.update(pred,n,args.fps)
                ge = gt_wire.update(ref,n,args.fps)
                if pe:
                    alert_until = n+max(1,round(2*args.fps))-1
                    print(f'PRED ALERT frame {n}: {pe.direction}',flush=True)
                if ge:
                    gt_alert_until = n+max(1,round(2*args.fps))-1
                    print(f'GT CROSSING frame {n}: {ge.direction}',flush=True)
                cv2.line(frame,draw_a,draw_b,WIRE_COLOR,2,cv2.LINE_AA)
                for box, buf, color in [(ref,gt_trail,GT_COLOR),(pred,trail,TRAIL_COLOR)]:
                    if valid(box):
                        buf.append(tuple(map(round,VirtualTripwire.get_center(box))))
                        if len(buf)>1:
                            cv2.polylines(frame,[np.asarray(buf,dtype=np.int32).reshape(-1,1,2)],False,color,1,cv2.LINE_AA)
                    else:
                        buf.clear()
            # Thicker outer green stroke remains visible when boxes coincide.
            draw_box(frame,ref,GT_COLOR,'GT',True)
            draw_box(frame,pred,PRED_COLOR,'MVT')
            canvas = np.full((oh,ow,3), (22,25,29), np.uint8)
            canvas[hud:hud+h,:w] = frame
            put = lambda s,y,c=(235,235,235): cv2.putText(canvas,s,(10,y),cv2.FONT_HERSHEY_SIMPLEX,.49,c,1,cv2.LINE_AA)
            put(f'{args.sequence.name} | Frame {n}/{len(frames)}',18)
            cv2.putText(canvas,'GREEN: Ground truth',(10,38),cv2.FONT_HERSHEY_SIMPLEX,.46,GT_COLOR,1,cv2.LINE_AA)
            cv2.putText(canvas,'BLUE: MVT prediction',(245,38),cv2.FONT_HERSHEY_SIMPLEX,.46,PRED_COLOR,1,cv2.LINE_AA)
            s = f'{score:.3f}' if score is not None else 'N/A'
            m = f'{mean:.3f}' if mean is not None else 'N/A'
            put(f'IoU: {s} | Running AO (excl. init): {m}',58)
            if args.no_tripwire:
                put('First-frame GT initialization only',81)
            else:
                put(f'Crossings  GT: {len(gt_wire.events)}  |  MVT: {len(pred_wire.events)}',78)
                alert = ('MVT ALERT' if n<=alert_until else 'MVT monitoring') + (' | GT crossing' if n<=gt_alert_until else '')
                put(alert,95,WIRE_COLOR if n<=alert_until else (235,235,235))
            writer.write(canvas)
            predictions.append(pred)
            rows.append({'frame_number':n,'time_seconds':(n-1)/args.fps,'iou':score,'running_ao':mean,'gt_valid':int(valid(ref)), 'pred_valid':int(valid(pred)), 'gt_crossings':len(gt_wire.events) if not args.no_tripwire else None,'pred_crossings':len(pred_wire.events) if not args.no_tripwire else None})
            if args.display:
                cv2.imshow('MVT vs Ground Truth',canvas)
                if cv2.waitKey(1)&0xff in (ord('q'),27):
                    stopped=True; break
            if n%50==0:
                print(f'Processed {n}/{len(frames)} frames | AO: {m}',flush=True)
    finally:
        writer.release()
        if args.display:
            cv2.destroyAllWindows()
    with (out/'frame_metrics.csv').open('w',newline='',encoding='utf-8') as f:
        log=csv.DictWriter(f,fieldnames=list(rows[0]))
        log.writeheader(); log.writerows(rows)
    np.savetxt(out/'predictions.txt',predictions,delimiter=',',fmt='%.6f')
    if not args.no_tripwire:
        write_events(out/'predicted_events.csv',pred_wire)
        write_events(out/'groundtruth_events.csv',gt_wire)
    else:
        # Remove stale event files if this output directory is reused.
        for name in ('predicted_events.csv','groundtruth_events.csv'):
            (out/name).unlink(missing_ok=True)
    summary = dict(sequence=args.sequence.name,weights=str(args.weights),device=str(device),frames=len(rows),total_frames=len(frames),stopped_by_user=stopped,scored_frames=len(scores),average_overlap=float(np.mean(scores)) if scores else None,tracker_fps=(len(rows)-1)/inference_seconds if inference_seconds>0 else None,playback_fps=args.fps,tripwire_enabled=not args.no_tripwire,margin_pixels=args.margin,line_normalized=args.line,predicted_crossings=len(pred_wire.events) if not args.no_tripwire else None,groundtruth_crossings=len(gt_wire.events) if not args.no_tripwire else None,absence_labels_used=absent_path.is_file())
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(f'Video saved: {video_path.resolve()}')


if __name__ == '__main__':
    main()
