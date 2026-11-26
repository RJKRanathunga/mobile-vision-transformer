"""Track a single target and record fixed-boundary crossing events."""
import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import time
from collections import deque
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from mvt.applications.tripwire import VirtualTripwire, CrossingEvent


def natural_key(path):
    return tuple((0, int(s)) if s.isdigit() else (1, s.lower())
                 for s in re.split(r'(\d+)', path.name))


def read_image(path):
    image = cv2.imread(str(path))
    if image is None:
        raise RuntimeError(f'Cannot read {path}')
    return image


def sequence_frames(folder):
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    root = folder / 'img' if (folder / 'img').is_dir() else folder
    paths = sorted((p for p in root.iterdir() if p.is_file() and p.suffix.lower()
                    in {'.jpg', '.jpeg', '.png', '.bmp'}), key=natural_key)
    if not paths:
        raise ValueError(f'No images in {root}')
    for p in paths:
        yield read_image(p)


def video_frames(cap):
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield frame


def synthetic_frames():
    for i in range(150):
        x = round(75 + 490 * (1 - abs(i - 74.5) / 74.5))
        frame = np.full((360, 640, 3), (38, 48, 46), np.uint8)
        for gy in range(0, 360, 40):
            cv2.line(frame, (0, gy), (639, gy), (52, 64, 60), 1)
        cv2.rectangle(frame, (x-25, 155), (x+25, 205), (180, 120, 50), -1)
        yield frame, [x-25, 155, 50, 50]


def load_engine(args):
    import torch
    from mvt.models.mvt import MVT
    from mvt.evaluation.tracker import MVTTracker
    from mvt.training.pretrained import load_training_weights
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu')
    model = MVT()
    print(load_training_weights(model, args.weights, backbone_only=False), flush=True)
    engine = MVTTracker(model=model, device=device)
    sync = torch.cuda.synchronize if device == 'cuda' else lambda: None
    return engine, sync, device


def valid_box(box, width, height):
    VirtualTripwire.get_center(box)
    x, y, w, h = box
    if x < 0 or y < 0 or x+w > width or y+h > height:
        raise ValueError('Initial box must lie inside the first frame')
    return list(map(float, box))


def choose_box(args, first):
    if args.init_box is not None:
        return args.init_box
    if args.select_roi:
        try:
            return cv2.selectROI('Select target; Enter confirms', first, False, False)
        finally:
            cv2.destroyAllWindows()
    if args.sequence:
        gt = args.sequence / 'groundtruth.txt'
        if gt.is_file():
            with gt.open(encoding='utf-8-sig') as f:
                return [float(v) for v in re.split(r'[,\s]+', f.readline().strip())]
    raise ValueError('Provide --init-box X Y W H or --select-roi (desktop only)')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--sequence', type=Path)
    source.add_argument('--video', type=Path)
    source.add_argument('--synthetic', action='store_true', help='Algorithm demo only; no MVT inference')
    p.add_argument('--weights', type=Path)
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--line', type=float, nargs=4, default=[.5,0,.5,1], metavar=('X1','Y1','X2','Y2'), help='Normalized infinite line, fixed before tracking')
    init = p.add_mutually_exclusive_group()
    init.add_argument('--init-box', type=float, nargs=4, metavar=('X','Y','W','H'))
    init.add_argument('--select-roi', action='store_true', help='Desktop GUI: select first-frame target')
    p.add_argument('--margin', type=float, default=10)
    p.add_argument('--fps', type=float, help='Sequence playback FPS (default 15); video uses source FPS unless overridden')
    p.add_argument('--trail', type=int, default=100)
    p.add_argument('--display', action='store_true', help='Desktop preview; Q stops')
    p.add_argument('--browser-video', action='store_true', help='Encode H.264 with ffmpeg if installed')
    p.add_argument('--output-dir', type=Path, default=Path('output/novelty_demo'))
    args = p.parse_args()
    if not args.synthetic and (args.weights is None or not args.weights.is_file()):
        p.error('A valid full-model --weights checkpoint is required for real MVT inference')
    if args.synthetic and (args.weights or args.init_box or args.select_roi):
        p.error('--synthetic supplies its own boxes; do not pass weights or initialization options')
    if not all(math.isfinite(v) and 0 <= v <= 1 for v in args.line):
        p.error('--line coordinates must be finite and between 0 and 1')
    if not math.isfinite(args.margin) or args.margin < 0 or args.trail < 1:
        p.error('Margin must be nonnegative and trail must be positive')
    if args.fps is not None and (not math.isfinite(args.fps) or args.fps <= 0):
        p.error('--fps must be positive')
    if args.browser_video and not shutil.which('ffmpeg'):
        p.error('--browser-video requires ffmpeg on PATH')
    cap = None
    writer = None
    try:
        if args.synthetic:
            frames = iter(synthetic_frames())
            name = 'synthetic_algorithm_only'
            source_fps = 15.0
        elif args.sequence:
            frames = ((f, None) for f in sequence_frames(args.sequence))
            name = args.sequence.name
            source_fps = 15.0
        else:
            cap = cv2.VideoCapture(str(args.video))
            if not cap.isOpened():
                raise RuntimeError(f'Cannot open video {args.video}')
            frames = ((f, None) for f in video_frames(cap))
            name = args.video.stem
            source_fps = cap.get(cv2.CAP_PROP_FPS)
            if not math.isfinite(source_fps) or source_fps <= 0:
                if args.fps is None:
                    raise ValueError('Unknown video FPS; provide --fps')
        fps = args.fps if args.fps is not None else source_fps
        try:
            first, synthetic_box = next(frames)
        except StopIteration:
            raise RuntimeError('Source contains no readable frames') from None
        height, width = first.shape[:2]
        a = (args.line[0]*(width-1), args.line[1]*(height-1))
        b = (args.line[2]*(width-1), args.line[3]*(height-1))
        wire = VirtualTripwire(a, b, args.margin)
        initial = valid_box(synthetic_box if args.synthetic else choose_box(args, first), width, height)
        engine = None
        sync = lambda: None
        device = 'none (synthetic)'
        if not args.synthetic:
            engine, sync, device = load_engine(args)
            engine.initialize(cv2.cvtColor(first, cv2.COLOR_BGR2RGB), initial)
            sync()
        # Even dimensions prevent silent cropping by common video encoders.
        out_w, out_h = width + width % 2, height + height % 2
        out = args.output_dir / name
        out.mkdir(parents=True, exist_ok=True)
        video_path = out / 'MVT_tripwire.mp4'
        if args.synthetic:
            video_path = out / 'synthetic_tripwire.mp4'
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w,out_h))
        if not writer.isOpened():
            raise RuntimeError('Cannot create output video')
        reach = 2*math.hypot(width,height)
        v = np.array([wire.dx,wire.dy])/wire.length
        draw_a = tuple(np.rint(np.array(a)-reach*v).astype(int))
        draw_b = tuple(np.rint(np.array(b)+reach*v).astype(int))
        trail = deque(maxlen=args.trail)
        alert_until = 0
        inference_seconds = application_seconds = 0.0
        tracked = count = invalid = 0
        stopped = False
        started = time.perf_counter()
        from itertools import chain
        with (out/'events.csv').open('w', newline='', encoding='utf-8') as ef, (out/'predictions.txt').open('w', encoding='utf-8') as pf:
            log = csv.DictWriter(ef, fieldnames=list(CrossingEvent.__dataclass_fields__))
            log.writeheader()
            for n, (frame, supplied) in enumerate(chain([(first,synthetic_box)], frames), 1):
                if frame.shape[:2] != (height,width):
                    raise ValueError(f'Frame {n} dimensions changed')
                if n == 1:
                    box = initial
                elif args.synthetic:
                    box = supplied
                else:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    sync(); t = time.perf_counter()
                    box = engine.track(rgb)['target_bbox']
                    sync(); inference_seconds += time.perf_counter()-t
                    tracked += 1
                t = time.perf_counter()
                event = wire.update(box,n,fps)
                application_seconds += time.perf_counter()-t
                if event:
                    log.writerow(asdict(event)); ef.flush()
                    alert_until = n + round(2*fps) - 1
                    print(f'ALERT | frame={n} | t={event.timestamp:.2f}s | {event.direction}', flush=True)
                cv2.line(frame, draw_a, draw_b, (0,0,255), 2)
                try:
                    cx,cy = wire.get_center(box)
                    x,y,w,h = map(lambda k: round(float(k)),box)
                    trail.append((round(cx),round(cy)))
                    if len(trail)>1:
                        cv2.polylines(frame,[np.array(trail,dtype=np.int32).reshape(-1,1,2)],False,(0,255,255),2)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    cv2.circle(frame,(round(cx),round(cy)),4,(0,255,255),-1)
                    pf.write(','.join(f'{float(k):.6f}' for k in box)+'\n')
                except (ValueError,TypeError,OverflowError):
                    invalid += 1; trail.clear(); pf.write('nan,nan,nan,nan\n')
                alert = n <= alert_until
                cv2.putText(frame,'ALERT: TARGET CROSSED' if alert else 'Monitoring target',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,255) if alert else (255,255,255),2)
                cv2.putText(frame,f'Frame {n} | Crossings: {len(wire.events)}',(10,50),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
                if args.synthetic:
                    cv2.putText(frame,'SYNTHETIC - NO MVT INFERENCE',(10,height-12),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
                frame = cv2.copyMakeBorder(frame,0,out_h-height,0,out_w-width,cv2.BORDER_CONSTANT,value=(0,0,0))
                writer.write(frame); count = n
                if args.display:
                    cv2.imshow('MVT Tripwire',frame)
                    if cv2.waitKey(1)&0xff == ord('q'):
                        stopped = True; break
                if n%50 == 0:
                    print(f'Processed {n} frames',flush=True)
        writer.release(); writer = None
        elapsed = time.perf_counter()-started
        summary = dict(source=name, mode='synthetic_algorithm_only' if args.synthetic else 'mvt_inference', device=device,
                       weights=str(args.weights) if args.weights else None, frames=count, crossings=len(wire.events), invalid_boxes=invalid,
                       stopped_by_user=stopped, playback_fps=fps, margin_pixels=args.margin, line_normalized=args.line,
                       initial_bbox=initial, tracker_fps=tracked/inference_seconds if inference_seconds else None,
                       tripwire_ms_per_frame=1000*application_seconds/count if count else None,
                       processing_fps=count/elapsed if elapsed else None,
                       timing_note='Processing FPS includes frame acquisition, overlays and encoding; excludes initialization and H264 conversion. Source timestamps are nominal frame/FPS.')
        if args.browser_video:
            browser = out/'tripwire_browser.mp4'
            subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(video_path),'-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(browser)],check=True)
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print(json.dumps(summary,indent=2)); print(f'Outputs: {out.resolve()}')
    finally:
        if writer is not None:
            writer.release()
        if cap is not None:
            cap.release()
        if args.display:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
