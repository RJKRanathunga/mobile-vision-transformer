#!/usr/bin/env python
"""Patch the MVT tracker and optionally evaluate GOT-10k validation.
Target: RJKRanathunga/mobile-vision-transformer, commit 9b6910e.
Python 3.7+; patch-only requires only the standard library.
Run from the repository root. See --help for options.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from datetime import datetime

TRACKER = Path('lib/test/tracker/mobilevit_track.py')
MODULE = Path('lib/test/tracker/motion_predictor.py')
BACKUP = Path('.mvt_motion_patch/original_tracker.py.bak')
MANIFEST = Path('.mvt_motion_patch/manifest.json')
CONFIG = 'mobilevit_256_128x1_got10k_ep100_cosine_annealing'
VAL = r'D:\Uni_Projects\movit_object_tracking\data\val'
MODES = ('baseline', 'velocity', 'acceleration')
VERSION = '1.0'
MOTION_SOURCE = '''"""Constant velocity/acceleration motion prior for MVT inference."""
from collections import deque
import math
import numpy as np
import torch


class MotionPredictor:
    def __init__(self, mode="baseline"):
        if mode not in ("baseline", "velocity", "acceleration"):
            raise ValueError("Invalid motion mode: " + str(mode))
        self.mode = mode
        self.history = deque(maxlen=3)

    def reset(self, bbox):
        self.history.clear()
        if self.mode != "baseline":
            self.update(bbox)

    def update(self, bbox):
        if self.mode == "baseline":
            return
        x, y, w, h = bbox
        self.history.append(np.array([x + w / 2, y + h / 2], dtype=np.float64))

    def predict(self):
        if self.mode == "baseline" or len(self.history) < 2:
            return None
        p1, p2 = self.history[-1], self.history[-2]
        v = p1 - p2
        if self.mode == "acceleration" and len(self.history) == 3:
            a = v - (p2 - self.history[-3])
            return p1 + v + a
        return p1 + v


def motion_window(center, previous_bbox, resize_factor, search_size,
                  score_map, offset_map, sigma):
    """Gaussian at predicted full-frame center, evaluated at decoded centers.

    Match MVT map_box_back and CENTER cal_bbox: cell index + predicted offset.
    Sigma is in feature cells. If the forecast is outside the search crop,
    return None so the original Hann response is used unchanged.
    """
    if center is None or not np.isfinite(center).all():
        return None
    h, w = score_map.shape[-2:]
    x, y, bw, bh = previous_bbox
    side = float(search_size) / float(resize_factor)
    if not math.isfinite(side) or side <= 0:
        return None
    mx = (float(center[0]) - (x + bw / 2 - side / 2)) * w / side
    my = (float(center[1]) - (y + bh / 2 - side / 2)) * h / side
    if not (0 <= mx < w and 0 <= my < h):
        return None
    # Original head uses (index + offset) / feature size, not cell centers.
    gy, gx = torch.meshgrid(
        torch.arange(h, device=score_map.device, dtype=torch.float32),
        torch.arange(w, device=score_map.device, dtype=torch.float32),
        indexing="ij")
    cx = gx + offset_map[:, 0].float()
    cy = gy + offset_map[:, 1].float()
    prior = torch.exp(-((cx - mx)**2 + (cy - my)**2) / (2 * sigma**2))
    return prior.unsqueeze(1).to(dtype=score_map.dtype)
'''


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    with path.open('rb') as f:
        h = hashlib.sha256()
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_bytes(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        f.write(data)
    os.replace(str(tmp), str(path))


def json_write(path, obj):
    atomic_bytes(path, (json.dumps(obj, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError('Expected exactly one patch anchor:\n' + old + '\nNo source files were changed.')
    return source.replace(old, new, 1)


def build_patch(raw):
    source = raw.decode('utf-8')
    newline = '\r\n' if '\r\n' in source else '\n'
    source = source.replace('\r\n', '\n')
    source = replace_once(source, 'from lib.utils.ce_utils import generate_mask_cond\n',
        'from lib.utils.ce_utils import generate_mask_cond\n'
        'from lib.test.tracker.motion_predictor import MotionPredictor, motion_window\n')
    anchor = '        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).to(self.device)\n'
    source = replace_once(source, anchor, anchor + '''        # MVT_MOTION_PATCH_V1: baseline is the default.
        self.motion_predictor = MotionPredictor(os.getenv("MVT_MOTION_MODE", "baseline"))
        self.motion_weight = float(os.getenv("MVT_MOTION_WEIGHT", "0.5"))
        self.motion_sigma = float(os.getenv("MVT_MOTION_SIGMA", "2.0"))
        if not 0 <= self.motion_weight <= 1 or not math.isfinite(self.motion_sigma) or self.motion_sigma <= 0:
            raise ValueError("Motion weight must be in [0,1] and sigma must be positive")
        self.last_motion_center = None
        self.last_motion_applied = False
''')
    source = replace_once(source, '        self.save_state_dict = True\n',
        '        self.save_state_dict = os.getenv("MVT_SAVE_STATE_DICT", "1") != "0"\n')
    source = replace_once(source, "        self.state = info['init_bbox']\n",
        "        self.state = info['init_bbox']\n"
        '        self.motion_predictor.reset(self.state)\n'
        '        self.last_motion_center = None\n'
        '        self.last_motion_applied = False\n')
    source = replace_once(source, '        response = self.output_window * pred_score_map\n', '''        # Motion history contains only previous predictions, never current GT.
        self.last_motion_center = None
        self.last_motion_applied = False
        if self.motion_predictor.mode == "baseline" or self.motion_weight == 0:
            response = self.output_window * pred_score_map
        else:
            self.last_motion_center = self.motion_predictor.predict()
            prior = motion_window(self.last_motion_center, self.state, resize_factor,
                                  self.params.search_size, pred_score_map,
                                  out_dict['offset_map'], self.motion_sigma)
            if prior is None:
                response = self.output_window * pred_score_map
            else:
                response = pred_score_map * ((1 - self.motion_weight) * self.output_window
                                             + self.motion_weight * prior)
                self.last_motion_applied = True
''')
    anchor = '        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)\n'
    source = replace_once(source, anchor, anchor + '        self.motion_predictor.update(self.state)\n')
    # Keep existing raw/Hann debug views and show the actual combined response.
    anchor = "                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')\n"
    source = replace_once(source, anchor, anchor + "                self.visdom.register(response.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_final')\n")
    compile(source, str(TRACKER), 'exec')
    compile(MOTION_SOURCE, str(MODULE), 'exec')
    return source.replace('\n', newline).encode('utf-8')


def verify_patch(root):
    m = json.loads((root / MANIFEST).read_text(encoding='utf-8'))
    for path, key in ((TRACKER, 'patched'), (MODULE, 'module'), (BACKUP, 'original')):
        if not (root / path).is_file() or file_digest(root / path) != m[key]:
            raise RuntimeError('Patch state differs from saved manifest: ' + str(path) + '. Refusing to overwrite edits.')
    return m


def apply_patch(root):
    if (root / MANIFEST).exists():
        verify_patch(root)
        print('Patch already installed; verified. No changes made.')
        return
    if (root / MODULE).exists() or (root / BACKUP).exists():
        raise RuntimeError('Motion module or backup already exists without a manifest. Refusing to overwrite.')
    raw = (root / TRACKER).read_bytes()
    patched = build_patch(raw)
    module = MOTION_SOURCE.encode('utf-8')
    manifest = dict(version=VERSION, original=digest(raw), patched=digest(patched), module=digest(module))
    # Preflight complete. On a normal exception, restore original tracker.
    try:
        atomic_bytes(root / BACKUP, raw)
        atomic_bytes(root / MODULE, module)
        atomic_bytes(root / TRACKER, patched)
        json_write(root / MANIFEST, manifest)
    except BaseException:
        atomic_bytes(root / TRACKER, raw)
        for p in (MODULE, BACKUP, MANIFEST):
            if (root / p).exists():
                (root / p).unlink()
        raise
    print('Patch applied:', root / TRACKER)
    print('New module:', root / MODULE)
    print('Original backup:', root / BACKUP)


def restore_patch(root):
    verify_patch(root)
    atomic_bytes(root / TRACKER, (root / BACKUP).read_bytes())
    for p in (MODULE, BACKUP, MANIFEST):
        (root / p).unlink()
    (root / MANIFEST).parent.rmdir()
    print('Original tracker restored byte-for-byte. Evaluation outputs preserved.')


def load_numeric(path, np):
    try:
        return np.loadtxt(str(path), delimiter=',')
    except ValueError:
        return np.loadtxt(str(path))


def collect_sequences(val, limit, allow_missing_cover, np):
    if not val.is_dir():
        raise RuntimeError('Validation directory does not exist: ' + str(val))
    listing = val / 'list.txt'
    names = ([s.strip() for s in listing.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
             if listing.exists() else sorted(p.name for p in val.glob('GOT-10k_Val_*') if p.is_dir()))
    if not names or len(set(names)) != len(names):
        raise RuntimeError('No sequences or duplicate sequence names in validation dataset.')
    if limit:
        names = names[:limit]
    sequences = []
    for name in names:
        if Path(name).name != name:
            raise RuntimeError('Invalid sequence name: ' + name)
        folder = val / name
        frames = sorted((p for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg') and p.stem.isdigit()), key=lambda p: int(p.stem))
        gt = np.atleast_2d(load_numeric(folder / 'groundtruth.txt', np)).astype(np.float64)
        if gt.shape != (len(frames), 4) or len(frames) < 2:
            raise RuntimeError('Image/annotation count mismatch or too few frames in ' + name)
        if not np.isfinite(gt[0]).all() or not (gt[0, 2:] > 0).all():
            raise RuntimeError('Invalid initial ground truth in ' + name)
        valid = np.isfinite(gt).all(axis=1) & (gt[:, 2:] > 0).all(axis=1)
        cover_path = folder / 'cover.label'
        if cover_path.is_file():
            cover = load_numeric(cover_path, np).reshape(-1)
            if len(cover) != len(frames) or not np.isfinite(cover).all():
                raise RuntimeError('Invalid cover.label in ' + name)
            valid &= cover > 0
        elif not allow_missing_cover:
            raise RuntimeError('Missing ' + str(cover_path) + '. Restore labels or explicitly pass --allow-missing-cover for a nonstandard comparison.')
        else:
            print('WARNING: missing cover.label in ' + name + '; using valid boxes only.')
        valid[0] = False  # Initial ground-truth box is supplied, not predicted.
        sequences.append((name, frames, gt, valid, cover_path.is_file()))
    if not any(s[3].any() for s in sequences):
        raise RuntimeError('No evaluable visible frames.')
    return sequences


def iou_xywh(pred, gt, bounds, np):
    # GOT-10k-style intersection of rectangles clipped to image bounds.
    bound = np.asarray(bounds, dtype=np.float64)
    p0 = np.clip(pred[:, :2], 0, bound)
    p1 = np.clip(pred[:, :2] + pred[:, 2:], 0, bound)
    g0 = np.clip(gt[:, :2], 0, bound)
    g1 = np.clip(gt[:, :2] + gt[:, 2:], 0, bound)
    inter = np.maximum(0, np.minimum(p1, g1) - np.maximum(p0, g0)).prod(axis=1)
    union = np.maximum(0, p1 - p0).prod(axis=1) + np.maximum(0, g1 - g0).prod(axis=1) - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def stats(ious, times, errors, applied, np):
    return dict(frames=int(len(ious)), AO=float(ious.mean()) if len(ious) else None,
                SR_50=float((ious > 0.5).mean()) if len(ious) else None,
                SR_75=float((ious > 0.75).mean()) if len(ious) else None,
                FPS=float(len(times)/times.sum()) if len(times) and times.sum() > 0 else None,
                motion_MAE_px=float(errors.mean()) if len(errors) else None,
                motion_error_frames=int(len(errors)), motion_applied_frames=int(applied.sum()))


def write_csv(path, rows):
    if not rows:
        return
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(str(tmp), str(path))


def evaluate(root, args):
    verify_patch(root)
    import numpy as np
    import torch
    import cv2
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required: this repository Preprocessor uses .cuda(). Activate the working MVT conda environment.')
    sequences = collect_sequences(Path(args.val_dir).resolve(), args.limit, args.allow_missing_cover, np)
    checkpoint = Path(args.checkpoint).resolve() if args.checkpoint else root / 'output/checkpoints/train/mobilevit_track' / args.config / 'MobileViT_Track_ep0100.pth.tar'
    config = root / 'experiments/mobilevit_track' / (args.config + '.yaml')
    if not checkpoint.is_file():
        raise RuntimeError('Checkpoint missing: ' + str(checkpoint) + '. Pass --checkpoint with the trusted original tracking checkpoint (dict containing net).')
    if not config.is_file():
        raise RuntimeError('Configuration missing: ' + str(config))
    # Local validation runner: no dependency on the repo's hard-coded environment paths.
    # Source/checkpoint hashes prevent accidental resume across changed experiments.
    source_hashes = {str(p.relative_to(root)).replace('\\', '/'): file_digest(p)
                     for p in sorted((root / 'lib').rglob('*.py'))}
    dataset_signature = []
    for name, frames, gt, valid, has_cover in sequences:
        dataset_signature.append(dict(name=name, count=len(frames), annotations=digest(gt.tobytes()),
                                      valid=digest(valid.tobytes()), cover_present=has_cover,
                                      images=digest('\n'.join('{}:{}:{}'.format(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in frames).encode())))
    metadata = dict(version=VERSION, val_dir=str(Path(args.val_dir).resolve()), checkpoint=str(checkpoint),
                    checkpoint_sha256=file_digest(checkpoint), config=str(config), config_sha256=file_digest(config),
                    source_hashes=source_hashes, modes=args.modes, motion_weight=args.weight, sigma=args.sigma,
                    dataset=dataset_signature, device=torch.cuda.get_device_name(0), torch=str(torch.__version__),
                    metrics='frame-weighted; visible valid frames except initialization; IoU clipped to image; strict > thresholds; synchronized tracking-only throughput')
    out = Path(args.output_dir).resolve() if args.output_dir else root / 'output/motion_validation' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    meta_file = out / 'experiment.json'
    if args.resume:
        if not meta_file.is_file() or json.loads(meta_file.read_text(encoding='utf-8')) != metadata:
            raise RuntimeError('Cannot resume: experiment metadata differs or is missing. Use a new output directory.')
    else:
        if out.exists() and any(out.iterdir()):
            raise RuntimeError('Output directory is not empty. Use --resume or choose a new directory.')
        out.mkdir(parents=True, exist_ok=True)
        json_write(meta_file, metadata)
    print('Results:', out, flush=True)
    print('Checkpoint:', checkpoint, flush=True)
    print('Sequences:', len(sequences), 'Modes:', ', '.join(args.modes), flush=True)
    sys.path.insert(0, str(root))
    os.environ['MVT_MOTION_MODE'] = 'baseline'
    os.environ['MVT_MOTION_WEIGHT'] = str(args.weight)
    os.environ['MVT_MOTION_SIGMA'] = str(args.sigma)
    os.environ['MVT_SAVE_STATE_DICT'] = '0'
    from types import SimpleNamespace
    from lib.config.mobilevit_track.config import cfg, update_config_from_file
    from lib.test.tracker.mobilevit_track import MobileViTTrack
    update_config_from_file(str(config))
    if cfg.TEST.DEVICE != 'cuda' or cfg.MODEL.HEAD.TYPE != 'CENTER':
        raise RuntimeError('This patch evaluator expects the original CUDA CENTER-head MVT configuration.')
    params = SimpleNamespace(cfg=cfg, checkpoint=str(checkpoint), debug=0, save_all_boxes=False,
                             template_factor=cfg.TEST.TEMPLATE_FACTOR, template_size=cfg.TEST.TEMPLATE_SIZE,
                             search_factor=cfg.TEST.SEARCH_FACTOR, search_size=cfg.TEST.SEARCH_SIZE)
    tracker = MobileViTTrack(params, 'got10k_val')
    torch.backends.cudnn.benchmark = False
    def read_image(p):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError('Cannot read image: ' + str(p))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Warm GPU once, then initialize again for every sequence/mode; no GT leakage.
    _, warm_frames, warm_gt, _, _ = sequences[0]
    tracker.initialize(read_image(warm_frames[0]), {'init_bbox': warm_gt[0].tolist()})
    tracker.track(read_image(warm_frames[1]))
    torch.cuda.synchronize()
    by_mode = {m: [] for m in args.modes}
    rows = []
    for si, (name, frames, gt, valid, _) in enumerate(sequences):
        # Rotate order to reduce systematic warm-up/thermal bias.
        order = args.modes[si % len(args.modes):] + args.modes[:si % len(args.modes)]
        for mode in order:
            dest = out / mode / (name + '.npz')
            if dest.exists() and args.resume:
                with np.load(str(dest), allow_pickle=False) as z:
                    boxes, times, forecast, applied, bounds = (z[k] for k in ('boxes','times','forecast','applied','bounds'))
                if boxes.shape != gt.shape or times.shape != (len(frames)-1,) or forecast.shape != (len(frames),2) or applied.shape != (len(frames),) or bounds.shape != (len(frames),2):
                    raise RuntimeError('Invalid cached sequence: ' + str(dest))
                print('[{}/{}] {} {} (resumed)'.format(si+1,len(sequences),name,mode), flush=True)
            else:
                tracker.motion_predictor.mode = mode
                first = read_image(frames[0])
                tracker.initialize(first, {'init_bbox': gt[0].tolist()})
                boxes = np.empty_like(gt); boxes[0] = gt[0]
                times = np.empty(len(frames)-1, dtype=np.float64)
                forecast = np.full((len(frames),2), np.nan, dtype=np.float64)
                applied = np.zeros(len(frames), dtype=bool)
                bounds = np.empty((len(frames),2), dtype=np.float64); bounds[0] = (first.shape[1],first.shape[0])
                for i, frame in enumerate(frames[1:], 1):
                    image = read_image(frame)
                    bounds[i] = (image.shape[1], image.shape[0])
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    result = tracker.track(image)  # No annotations passed after initialization.
                    torch.cuda.synchronize()
                    times[i-1] = time.perf_counter() - t0
                    boxes[i] = result['target_bbox']
                    if tracker.last_motion_center is not None:
                        forecast[i] = tracker.last_motion_center
                    applied[i] = tracker.last_motion_applied
                    if not np.isfinite(boxes[i]).all() or not (boxes[i,2:] > 0).all():
                        raise RuntimeError('Invalid prediction in {} / {} / frame {}'.format(name, mode, i))
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + '.tmp')
                with tmp.open('wb') as f:
                    np.savez_compressed(f, boxes=boxes, times=times, forecast=forecast, applied=applied, bounds=bounds)
                os.replace(str(tmp), str(dest))
            if not np.isfinite(boxes).all() or not (boxes[:,2:] > 0).all() or not np.isfinite(times).all() or not (times > 0).all():
                raise RuntimeError('Invalid saved predictions/timings: ' + str(dest))
            # Export float coordinates and synchronized seconds for inspection.
            np.savetxt(str(dest.with_suffix('.txt')), boxes, delimiter=',', fmt='%.6f')
            np.savetxt(str(dest.with_name(name + '_time.txt')), times, fmt='%.9f')
            ious = iou_xywh(boxes[valid], gt[valid], bounds[valid], np)
            ok = valid & np.isfinite(forecast).all(axis=1)
            errors = np.linalg.norm(forecast[ok] - (gt[ok,:2] + gt[ok,2:]/2), axis=1)
            s = stats(ious, times, errors, applied, np)
            rows.append(dict(sequence=name, mode=mode, **s))
            by_mode[mode].append((ious, times, errors, applied))
            print('[{}/{}] {} {:12s} AO={} FPS={:.2f}'.format(si+1,len(sequences),name,mode,
                  'n/a' if s['AO'] is None else '{:.4f}'.format(s['AO']),s['FPS']), flush=True)
            write_csv(out / 'per_sequence.csv', rows)
    summary = []
    curves = {}
    thresholds = np.linspace(0,1,101)
    for mode in args.modes:
        data = [np.concatenate(a) for a in zip(*by_mode[mode])]
        summary.append(dict(mode=mode, **stats(*data, np)))
        curves[mode] = [(data[0] > t).mean() for t in thresholds]
    baseline = next((r for r in summary if r['mode']=='baseline'), None)
    for r in summary:
        r['delta_AO_pp'] = 100*(r['AO']-baseline['AO']) if baseline else None
        r['delta_SR_50_pp'] = 100*(r['SR_50']-baseline['SR_50']) if baseline else None
        r['delta_SR_75_pp'] = 100*(r['SR_75']-baseline['SR_75']) if baseline else None
        r['delta_FPS_pct'] = 100*(r['FPS']/baseline['FPS']-1) if baseline else None
    write_csv(out / 'summary.csv', summary)
    json_write(out / 'summary.json', summary)
    write_csv(out / 'success_curve.csv', [dict(threshold=float(t), **{m:float(curves[m][i]) for m in args.modes}) for i,t in enumerate(thresholds)])
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for mode in args.modes:
            plt.plot(thresholds, curves[mode], label=mode)
        plt.xlabel('IoU threshold'); plt.ylabel('Success rate'); plt.xlim(0,1); plt.ylim(0,1)
        plt.grid(alpha=.25); plt.legend(); plt.tight_layout()
        plt.savefig(str(out / 'success_curve.png'), dpi=160); plt.close()
    except Exception as e:
        print('Optional plot unavailable:', e)
    print('\nFINAL VALIDATION RESULTS (same checkpoint and sequences)')
    print('{:14s} {:>9s} {:>9s} {:>9s} {:>9s} {:>12s}'.format('Mode','AO','SR50','SR75','FPS','Delta AO pp'))
    for r in summary:
        delta = 'n/a' if r['delta_AO_pp'] is None else '{:+.3f}'.format(r['delta_AO_pp'])
        print('{:14s} {:9.4f} {:9.4f} {:9.4f} {:9.2f} {:>12s}'.format(r['mode'],r['AO'],r['SR_50'],r['SR_75'],r['FPS'],delta))
    print('Saved:', out / 'summary.csv')
    print('Local validation comparison, not a held-out GOT-10k test-server result.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--evaluate', action='store_true', help='Apply/verify patch and run validation')
    action.add_argument('--restore', action='store_true', help='Restore original tracker if patched files are unmodified')
    parser.add_argument('--val-dir', default=VAL)
    parser.add_argument('--config', default=CONFIG, help='Experiment YAML stem')
    parser.add_argument('--checkpoint', help='Explicit trusted checkpoint containing net (default: official ep0100)')
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--weight', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=2.0)
    parser.add_argument('--limit', type=int, default=0, help='First N validation sequences; 0 = all')
    parser.add_argument('--output-dir', help='New results directory; default is timestamped under output/motion_validation')
    parser.add_argument('--resume', action='store_true', help='Resume matching experiment; requires --output-dir')
    parser.add_argument('--allow-missing-cover', action='store_true', help='Use valid-box mask for missing visibility labels (nonstandard)')
    args = parser.parse_args()
    if not 0 <= args.weight <= 1 or not math.isfinite(args.sigma) or args.sigma <= 0 or args.limit < 0:
        parser.error('Weight must be in [0,1], sigma positive, limit nonnegative.')
    if len(set(args.modes)) != len(args.modes):
        parser.error('Do not repeat modes.')
    if args.resume and (not args.evaluate or not args.output_dir):
        parser.error('--resume requires --evaluate and --output-dir.')
    root = Path.cwd()
    if not (root / TRACKER).is_file() or not (root / 'tracking/test.py').is_file():
        parser.error('Run this script from the MVT repository root, not from Downloads.')
    if args.restore:
        restore_patch(root)
    else:
        apply_patch(root)
        if args.evaluate:
            evaluate(root, args)
        else:
            print('Run again with --evaluate --limit 3 for a smoke comparison, or --evaluate for all sequences.')


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print('\nERROR: ' + (str(exc) or type(exc).__name__), file=sys.stderr)
        if isinstance(exc, KeyboardInterrupt):
            print('Completed sequence files are retained. Resume with identical options and --output-dir PATH --resume.', file=sys.stderr)
        else:
            import traceback
            traceback.print_exc()
        sys.exit(1)
