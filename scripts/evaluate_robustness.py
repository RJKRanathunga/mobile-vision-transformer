"""Measure local LaSOT failure rates from a labelled sequence manifest.

Manifest CSV: sequence,path,attributes. Path is relative to --data-root;
attributes are space-separated official abbreviations. Only valid GT boxes
are scored; initialization frame excluded. Report both frame and sequence means.
"""
import argparse
import csv
import json
import re
from pathlib import Path
import numpy as np
from report_common import ATTRIBUTES, csv_write, plt, save


def overlaps(pred, gt):
    pred=np.asarray(pred,dtype=float); gt=np.asarray(gt,dtype=float)
    if pred.shape!=gt.shape or pred.ndim!=2 or pred.shape[1]!=4: raise ValueError('Prediction and GT must match Nx4 xywh')
    valid=np.isfinite(gt).all(axis=1)&(gt[:,2:]>0).all(axis=1)
    if not np.isfinite(pred).all(): raise ValueError('Non-finite predicted box')
    p=pred[valid]; g=gt[valid]; ps=np.maximum(p[:,2:],0); gs=g[:,2:]
    wh=np.maximum(np.minimum(p[:,:2]+ps,g[:,:2]+gs)-np.maximum(p[:,:2],g[:,:2]),0)
    inter=wh.prod(axis=1); union=ps.prod(axis=1)+gs.prod(axis=1)-inter
    return inter/union


def read_boxes(path):
    return np.loadtxt(path,delimiter=',',ndmin=2)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--manifest',required=True,type=Path); p.add_argument('--data-root',required=True,type=Path)
    g=p.add_mutually_exclusive_group(required=True); g.add_argument('--weights',type=Path); g.add_argument('--predictions-dir',type=Path,help='One <sequence>.txt (comma-separated xywh) per manifest row')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto'); p.add_argument('--name',default='MVT_local'); p.add_argument('--output-dir',type=Path,default=Path('reports/robustness')); a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',a.name): p.error('--name must be a simple file-safe name')
    with a.manifest.open(encoding='utf-8-sig',newline='') as f: manifest=list(csv.DictReader(f))
    if not manifest or not {'sequence','path','attributes'}<=set(manifest[0]): raise ValueError('Manifest needs sequence,path,attributes columns')
    seen=set(); rows=[]; o=a.output_dir; o.mkdir(parents=True,exist_ok=True)
    if a.weights:
        import torch
        from PIL import Image
        from mvt.models.mvt import MVT
        from mvt.evaluation.tracker import MVTTracker
        from mvt.training.pretrained import load_training_weights
        device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device
        if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
        model=MVT(); print(load_training_weights(model,a.weights,backbone_only=False)); model=model.to(device).eval()
        pred_dir=o/'predictions'; pred_dir.mkdir(exist_ok=True)
    for item in manifest:
        name=item['sequence'].strip(); attrs=set(re.split(r'[\s;|,]+',item['attributes'].strip().upper()))-{''}
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',name) or name in ('.','..') or name in seen: raise ValueError(f'Invalid/duplicate sequence {name}')
        if not attrs or attrs-set(ATTRIBUTES): raise ValueError(f'{name}: invalid attributes {attrs}')
        seen.add(name); seq=a.data_root/item['path']; gt=read_boxes(seq/'groundtruth.txt')
        if len(gt)<2 or gt.shape[1]!=4 or not np.isfinite(gt[0]).all() or not (gt[0,2:]>0).all(): raise ValueError(f'{name}: invalid GT/init')
        if a.weights:
            frames=sorted((seq/'img').glob('*.jpg'),key=lambda x:int(x.stem))
            if len(frames)!=len(gt): raise ValueError(f'{name}: image/GT count mismatch')
            engine=MVTTracker(model=model,device=device)
            def image(path):
                with Image.open(path) as im: return np.asarray(im.convert('RGB'))
            with torch.inference_mode():
                engine.initialize(image(frames[0]),gt[0].tolist()); preds=[gt[0].tolist()]
                for frame in frames[1:]: preds.append(engine.track(image(frame))['target_bbox'])
            pred=np.asarray(preds,dtype=float); np.savetxt(pred_dir/f'{name}.txt',pred,delimiter=',',fmt='%.6f')
        else: pred=read_boxes(a.predictions_dir/f'{name}.txt')
        if pred.shape!=gt.shape: raise ValueError(f'{name}: prediction/GT count mismatch')
        iou=overlaps(pred[1:],gt[1:]); n=len(iou)
        if not n: raise ValueError(f'{name}: no valid evaluation frames')
        fails=int(np.count_nonzero(iou==0)); rows.append({'sequence':name,'attributes':' '.join(sorted(attrs)),'frames':n,'failures':fails,'fr':fails/n,'ao':float(iou.mean())}); print(f'{name}: FR={fails/n:.4f} ({n} scored frames)',flush=True)
    def aggregate(key,group):
        if not group: return {'attribute':key,'sequences':0,'frames':0,'failures':0,'fr_frame_weighted':None,'fr_sequence_mean':None}
        n=sum(r['frames'] for r in group); f=sum(r['failures'] for r in group)
        return {'attribute':key,'sequences':len(group),'frames':n,'failures':f,'fr_frame_weighted':f/n,'fr_sequence_mean':float(np.mean([r['fr'] for r in group]))}
    summary=[aggregate(k,[r for r in rows if k in r['attributes'].split()]) for k in ATTRIBUTES]; overall=aggregate('overall',rows)
    csv_write(o/'sequence_metrics.csv',rows); csv_write(o/'attribute_metrics.csv',summary+[overall])
    (o/'protocol.json').write_text(json.dumps({'model':a.name,'manifest':str(a.manifest),'weights':str(a.weights) if a.weights else None,'predictions_dir':str(a.predictions_dir) if a.predictions_dir else None,'protocol':'Valid xywh GT only, excludes initialization; sequence-level attribute membership, no frame-level attribute masks; failure IoU=0. Exact paper aggregation not established.'},indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(11,4.7)); x=[i for i,r in enumerate(summary) if r['fr_frame_weighted'] is not None]; bars=ax.bar(x,[summary[i]['fr_frame_weighted'] for i in x],color='#b91c1c'); ax.bar_label(bars,fmt='%.3f',fontsize=9,padding=3); ax.set_xticks(range(len(ATTRIBUTES)),ATTRIBUTES); ax.axhline(overall['fr_frame_weighted'],linestyle='--',label='Overall FR'); ax.legend(); ax.set(title=f'{a.name} — Local LaSOT attribute robustness',ylabel='Failure rate',xlabel='Attribute'); ax.margins(y=.20); save(fig,o,'lasot_robustness',f'Local frame-weighted FR | Valid GT, initialization excluded | Missing attributes omitted; {len(rows)} sequences.')
    print(f'Wrote local robustness report: {o}')
if __name__=='__main__': main()
