"""Generate training curves from train.py's metrics.jsonl, including sparse validation."""
import argparse
import json
from pathlib import Path
from report_common import plt, save, csv_write, number

METRICS = {'loss':'Total loss','focal':'Focal loss','l1':'L1 loss','giou':'GIoU loss','iou':'Mean pairwise IoU'}

def load_metrics(path):
    rows = {}; lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    for i,line in enumerate(lines,1):
        if not line.strip(): continue
        try: item = json.loads(line)
        except json.JSONDecodeError as exc: raise ValueError(f'Invalid JSON on line {i}; retry after epoch logging completes') from exc
        epoch = item['epoch']
        if not isinstance(epoch,int) or epoch<1 or not isinstance(item.get('train'),dict): raise ValueError(f'Invalid metrics at line {i}')
        if epoch in rows: print(f'Warning: duplicate epoch {epoch}; using last logged row')
        rows[epoch] = item
    if not rows: raise ValueError('No completed epochs in metrics file')
    return [rows[e] for e in sorted(rows)]


def draw(ax, rows, metric):
    for split,label,color in [('train','Training','#2563eb'),('val','Validation','#ea580c')]:
        vals = [(r['epoch'],number((r.get(split) or {}).get(metric))) for r in rows]
        vals = [(e,v) for e,v in vals if v is not None]
        if vals:
            x,y = zip(*vals); ax.plot(x,y,marker='o' if split=='val' or len(x)==1 else None,markersize=4,label=label,color=color,linestyle='--' if split=='val' else '-')
    ax.set(title=METRICS[metric],xlabel='Epoch',ylabel='IoU' if metric=='iou' else 'Loss')
    if metric=='iou': ax.set_ylim(0,1.02)
    if ax.lines: ax.legend()


def learning_rate(ax,rows):
    for i,label in enumerate(('Neck / head','Backbone')):
        vals = [(r['epoch'],number(r['lr'][i])) for r in rows if isinstance(r.get('lr'),list) and len(r['lr'])>i]
        vals = [(e,v) for e,v in vals if v is not None]
        if vals:
            x,y = zip(*vals); ax.plot(x,y,marker='o' if len(x)==1 else None,label=label)
    ax.set(title='Learning rate schedule',xlabel='Epoch',ylabel='Learning rate')
    ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    if ax.lines: ax.legend()


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--metrics',type=Path,default=Path('checkpoints/mvt_got10k/metrics.jsonl')); p.add_argument('--output-dir',type=Path); p.add_argument('--name'); a=p.parse_args()
    rows=load_metrics(a.metrics); name=a.name or a.metrics.parent.name
    out=a.output_dir or Path('reports/training')/name
    note=f'Local run: {name} | Validation markers are measured epochs only; no missing values fabricated.'
    for key,stem in [('loss','total_loss'),('focal','focal_loss'),('l1','l1_loss'),('giou','giou_loss'),('iou','mean_iou')]:
        fig,ax=plt.subplots(figsize=(8,4.6)); draw(ax,rows,key); save(fig,out,stem,note)
    fig,ax=plt.subplots(figsize=(8,4.6)); learning_rate(ax,rows); save(fig,out,'learning_rate',note)
    fig,axes=plt.subplots(2,3,figsize=(14,8));
    for ax,key in zip(axes.flat,METRICS): draw(ax,rows,key)
    learning_rate(axes.flat[-1],rows); fig.suptitle(f'MVT training — {name}',fontsize=16); save(fig,out,'training_summary',note)
    flat=[]
    for r in rows:
        row={'epoch':r['epoch']}
        for split in ('train','val'):
            for key in METRICS: row[f'{split}_{key}']=number((r.get(split) or {}).get(key))
        for i,key in enumerate(('lr_non_backbone','lr_backbone')): row[key]=number(r['lr'][i]) if isinstance(r.get('lr'),list) and len(r['lr'])>i else None
        flat.append(row)
    csv_write(out/'metrics.csv',flat); print(f'Wrote training report: {out} ({len(rows)} epochs)')

if __name__=='__main__': main()
