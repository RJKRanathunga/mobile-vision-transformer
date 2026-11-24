"""Shared presentation plotting helpers (Matplotlib, non-interactive)."""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.dpi': 120, 'axes.grid': True, 'grid.alpha': .22,
                     'axes.axisbelow': True, 'savefig.facecolor': 'white'})

PAPER = 'Gopal & Amer, BMVC 2023 | https://papers.bmvc2023.org/0800.pdf'
ATTRIBUTES = ['ARC','BC','CM','DEF','FM','FOC','IV','LR','MB','OV','POC','ROT','SV','VC']

def save(fig, out, stem, note=None):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    if note:
        fig.text(.01, .012, note, fontsize=8, color='#475569', wrap=True)
    fig.tight_layout(rect=(0, .065 if note else 0, 1, 1))
    for ext in ('png','pdf'):
        fig.savefig(out / f'{stem}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


def csv_write(path, rows):
    if not rows: raise ValueError('No rows to export')
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int,float)) or not math.isfinite(value):
        return None
    return float(value)


def grouped(out, stem, title, names, series, ylabel, note, ylim=None):
    fig, ax = plt.subplots(figsize=(max(8, len(names)*1.25), 4.7))
    width = .78 / len(series)
    for j, (label, vals) in enumerate(series.items()):
        xs = [i+(j-(len(series)-1)/2)*width for i,v in enumerate(vals) if v is not None]
        ys = [v for v in vals if v is not None]
        bars = ax.bar(xs, ys, width=width, label=label)
        ax.bar_label(bars, fmt='%.2f' if ylabel == 'Fraction' else '%.1f', fontsize=8, padding=2)
    ax.set_xticks(range(len(names)), names, rotation=20 if len(names)>5 else 0, ha='right' if len(names)>5 else 'center')
    ax.set(title=title, ylabel=ylabel)
    if ylim: ax.set_ylim(*ylim)
    else: ax.margins(y=.17)
    if len(series)>1: ax.legend(loc='best')
    save(fig,out,stem,note)
