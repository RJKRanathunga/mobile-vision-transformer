"""Compare locally evaluated trackers using GOT-10k toolkit performance.json files."""
import argparse
from pathlib import Path
from report_common import read_json, number, grouped, csv_write, plt, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',nargs='+',type=Path,required=True,help='Paths to performance.json; each may contain multiple trackers')
    p.add_argument('--protocol',required=True,help='Describe the identical dataset split, sequence list and hardware used for all runs')
    p.add_argument('--output-dir',type=Path,default=Path('reports/comparison'))
    a=p.parse_args(); rows=[]; names=set(); curves={}
    for path in a.reports:
        for name,result in read_json(path).items():
            if not isinstance(result,dict) or not isinstance(result.get('overall'),dict): continue
            if name in names: raise ValueError(f'Duplicate tracker name {name}; use distinct --name at evaluation')
            names.add(name); data=result['overall']; sr=data.get('sr'); sr50=number(data.get('sr50')); sr75=number(data.get('sr75'))
            if sr50 is None: sr50=number(data.get('sr_0.5'))
            if sr75 is None: sr75=number(data.get('sr_0.75'))
            if isinstance(sr,dict):
                if sr50 is None: sr50=number(sr.get('0.5',sr.get('0.50')))
                if sr75 is None: sr75=number(sr.get('0.75'))
            # Upstream toolkit uses a uniform [0,1] grid and strict IoU > threshold.
            # Only read SR75 if 0.75 exists exactly in the provided grid.
            curve=data.get('succ_curve', data.get('success_curve'))
            valid_curve=isinstance(curve,list) and len(curve)>1 and all(number(x) is not None and 0<=x<=1 for x in curve)
            if sr50 is None: sr50=number(sr)
            if sr75 is None and valid_curve and (len(curve)-1)%4==0:
                sr75=number(curve[3*(len(curve)-1)//4])
            fps=number(data.get('speed_fps'))
            if fps is not None and fps<=0: fps=None
            row={'tracker':name,'ao':number(data.get('ao')),'sr50':sr50,'sr75':sr75,'fps':fps,'source':str(path),'protocol':a.protocol}
            rows.append(row)
            if valid_curve: curves[name]=curve
    if not rows: raise ValueError('No tracker overall statistics found')
    for row in rows:
        for k in ('ao','sr50','sr75'):
            if row[k] is not None and not 0<=row[k]<=1: raise ValueError(f'{row["tracker"]}: {k} outside fraction range [0,1]')
    note=f'Local results only | User-specified common protocol: {a.protocol}'
    names=[r['tracker'] for r in rows]; o=a.output_dir
    csv_write(o/'comparison.csv',rows)
    metrics={label:[r[key]*100 if r[key] is not None else None for r in rows] for key,label in [('ao','AO'),('sr50','SR0.50'),('sr75','SR0.75')] if any(r[key] is not None for r in rows)}
    if metrics: grouped(o,'accuracy','Local tracker comparison',names,metrics,'%',note+' | Missing metrics omitted.',(0,100))
    if any(r['fps'] is not None for r in rows): grouped(o,'fps','Local tracking speed',names,{'FPS':[r['fps'] for r in rows]},'FPS',note)
    if curves:
        fig,ax=plt.subplots(figsize=(8,4.7))
        for name,curve in curves.items(): ax.plot([i/(len(curve)-1) for i in range(len(curve))],curve,label=name)
        ax.set(title='GOT-10k success curves',xlabel='Overlap threshold',ylabel='Success rate',xlim=(0,1),ylim=(0,1.02)); ax.legend(); save(fig,o,'success_comparison',note+' | Toolkit success curves, uniform threshold grid.')
    print(f'Wrote comparison: {o}; verify all input runs really used the same protocol.')

if __name__=='__main__': main()
