"""Reconstruct published BMVC 2023 tables 1–3 and Figure 2, not local measurements."""
import argparse
from pathlib import Path
from report_common import PAPER, ATTRIBUTES, grouped, csv_write, plt, save

LIGHT = [
 ('LightTrack',.582,.668,.442,72.9,79.3,69.9,.581,.146,.524,.116,99),
 ('Stark-Lightning',.596,.696,.479,72.7,77.9,67.4,.619,.111,.585,.151,205),
 ('FEAR-XS',.573,.681,.455,71.5,80.5,69.9,.487,.207,.508,.273,275),
 ('E.T.Track',.566,.646,.425,74.0,79.8,69.8,.589,.172,.597,.162,53),
 ('MVT',.633,.742,.551,74.8,81.5,71.9,.603,.085,.553,.137,175)]
HEAVY = [('DiMP-50',.611,.717,74,80.1,26.1,61.5,15),('TransT',.671,.768,81.2,85.4,23,87.7,2.3),('STARK-ST101',.688,.781,82,86.9,47.2,80,7.8),('OSTrack-384',.740,.835,83.9,88.5,92.1,74.4,4.4),('MixFormer-L',.756,.857,83.9,88.9,183.9,45.2,None),('MVT',.633,.742,74.8,81.5,5.5,175,29.4)]
ABLATION = [('Without fusion',.600,.703,74.9,80,.566,.122,.544,.163),('With fusion',.633,.742,74.8,81.5,.603,.085,.553,.137)]
FR=[.145,.211,.153,.103,.249,.207,.121,.217,.176,.180,.161,.126,.133,.110]

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output-dir',type=Path,default=Path('reports/paper')); a=p.parse_args(); o=a.output_dir
    def bars(stem,title,rows,cols,table,ylim=None):
        grouped(o,stem,title,[r[0] for r in rows],{label:[r[i]*scale if r[i] is not None else None for r in rows] for label,i,scale in cols},'%' if ylim else 'FPS',f'Published results, {table} | {PAPER}',ylim)
    bars('table1_got10k','Lightweight trackers — GOT-10k test',LIGHT,[('AO',1,100),('SR0.50',2,100),('SR0.75',3,100)],'Table 1',(0,100))
    bars('table1_trackingnet','Lightweight trackers — TrackingNet test',LIGHT,[('AUC',4,1),('Normalized precision',5,1),('Precision',6,1)],'Table 1',(0,100))
    bars('table1_nfs30','Lightweight trackers — NfS30',LIGHT,[('AUC',7,100),('Failure rate (lower better)',8,100)],'Table 1',(0,100))
    bars('table1_lasot','Lightweight trackers — LaSOT test',LIGHT,[('AUC',9,100),('Failure rate (lower better)',10,100)],'Table 1',(0,100))
    bars('table1_gpu_fps','Lightweight trackers — reported GPU speed',LIGHT,[('GPU',11,1)],'Table 1')
    bars('table2_got10k','Heavyweight comparison — GOT-10k test',HEAVY,[('AO',1,100),('SR0.50',2,100)],'Table 2',(0,100))
    bars('table2_trackingnet','Heavyweight comparison — TrackingNet test',HEAVY,[('AUC',3,1),('Normalized precision',4,1)],'Table 2',(0,100))
    grouped(o,'table2_parameters','Model complexity',[r[0] for r in HEAVY],{'Parameters':[r[5] for r in HEAVY]},'Millions',f'Published Table 2 | {PAPER}')
    grouped(o,'table2_gpu_cpu_fps','Reported inference speed',[r[0] for r in HEAVY],{'GPU':[r[6] for r in HEAVY],'CPU':[r[7] for r in HEAVY]},'FPS',f'Published Table 2 | MixFormer-L CPU reported <5 FPS; omitted from bars | {PAPER}')
    for stem,metrics,indices,scales in [('got10k',['AO','SR0.50'],[1,2],[100,100]),('trackingnet',['AUC','Normalized precision'],[3,4],[1,1]),('nfs30',['AUC','Failure rate'],[5,6],[100,100]),('lasot',['AUC','Failure rate'],[7,8],[100,100])]:
        grouped(o,'table3_ablation_'+stem,'Backbone fusion ablation — '+stem,metrics,{r[0]:[r[i]*s for i,s in zip(indices,scales)] for r in ABLATION},'%',f'Published Table 3 | {PAPER}',(0,100))
    fig,ax=plt.subplots(figsize=(11,4.7)); bars_=ax.bar(ATTRIBUTES,FR,color='#b91c1c'); ax.bar_label(bars_,fmt='%.3f',padding=3,fontsize=9); ax.axhline(.137,color='#334155',linestyle='--',label='Reported overall FR = 0.137'); ax.legend(); ax.set(title='MVT robustness across LaSOT attributes',ylabel='Failure rate',xlabel='LaSOT attribute',ylim=(0,.30)); save(fig,o,'figure2_lasot_robustness',f'Reconstructed published Figure 2 | {PAPER}')
    for filename,rows,header in [('table1',LIGHT,'tracker ao sr50 sr75 tn_auc_pct tn_pnorm_pct tn_precision_pct nfs_auc nfs_fr lasot_auc lasot_fr gpu_fps'),('table2',HEAVY,'tracker ao sr50 tn_auc_pct tn_pnorm_pct params_m gpu_fps cpu_fps'),('table3',ABLATION,'fusion ao sr50 tn_auc_pct tn_pnorm_pct nfs_auc nfs_fr lasot_auc lasot_fr')]:
        data=[dict(zip(header.split(),r)) for r in rows]
        if filename=='table2': data[4]['cpu_fps']='<5'
        csv_write(o/(filename+'.csv'),data)
    csv_write(o/'figure2.csv',[{'attribute':k,'failure_rate':v} for k,v in zip(ATTRIBUTES,FR)])
    print(f'Published paper charts written to {o}')
if __name__=='__main__': main()
