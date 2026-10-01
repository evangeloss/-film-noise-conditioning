"""Paired comparison of an old FiLM hybrid and a clean-physics denoiser.

Both checkpoints are evaluated on exactly the same propagation/noise draws.
The script reports true undeformed-channel NMSE and distance to the clean
PhysicsBridge target, which shows whether the new model is actually a better
denoiser even when the final true-channel gain is modest.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from simulator import Simulator,pack
from bridge import PhysicsBridge
from model import Reconstructor,nmse
from train import clean_physics_target,load_state


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def same_state(a,b):
    return a.keys()==b.keys() and all(torch.equal(a[k].cpu(),b[k].cpu()) for k in a)


def load_run(root,regime,seed,device):
    cfg=json.loads((root/'configuration.json').read_text())
    b=torch.load(root/f'bridge_{regime}.pt',map_location=device,weights_only=True)
    bridge=PhysicsBridge(b['U'],b['T'],b['eigenvalues']).to(device).eval()
    model=Reconstructor(bridge,True,cfg['width'],cfg.get('noise_conditioning','none')).to(device)
    ck=torch.load(root/f'seed_{seed}'/f'{regime}_hybrid'/'best.pt',map_location=device,weights_only=True)
    load_state(model,ck['model']);model.eval()
    return cfg,b,bridge,model,ck


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True,help='Previous low-SNR+FiLM results directory')
    p.add_argument('--candidate',type=Path,required=True,help='New clean-physics denoiser results directory')
    p.add_argument('--output',type=Path,default=Path('comparison_old_vs_clean_physics'))
    p.add_argument('--regime',choices=['large','small'],default='large')
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--snrs',nargs='+',type=float,default=[-10,-5,0,5,10,15,20])
    p.add_argument('--scenes',type=int,default=500)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--evaluation-seed',type=int,default=3300000000)
    p.add_argument('--allow-smoke',action='store_true')
    a=p.parse_args()

    for root in [a.baseline,a.candidate]:
        if not (root/'configuration.json').is_file():p.error(f'Missing configuration.json in {root}')
    if a.output.exists() and any(a.output.iterdir()):p.error('Use a fresh output directory')
    if a.scenes<1 or a.batch_size<1 or any(not math.isfinite(x) for x in a.snrs):p.error('Invalid scenes/batch/SNR')

    base_cfg=json.loads((a.baseline/'configuration.json').read_text())
    cand_cfg=json.loads((a.candidate/'configuration.json').read_text())
    if (base_cfg.get('quick') or cand_cfg.get('quick')) and not a.allow_smoke:
        p.error('Smoke checkpoints require --allow-smoke')
    for key in ['paths','geometry_seed','pair_start']:
        if base_cfg[key]!=cand_cfg[key]:p.error(f'{key} differs between runs')
    alpha0=float(base_cfg[f'{a.regime}_alpha']);alpha1=float(cand_cfg[f'{a.regime}_alpha'])
    if abs(alpha0-alpha1)>1e-12:p.error(f'Deformation differs: baseline={alpha0}, candidate={alpha1}')

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sim=Simulator(base_cfg['geometry_seed'],base_cfg['paths'],base_cfg['pair_start']).to(device)
    g0=torch.load(a.baseline/'geometry.pt',map_location=device,weights_only=True)
    g1=torch.load(a.candidate/'geometry.pt',map_location=device,weights_only=True)
    if not same_state(g0,g1):p.error('Saved geometry differs between runs')
    sim.load_state_dict(g0)

    cfg0,b0,bridge0,model0,ck0=load_run(a.baseline,a.regime,a.seed,device)
    cfg1,b1,bridge1,model1,ck1=load_run(a.candidate,a.regime,a.seed,device)
    bridge_keys=['U','T','eigenvalues']
    if not same_state({k:b0[k] for k in bridge_keys},{k:b1[k] for k in bridge_keys}):
        p.error('PhysicsBridge differs between runs; paired model comparison would be confounded')

    rows=[];per_scene=[]
    with torch.inference_mode():
        for snr in a.snrs:
            values={'physics_true':[],'baseline_true':[],'candidate_true':[],
                    'baseline_cleanphys':[],'candidate_cleanphys':[]}
            for j,start in enumerate(range(0,a.scenes,a.batch_size)):
                n=min(a.batch_size,a.scenes-start)
                data=sim.batch(n,a.evaluation_seed+j,alpha0,snr,return_clean=True)
                physics=bridge0(data['x'],data['variance'])
                old=model0(data['x'],data['variance'],data['scale'])
                new=model1(data['x'],data['variance'],data['scale'])
                clean_target=clean_physics_target(bridge0,data)
                batch_values={
                    'physics_true':nmse(physics,data['y']),
                    'baseline_true':nmse(old,data['y']),
                    'candidate_true':nmse(new,data['y']),
                    'baseline_cleanphys':nmse(old,clean_target),
                    'candidate_cleanphys':nmse(new,clean_target),
                }
                for key,t in batch_values.items():
                    arr=t.cpu().numpy();values[key].extend(arr.tolist())
                    for i,e in enumerate(arr):
                        per_scene.append(dict(snr_db=snr,scene_id=start+i,metric=key,nmse=float(e)))
            means={k:float(np.mean(v)) for k,v in values.items()}
            db=lambda x:10*math.log10(max(x,1e-30))
            row=dict(
                snr_db=float(snr),
                physics_db=db(means['physics_true']),
                baseline_hybrid_db=db(means['baseline_true']),
                clean_physics_denoiser_db=db(means['candidate_true']),
                baseline_gain_over_physics_db=db(means['physics_true'])-db(means['baseline_true']),
                candidate_gain_over_physics_db=db(means['physics_true'])-db(means['candidate_true']),
                candidate_improvement_over_baseline_db=db(means['baseline_true'])-db(means['candidate_true']),
                baseline_to_clean_physics_db=db(means['baseline_cleanphys']),
                candidate_to_clean_physics_db=db(means['candidate_cleanphys']),
                denoising_target_improvement_db=db(means['baseline_cleanphys'])-db(means['candidate_cleanphys']),
            )
            rows.append(row)
            print(f"SNR {snr:>5g}: physics {row['physics_db']:.3f} dB | old {row['baseline_hybrid_db']:.3f} | "
                  f"new {row['clean_physics_denoiser_db']:.3f} | new-old {row['candidate_improvement_over_baseline_db']:+.3f} dB | "
                  f"denoise-target gain {row['denoising_target_improvement_db']:+.3f} dB",flush=True)

    a.output.mkdir(parents=True,exist_ok=True)
    write_csv(a.output/'summary.csv',rows);write_csv(a.output/'per_scene_errors.csv',per_scene)
    metadata={'baseline':str(a.baseline.resolve()),'candidate':str(a.candidate.resolve()),'regime':a.regime,
              'alpha':alpha0,'seed':a.seed,'scenes':a.scenes,'snrs':a.snrs,'evaluation_seed':a.evaluation_seed,
              'baseline_training_target':cfg0.get('training_target','true_channel'),
              'candidate_training_target':cfg1.get('training_target','true_channel'),
              'baseline_best_epoch':ck0.get('epoch'),'candidate_best_epoch':ck1.get('epoch')}
    (a.output/'metadata.json').write_text(json.dumps(metadata,indent=2))

    x=[r['snr_db'] for r in rows]
    fig,ax=plt.subplots(figsize=(8.5,5.5))
    ax.plot(x,[r['physics_db'] for r in rows],'o-',label='Physics')
    ax.plot(x,[r['baseline_hybrid_db'] for r in rows],'s-',label='Previous FiLM hybrid')
    ax.plot(x,[r['clean_physics_denoiser_db'] for r in rows],'D-',label='Clean-physics denoiser')
    ax.set(xlabel='SNR (dB)',ylabel='NMSE (dB)',title=f'Paired true-channel NMSE, b/lambda={alpha0:g}')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(a.output/f'true_nmse_comparison.{ext}',dpi=200)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(8.5,5))
    ax.plot(x,[r['candidate_improvement_over_baseline_db'] for r in rows],'o-',label='New vs previous hybrid')
    ax.plot(x,[r['denoising_target_improvement_db'] for r in rows],'s--',label='New vs previous on clean-physics target')
    ax.axhline(0,color='black',linewidth=1,linestyle='--')
    ax.set(xlabel='SNR (dB)',ylabel='Improvement (dB)',title='Effect of clean-physics denoising supervision')
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(a.output/f'improvement_comparison.{ext}',dpi=200)
    plt.close(fig)
    print('Saved comparison to',a.output.resolve())


if __name__=='__main__':main()
