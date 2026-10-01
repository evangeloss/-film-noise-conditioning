"""Kaggle launcher for the clean-PhysicsBridge denoiser experiment.

Paste this file into one Kaggle cell or run it as a script. It trains ONLY the
large-deformation hybrid model with clean-physics denoising supervision, then evaluates it together with the physics baseline over the requested SNR sweep.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
REPO_URL='https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git'
BRANCH=''
PROJECT_SUBDIR=''
QUICK=False

# Set this to the large-deformation value you want to study.
LARGE_ALPHA=1.0

SEEDS=[11]
MAX_EPOCHS=100
STEPS_PER_EPOCH=64
BATCH_SIZE=32
VALIDATION_SCENES=256
TEST_SCENES=500
QUADRATURE_ATOMS=1024
BRIDGE_RANK=384
WIDTH=32

TRAINING_SNRS=[-10,-5,0,5,10,15,20]
TRAINING_SNR_PROBS=[0.25,0.25,0.20,0.15,0.07,0.04,0.04]
TEST_SNRS=[-10,-5,0,5,10,15,20]

RESUME_OUTPUT=''

# Optional: point this to the previous low-SNR+FiLM results folder to run a paired
# old-vs-new comparison automatically after training. Leave empty to skip.
BASELINE_RESULTS_DIR=''


def main():
    u=urlparse(REPO_URL)
    if (u.scheme!='https' or u.hostname!='github.com' or u.username or u.password or u.query or u.fragment
        or 'YOUR_' in REPO_URL or len(u.path.strip('/').split('/'))!=2):
        raise ValueError('Set REPO_URL to your public GitHub repository URL.')

    working=Path('/kaggle/working')
    if not working.exists():
        raise RuntimeError('This launcher is intended for Kaggle.')

    folder=Path(tempfile.mkdtemp(prefix='clean_physics_denoiser_',dir=working))
    repo=folder/'repo'
    output=Path(RESUME_OUTPUT).resolve() if RESUME_OUTPUT else folder/'results_clean_physics_denoiser'
    evaluation=folder/'evaluation_clean_physics_denoiser'

    if RESUME_OUTPUT and (not output.is_relative_to(working.resolve()) or not output.is_dir()):
        raise ValueError('RESUME_OUTPUT must be an existing results folder under /kaggle/working')

    env=os.environ.copy()
    env.update(GIT_TERMINAL_PROMPT='0',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')

    clone=['git','clone','--depth','1']+(['--branch',BRANCH] if BRANCH else [])+['--',REPO_URL,str(repo)]
    subprocess.run(clone,env=env,check=True)

    if PROJECT_SUBDIR:
        project=(repo/PROJECT_SUBDIR).resolve()
        candidates=[project] if (project/'train.py').is_file() and (project/'bridge.py').is_file() else []
    else:
        candidates=[p.parent for p in repo.rglob('train.py') if (p.parent/'bridge.py').is_file() and (p.parent/'simulator.py').is_file()]
    if len(candidates)!=1:
        raise RuntimeError('Could not uniquely locate the project folder. Set PROJECT_SUBDIR if needed.')
    project=candidates[0]

    subprocess.run([sys.executable,'-m','pip','install','--quiet','-r',str(project/'requirements.txt')],check=True)

    import torch
    print('Device:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    print('Training SNR distribution:',dict(zip(TRAINING_SNRS,TRAINING_SNR_PROBS)))
    print('Large deformation b/lambda:',LARGE_ALPHA)
    print('FiLM conditioning: enabled')
    print('Training target: clean PhysicsBridge output')

    subprocess.run([sys.executable,str(project/'verify.py')],cwd=project,env=env,check=True)

    train_command=[
        sys.executable,'-u',str(project/'train.py'),
        '--output',str(output),
        '--train-only',
        '--regimes','large',
        '--model-kinds','hybrid',
        '--large-alpha',str(LARGE_ALPHA),
        '--epochs',str(MAX_EPOCHS),
        '--steps',str(STEPS_PER_EPOCH),
        '--batch-size',str(BATCH_SIZE),
        '--validation-scenes',str(VALIDATION_SCENES),
        '--test-scenes',str(TEST_SCENES),
        '--atoms',str(QUADRATURE_ATOMS),
        '--rank',str(BRIDGE_RANK),
        '--width',str(WIDTH),
        '--seeds',*map(str,SEEDS),
        '--training-snrs',*map(str,TRAINING_SNRS),
        '--training-snr-probs',*map(str,TRAINING_SNR_PROBS),
        '--test-snrs',*map(str,TEST_SNRS),
        '--noise-conditioning','film',
        '--training-target','clean_physics',
    ]
    if QUICK:
        train_command+=['--quick']
    if RESUME_OUTPUT:
        train_command+=['--resume']

    success=False
    try:
        with (folder/'train.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(train_command,cwd=project,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            try:
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                code=process.wait()
            finally:
                if process.poll() is None:
                    process.terminate();process.wait()
            if code:
                raise subprocess.CalledProcessError(code,train_command)

        eval_command=[
            sys.executable,'-u',str(project/'evaluate_sweep.py'),
            '--results',str(output),
            '--output',str(evaluation),
            '--snrs',*map(str,TEST_SNRS),
            '--fixed-snrs','-10','-5','0','10','20',
            '--scenes',str(TEST_SCENES),
            '--batch-size',str(BATCH_SIZE),
            '--seeds',*map(str,SEEDS),
        ]
        if QUICK:
            eval_command+=['--allow-smoke']
        subprocess.run(eval_command,cwd=project,env=env,check=True)

        if BASELINE_RESULTS_DIR:
            baseline=Path(BASELINE_RESULTS_DIR).resolve()
            comparison=folder/'comparison_old_vs_clean_physics'
            compare_command=[
                sys.executable,'-u',str(project/'compare_denoising_runs.py'),
                '--baseline',str(baseline),
                '--candidate',str(output),
                '--output',str(comparison),
                '--regime','large',
                '--seed',str(SEEDS[0]),
                '--scenes',str(TEST_SCENES),
                '--batch-size',str(BATCH_SIZE),
                '--snrs',*map(str,TEST_SNRS),
            ]
            if QUICK:
                compare_command+=['--allow-smoke']
            subprocess.run(compare_command,cwd=project,env=env,check=True)
        success=True
    finally:
        output.mkdir(parents=True,exist_ok=True)
        if (folder/'train.log').is_file():
            shutil.copy2(folder/'train.log',output/f'launcher_{folder.name}.log')
        provenance={
            'repo':REPO_URL,
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
            'completed':success,
            'command':train_command,
            'large_alpha':LARGE_ALPHA,
            'training_snrs':TRAINING_SNRS,
            'training_snr_probs':TRAINING_SNR_PROBS,
            'test_snrs':TEST_SNRS,
            'noise_conditioning':'film',
            'training_target':'clean_physics',
        }
        (output/f'provenance_{folder.name}.json').write_text(json.dumps(provenance,indent=2))

        import zipfile
        review_zip=folder/'clean_physics_denoiser_for_review.zip'
        with zipfile.ZipFile(review_zip,'w',zipfile.ZIP_DEFLATED) as z:
            roots=[output,evaluation]
            comparison=folder/'comparison_old_vs_clean_physics'
            if comparison.exists(): roots.append(comparison)
            for root in roots:
                if not root.exists():
                    continue
                for f in root.rglob('*'):
                    if f.is_file() and f.suffix in ('.csv','.json','.md','.log','.png','.pdf','.svg'):
                        z.write(f,Path(root.name)/f.relative_to(root))
        print('\nReview ZIP:',review_zip)
        print('Full checkpoint folder:',output)
        if evaluation.exists():
            print('Evaluation folder:',evaluation)

        try:
            from IPython.display import FileLink,display
            display(FileLink(str(review_zip.relative_to(working))))
        except Exception:
            pass

    print('\nExperiment complete.')


if __name__=='__main__':
    main()
