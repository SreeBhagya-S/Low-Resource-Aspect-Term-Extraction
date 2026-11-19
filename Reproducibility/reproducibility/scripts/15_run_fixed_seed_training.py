#!/usr/bin/env python3
"""Run all five predeclared baseline seeds using the original training algorithm.

By default requires CUDA to avoid unplanned lengthy CPU fine-tunes. Pass --allow-cpu
explicitly to permit CPU. Outputs are separate; never overwrite existing runs.
"""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',required=True)
    ap.add_argument('--plan',required=True)
    ap.add_argument('--allow-cpu',action='store_true')
    ap.add_argument('--require-cuda',action='store_true')
    ap.add_argument('--only-repeat',type=int,default=None,help='Resume one predeclared repeat, 0..4')
    args=ap.parse_args()
    if args.allow_cpu and args.require_cuda:raise ValueError('Choose only one of --allow-cpu and --require-cuda')
    try:
        import torch,transformers,datasets,accelerate,seqeval
    except ImportError as exc:
        raise SystemExit(f'Missing dependency: {exc}. Install requirements in the training environment.')
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit('CUDA unavailable. Use an NVIDIA GPU training environment, or explicitly pass --allow-cpu.')
    src=Path(args.source_root).resolve();planfile=Path(args.plan).resolve()
    out=planfile.parent;plan=json.loads(planfile.read_text())
    manifest=out/'low_resource_baseline_manifest.csv'
    verified={
       'data/dual_regime_protocol_manifest.csv':src/'data/dual_regime_protocol_manifest.csv',
       'data/fixed_test_manifest_seed42.csv':src/'data/fixed_test_manifest_seed42.csv',
       'config/transformer_config.json':src/'config/transformer_config.json',
       'code/train_transformer_baselines.py':src/'code/train_transformer_baselines.py',
       'data/evaluation_gold.csv':src/'data/evaluation_gold.csv',
    }
    for rel,p in verified.items():
        if digest(p)!=plan['source_sha256'][rel]:raise SystemExit(f'Source changed since seed plan: {p}')
    if digest(manifest)!=plan['train_manifest_sha256']:raise SystemExit('Training split manifest changed')
    # Only non-algorithmic alteration: persist restored best-validation model
    # from predeclared seed 3003/3004/3005 for a genuinely new future test.
    archive_trainer=Path(__file__).resolve().parent/'train_transformer_baselines_archive_model.py'
    if not archive_trainer.is_file():raise FileNotFoundError(archive_trainer)
    env_record={'python':sys.version,'torch':torch.__version__,'cuda':torch.version.cuda,
                'cuda_available':torch.cuda.is_available(),'transformers':transformers.__version__,
                'datasets':datasets.__version__,'accelerate':accelerate.__version__,
                'trainer_original_sha256':digest(verified['code/train_transformer_baselines.py']),
                'trainer_archive_copy_sha256':digest(archive_trainer)}
    (out/'training_environment.json').write_text(json.dumps(env_record,indent=2)+'\n')
    print('Environment:',env_record)
    for rep in plan['repeats']:
        if args.only_repeat is not None and rep['repeat']!=args.only_repeat:continue
        rp=out/'runs'/f'repeat_{rep["repeat"]:02d}_seed_{rep["base_seed"]}'
        if (rp/'baseline_metrics.csv').exists():
            print('ALREADY COMPLETE, skipping:',rp)
            continue
        if rp.exists() and any(rp.iterdir()):
            raise SystemExit(f'Nonempty incomplete output directory; inspect and explicitly move before rerunning: {rp}')
        rp.mkdir(parents=True,exist_ok=True)
        cmd=[sys.executable,str(archive_trainer),'--gold',str(src/'data/evaluation_gold.csv'),
             '--split-manifest',str(manifest),'--test-manifest',str(src/'data/fixed_test_manifest_seed42.csv'),
             '--config',str(out/rep['config_file']),'--output-dir',str(rp)]
        (rp/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
        print('Executing pre-registered repeat',rep['repeat'],rep['model_seeds'],flush=True)
        with (rp/'train.log').open('w',encoding='utf8') as log:
            proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=False)
        if proc.returncode:
            raise SystemExit(f'Run failed (exit {proc.returncode}): {rp/"train.log"}')
        print('COMPLETED:',rp)
    print('Run 16_summarize_seed_runs.py after all five runs.')
if __name__=='__main__':main()
