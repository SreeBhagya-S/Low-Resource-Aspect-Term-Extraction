#!/usr/bin/env python3
"""Pre-register five reproducible low-resource transformer runs without training.

Uses the original 64/16 from dual_regime_protocol_manifest.csv; leaves fixed test
and code unchanged. Outputs seed configs, manifest and plan in outdir.
"""
import argparse,hashlib,json
from pathlib import Path
import pandas as pd

BASE_SEEDS=(3003,4003,5003,6003,7003)
MODELS=('mbert','indicbert','xlmr')

def hashfile(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-root',required=True,help='Original GitHub_R2 folder')
    ap.add_argument('--outdir',default='R4_repeated_seed_runs')
    args=ap.parse_args()
    root=Path(args.source_root).resolve();out=Path(args.outdir).resolve();out.mkdir(parents=True,exist_ok=True)
    source_manifest=root/'data/dual_regime_protocol_manifest.csv'
    test_file=root/'data/fixed_test_manifest_seed42.csv'
    cfg_file=root/'config/transformer_config.json'
    trainer_file=root/'code/train_transformer_baselines.py'
    gold_file=root/'data/evaluation_gold.csv'
    for p in [source_manifest,test_file,cfg_file,trainer_file,gold_file]:
        if not p.is_file():raise FileNotFoundError(str(p))
    protocol=pd.read_csv(source_manifest)
    test=pd.read_csv(test_file)
    idx=lambda x: set(protocol.loc[protocol['Low_Resource_Role']==x,'Original_Row_Index'].astype(int))
    train,valid,held=idx('train'),idx('validation'),idx('test')
    tidx=set(test['Original_Row_Index'].astype(int))
    assert (len(train),len(valid),len(held),len(tidx))==(64,16,200,200)
    assert held==tidx and not(train&valid or train&tidx or valid&tidx)
    assert len(train|valid|held)==280
    chosen=protocol[protocol.Low_Resource_Role.isin(['train','validation'])][['Original_Row_Index','Low_Resource_Role']].copy()
    chosen['Role']=chosen.Low_Resource_Role.map({'train':'supervised_train','validation':'supervised_validation'})
    chosen=chosen[['Original_Row_Index','Role']].sort_values('Original_Row_Index')
    manifest=out/'low_resource_baseline_manifest.csv'
    chosen.to_csv(manifest,index=False)
    cfg=json.loads(cfg_file.read_text(encoding='utf8'))
    assert cfg['models']==list(MODELS),f'Expected all three models for common-prefix protocol: {cfg["models"]}'
    assert cfg['selection_metric']=='eval_loss'
    seed_plan=[]
    for rep,base in enumerate(BASE_SEEDS):
        c=dict(cfg);c['seed']=base
        c['models']=list(MODELS)
        # Saves the final best-validation checkpoint for predeclared first seed
        # for a genuinely independent *future* test, no changed training logic.
        c['archive_best_model']= rep==0
        cp=out/f'config_seed_{base}.json'
        cp.write_text(json.dumps(c,indent=2)+'\n',encoding='utf8')
        seed_plan.append({'repeat':rep,'base_seed':base,'model_seeds':dict(zip(MODELS,[base,base+1,base+2])),
                          'config_file':cp.name,'archive_best_model':rep==0})
    plan={
       'purpose':'Prospectively fixed five-repeat low-resource supervised transformer evaluation',
       'original_split_seed':42,'train_reviews':64,'validation_reviews':16,'test_reviews':200,
       'models':list(MODELS),'repeats':seed_plan,
       'original_hyperparameters':cfg,
       'selection':'val eval_loss (not test); load_best_model_at_end',
       'pipeline':'frozen original; its test F1=0.5088495575 provided as fixed reference',
       'source_sha256':{str(p.relative_to(root)):hashfile(p) for p in [source_manifest,test_file,cfg_file,trainer_file,gold_file]},
       'train_manifest_sha256':hashfile(manifest),
       'notice':'Run all predeclared seeds and report all test results, including unfavorable results; no seed/test selection or hyperparameter tuning permitted.'
    }
    (out/'seed_plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf8')
    print('Created',out/'seed_plan.json')
    for item in seed_plan:print('repeat',item['repeat'],item['model_seeds'])
    print('Confirmed disjoint 64 training, 16 validation, 200 fixed-test indices.')
    print('Run with Python environment having GPU-capable torch, datasets, transformers, accelerate, seqeval:')
    print('python 15_run_fixed_seed_training.py --source-root "'+str(root)+'" --plan "'+str(out/'seed_plan.json')+'" --require-cuda')

if __name__=='__main__':main()
