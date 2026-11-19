"""Create a SHA-256 manifest and local environment record (does not modify experiment logic).

Run after outputs are generated. A hash manifest is NOT an immutable public archive;
publish a tagged commit/Zenodo archive separately and cite its commit/DOI.

 py 03_capture_evidence_manifest.py --repo Low-Resource-Aspect-Term-Extraction-main --results R2_test_run --outdir R2_checks
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo',required=True,type=Path)
    ap.add_argument('--results',required=True,type=Path)
    ap.add_argument('--outdir',default=Path('R2_checks'),type=Path)
    a=ap.parse_args()
    repo=a.repo.resolve(); results=a.results.resolve(); out=a.outdir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    versions={}
    for pkg in ['torch','transformers','datasets','seqeval','scikit-learn','numpy','pandas','nltk','accelerate']:
        try: versions[pkg]=metadata.version(pkg)
        except metadata.PackageNotFoundError: versions[pkg]='NOT INSTALLED'
    release={'python':sys.version,'platform':platform.platform(),'libraries':versions,
            'repo_source_files':{},'result_files':{},
            'warning':'Files/hashes are reproducibility evidence, not an immutable release.'}
    for root,key in [(repo,'repo_source_files'),(results,'result_files')]:
        if root.exists():
            for path in sorted(root.rglob('*')):
                if path.is_file() and not any(p in {'__pycache__','.git'} for p in path.parts):
                    release[key][path.relative_to(root).as_posix()]=sha256(path)
    try:
        r=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,check=True)
        release['git_commit']=r.stdout.strip()
    except (OSError,subprocess.CalledProcessError):
        release['git_commit']='NOT AVAILABLE in downloaded ZIP; archive/tag an actual Git repository commit.'
    (out/'evidence_manifest.json').write_text(json.dumps(release,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Wrote:',out/'evidence_manifest.json')
    print('Public immutable archive and full prediction files still required for Reviewer 2.6.')

if __name__=='__main__':main()
