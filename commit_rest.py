"""Commit every staged file that is not in the HF repo yet, in a few big commits (upload-large-folder makes one commit per
~20 files and hits the 128 commits/hour limit). The bytes are mostly pre-uploaded already, so this is quick."""
import os, sys, time
from huggingface_hub import HfApi, CommitOperationAdd

REPO, STAGE, CHUNK = 'Rami1997/ipole-abc-v2', '/scratch/ralbe/ipole_abc_web/hf_stage', 1000
api = HfApi()
online = set(api.list_repo_files(REPO, repo_type='dataset'))
todo = sorted(p for p in (os.path.relpath(os.path.join(r, f), STAGE) for r, _, fs in os.walk(STAGE) if '.cache' not in r for f in fs)
              if p not in online)
print(len(todo), 'files to commit', flush=True)
for a in range(0, len(todo), CHUNK):
    part = todo[a:a + CHUNK]
    ops = [CommitOperationAdd(p, os.path.realpath(f'{STAGE}/{p}')) for p in part]
    for attempt in range(6):
        try:
            api.create_commit(REPO, ops, commit_message=f'Add remaining files ({a + len(part)}/{len(todo)})', repo_type='dataset')
            print('committed', a + len(part), '/', len(todo), flush=True); break
        except Exception as e:
            print('retry in 15 min:', str(e).splitlines()[0], flush=True); time.sleep(900)
    else:
        sys.exit('gave up')
print('done; missing now:', len(set(todo) - set(api.list_repo_files(REPO, repo_type='dataset'))))
