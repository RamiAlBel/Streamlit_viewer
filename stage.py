"""Stage everything viewer.ipynb reads into one folder (symlinks + test-only array slices) for `hf upload-large-folder`.

    hf_stage/dataset/   samples/*.npz (originals), dataset_params.csv, manifest.csv, frame_info.csv, failures/, README.md
    hf_stage/ml/        code/, data/{movies,frames}.csv, data_obs/<tag>/movies.csv, dpa/, nb_cache/,
                        results/<run>/ (no checkpoints), results_params/ (no checkpoints), run_status.csv,
                        test_arrays/<tag>/{A,B,C}.npy + test_g.npy (global frame index of each row of A, C)
                        + test_m.npy (movie row of each row of B); same row order as results/<run>/test_pred.npy / test_B.npy
                        web/B_summary.csv: the disk-recovery table of notebook Section 1.8 (needs every test_B.npy in full)

Re-running only adds or refreshes what changed (e.g. ML runs that finished since).
"""
import glob, os, shutil, subprocess
import numpy as np, pandas as pd

DS, ML, OUT = '/scratch/ralbe/ipole_abc_dataset_v2', '/scratch/ralbe/ipole_abc_ml_v2', '/scratch/ralbe/ipole_abc_web/hf_stage'
TAGS = {'none': f'{ML}/data', 'ngeht_n0': f'{ML}/data_obs/ngeht_n0', 'eht_n0': f'{ML}/data_obs/eht_n0'}


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.islink(dst) or os.path.exists(dst):
        os.remove(dst)
    os.symlink(src, dst)


def link_tree(src, dst, skip=('.pt',)):
    for f in glob.glob(f'{src}/**/*', recursive=True):
        if os.path.isfile(f) and not f.endswith(skip) and '__pycache__' not in f:
            link(f, f'{dst}/{os.path.relpath(f, src)}')


# dataset
for f in ['dataset_params.csv', 'manifest.csv', 'frame_info.csv', 'README.md']:
    link(f'{DS}/{f}', f'{OUT}/dataset/{f}')
link_tree(f'{DS}/samples', f'{OUT}/dataset/samples')
link_tree(f'{DS}/failures', f'{OUT}/dataset/failures')

# ml: small files
link_tree(f'{ML}/code', f'{OUT}/ml/code')
for f in ['movies.csv', 'frames.csv']:
    link(f'{ML}/data/{f}', f'{OUT}/ml/data/{f}')
for tag in ('ngeht_n0', 'eht_n0'):
    link(f'{TAGS[tag]}/movies.csv', f'{OUT}/ml/data_obs/{tag}/movies.csv')
for d in ['dpa', 'nb_cache', 'results', 'results_params']:
    link_tree(f'{ML}/{d}', f'{OUT}/ml/{d}')

# ml: run status (the notebook asks squeue and looks for last.pt, neither exists online)
try:
    q = subprocess.run(['squeue', '-u', os.environ['USER'], '-h', '-o', '%j'], capture_output=True, text=True).stdout
except Exception:
    q = ''
rows = []
for d, f in [('results', 'test_frames.csv'), ('results_params', 'pred_test.csv')]:
    for r in sorted(os.listdir(f'{ML}/{d}')):
        if os.path.isdir(f'{ML}/{d}/{r}'):
            st = 'finished' if os.path.exists(f'{ML}/{d}/{r}/{f}') else 'stopped' if os.path.exists(f'{ML}/{d}/{r}/last.pt') else 'not run'
            rows.append((r, d, st))
os.makedirs(f'{OUT}/ml', exist_ok=True)
pd.DataFrame(rows, columns=['run', 'dir', 'status']).to_csv(f'{OUT}/ml/run_status.csv', index=False)
print(pd.Series([s for _, _, s in rows]).value_counts().to_dict(), '| abc2ml jobs in queue:', q.split().count('abc2ml'))

# ml: test-only slices of A, B, C for each data version (same construction as the notebook)
mov = pd.read_csv(f'{ML}/data/movies.csv')
te = mov[mov.split == 'test']
test_g = np.concatenate([o + np.arange(n) for o, n in zip(te.off, te['T'])])
test_m = te.index.to_numpy()
for tag, src in TAGS.items():
    out = f'{OUT}/ml/test_arrays/{tag}'
    os.makedirs(out, exist_ok=True)
    np.save(f'{out}/test_g.npy', test_g)
    np.save(f'{out}/test_m.npy', test_m)
    for name, idx in [('A', test_g), ('C', test_g), ('B', test_m)]:
        dst = f'{out}/{name}.npy'
        if os.path.exists(dst) and os.path.getmtime(dst) > os.path.getmtime(f'{src}/{name}.npy'):
            continue
        np.save(dst + '.tmp.npy', np.load(f'{src}/{name}.npy', mmap_mode='r')[idx])
        os.replace(dst + '.tmp.npy', dst)
        print(tag, name, np.load(dst, mmap_mode='r').shape, flush=True)

# ml: Section 1.8 table, B-hat of the method-4 / 5 runs against the true disk over all test movies (median over movies)
rows = {}
for r in sorted(os.listdir(f'{ML}/results')):
    p = r.split('_'); tag = '_'.join(p[2:]) or 'none'
    if p[0] not in ('opt4', 'opt5') or not os.path.exists(f'{ML}/results/{r}/test_B.npy'):
        continue
    bh = np.load(f'{ML}/results/{r}/test_B.npy').astype(np.float32); b = np.load(f'{OUT}/ml/test_arrays/{tag}/B.npy', mmap_mode='r')[:, 0].astype(np.float32)
    s = pd.read_csv(f'{TAGS[tag]}/movies.csv').scale.to_numpy()[test_m]; e = bh - b
    rows[r] = {'tag': tag, 'relL2 of B̂ (median)': np.median(np.linalg.norm(e, axis=(1, 2)) / np.linalg.norm(b, axis=(1, 2))),
               '|B̂−B| [µJy/px] (median)': np.median(np.abs(e).mean((1, 2)) * s * 1e6),
               'Σ(B̂−B) [mJy] (median)': np.median(e.sum((1, 2)) * s * 1e3), 'mean B [µJy/px]': np.median(b.mean((1, 2)) * s * 1e6)}
os.makedirs(f'{OUT}/ml/web', exist_ok=True)
pd.DataFrame(rows).T.rename_axis('run').to_csv(f'{OUT}/ml/web/B_summary.csv')
print('B_summary:', len(rows), 'runs')
print('staged', sum(len(fs) for _, _, fs in os.walk(OUT)), 'files')
