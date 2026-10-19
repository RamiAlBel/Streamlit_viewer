"""Make the web copy of the v2 dataset: one small npz per movie + params.csv, ready to upload to Hugging Face.

Per movie only what the viewer shows is kept: A, B, C (Stokes I, Q, U) divided by their common max |value| (`scale`, Jy/px)
and stored as float16, plus t_min, n0_on and the per-frame fluxes FA, FC. ~1.45 MB per movie instead of ~3 MB.
All per-movie numbers (parameters, fluxes, nf, in_10k) go to params.csv (dataset_params.csv without the long `cmd` column).

    python convert.py [--src /scratch/ralbe/ipole_abc_dataset_v2] [--out web_data] [--workers 8]
"""
import argparse, os
from multiprocessing import Pool
import numpy as np, pandas as pd

KEEP = ['t_min', 'n0_on', 'FA', 'FC']


def convert(args):
    src, dst = args
    if os.path.exists(dst):
        return
    with np.load(src) as d:
        s = float(max(np.abs(d[k]).max() for k in 'ABC'))
        out = {k: (d[k] / s).astype(np.float16) for k in 'ABC'}
        out.update({k: d[k] for k in KEEP}, scale=np.float32(s))
    tmp = dst[:-4] + '.tmp.npz'
    np.savez_compressed(tmp, **out)
    os.replace(tmp, dst)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='/scratch/ralbe/ipole_abc_dataset_v2')
    p.add_argument('--out', default='web_data')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=None, help='convert only the first N movies (testing)')
    a = p.parse_args()

    df = pd.read_csv(f'{a.src}/dataset_params.csv').drop(columns=['cmd'])
    ids = df.id.tolist()[:a.limit]
    os.makedirs(f'{a.out}/samples', exist_ok=True)
    df[df.id.isin(ids)].to_csv(f'{a.out}/params.csv', index=False)
    jobs = [(f'{a.src}/samples/{i}.npz', f'{a.out}/samples/{i}.npz') for i in ids]
    with Pool(a.workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(convert, jobs, chunksize=16), 1):
            if n % 1000 == 0 or n == len(jobs):
                print(f'{n}/{len(jobs)}', flush=True)
