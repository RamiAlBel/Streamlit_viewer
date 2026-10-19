"""Data access for the app: the private Hugging Face dataset repo (HF_REPO, read with HF_TOKEN), or a local folder with the same
layout (ABC_DATA_DIR, e.g. hf_stage/) for testing. Big .npy arrays are never downloaded whole: npy_rows reads a row range with
HTTP range requests."""
import logging, os
import numpy as np, pandas as pd
import streamlit as st
from numpy.lib import format as npf

LOCAL = os.environ.get('ABC_DATA_DIR')
REPO = os.environ.get('HF_REPO', '')
TOKEN = os.environ.get('HF_TOKEN')
# the caches are also used from worker threads (parallel reads); they work there, only the warning is noise
logging.getLogger('streamlit.runtime.scriptrunner_utils.script_run_context').setLevel(logging.ERROR)


def fetch(path):
    """local path of a repo file (downloaded and cached on first use)"""
    if LOCAL:
        p = os.path.join(LOCAL, path)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, path, repo_type='dataset', token=TOKEN)


@st.cache_resource(ttl=3600)
def _repo_files():
    from huggingface_hub import HfApi
    return frozenset(HfApi().list_repo_files(REPO, repo_type='dataset', token=TOKEN))


def exists(path):
    return os.path.exists(os.path.join(LOCAL, path)) if LOCAL else path in _repo_files()


def read_csv(path, **kw):
    return pd.read_csv(fetch(path), **kw)


@st.cache_resource
def _fs():
    from huggingface_hub import HfFileSystem
    return HfFileSystem(token=TOKEN)


@st.cache_resource
def _header(path):
    """shape, dtype and data offset of a .npy file in the repo"""
    with _fs().open(f'datasets/{REPO}/{path}', 'rb', block_size=2 ** 16) as f:
        v = npf.read_magic(f)
        shape, fortran, dt = npf._read_array_header(f, v)
        return shape, dt, f.tell()


def _read(path, a, b):
    if LOCAL:
        return np.array(np.load(os.path.join(LOCAL, path), mmap_mode='r')[a:b])
    shape, dt, off = _header(path)
    row = int(np.prod(shape[1:], dtype=np.int64)) * dt.itemsize
    buf = _fs().read_block(f'datasets/{REPO}/{path}', off + a * row, (b - a) * row)
    return np.frombuffer(buf, dt).reshape(b - a, *shape[1:])


@st.cache_data(max_entries=256, show_spinner=False)
def npy_rows(path, a, b):
    """rows a:b (first axis) of a .npy file"""
    return _read(path, a, b)


@st.cache_data(max_entries=64, show_spinner=False)
def npy_rows_multi(paths, a, b):
    """rows a:b of several .npy files with the same first axis, read in parallel"""
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(8) as ex:
        return list(ex.map(lambda p: _read(p, a, b), paths))
