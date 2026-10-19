"""Streamlit viewer for the ipole_abc ADAF hot-spot movie dataset v2 and its ML results (port of viewer.ipynb; ML pages in ml.py).

Data: a private Hugging Face dataset repo (HF_REPO, read with HF_TOKEN) with dataset/ and ml/ as staged by stage.py.
For local testing set ABC_DATA_DIR to a folder with the same layout instead (e.g. hf_stage/).
"""
import io
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import streamlit as st
from huggingface_hub.errors import EntryNotFoundError
from store import exists, fetch

st.set_page_config(page_title='ipole_abc movies v2', layout='wide')

FOV = 32.0                                         # M
EXT = [-FOV / 2, FOV / 2, -FOV / 2, FOV / 2]

GROUPS = {
    'black hole / viewer': [('a', 'spin a', '.3f'), ('inc', 'inclination [deg]', '.1f'), ('r_isco', 'r_ISCO [M]', '.2f')],
    'disk': [('ne5', 'ne5 [cm^-3]', '.3g'), ('th5', 'Theta_e5', '.3g'), ('b5', 'B5 [G]', '.3g'),
             ('ne_slope', 'ne slope', '.3f'), ('b_slope', 'B slope', '.3f')],
    'hot spot': [('r_spot', 'r_spot [M]', '.2f'), ('th_spot', 'theta_spot [deg]', '.1f'), ('diam', 'diameter [M]', '.2f'),
                 ('K', 'K (x Kepler)', '.3f'), ('f_n', 'f_n', '.3g'), ('f_T', 'f_T', '.3g'), ('f_B', 'f_B', '.3g'),
                 ('inside_isco', 'inside ISCO', '')],
    'timeline': [('n_pre', 'n_pre', ''), ('n_on', 'n_on', ''), ('nf', 'nf', ''),
                 ('P_min', 'period [min]', '.1f'), ('cadence_min', 'cadence [min]', '.2f')],
    'flux (230 GHz)': [('FA_mean', '<F_A> [Jy]', '.3f'), ('FC_mean', '<F_C> [Jy]', '.3f'), ('share', 'share <F_C>/<F_A>', '.3f'),
                       ('FB', 'F_B [Jy]', '.3f'), ('R', 'R = <F_C>/F_B', '.3f')],
}
ALL = [k for rows in GROUPS.values() for k, _, _ in rows]
DEFAULT = ['a', 'inc', 'ne5', 'th5', 'b5', 'ne_slope', 'r_spot', 'th_spot', 'diam', 'K', 'f_n', 'f_T', 'f_B', 'FA_mean', 'share']
LOG = {'ne5', 'th5', 'b5', 'f_n', 'f_T', 'f_B', 'FC_mean', 'share', 'FB', 'R', 'P_min', 'cadence_min'}
DISCRETE = {'inside_isco', 'n_pre', 'n_on', 'nf'}


@st.cache_data
def load_params():
    return pd.read_csv(fetch('dataset/dataset_params.csv')).drop(columns=['cmd']).set_index('id').sort_index()


@st.cache_data(max_entries=64)
def load_movie(mid):
    with np.load(fetch(f'dataset/samples/{mid}.npz')) as d:
        return {k: d[k] for k in ('A', 'B', 'C', 't_min', 'n0_on', 'FA', 'FC')}


def params_html(p):
    cols = []
    for g, rows in GROUPS.items():
        tr = ''.join(f'<tr><td style="padding:0 8px 0 0">{lab}</td><td style="text-align:right"><b>{format(p[k], f)}</b></td></tr>'
                     for k, lab, f in rows)
        cols.append(f'<div style="margin-right:28px"><div style="font-weight:bold;border-bottom:1px solid #999">{g}</div>'
                    f'<table style="font-size:12px">{tr}</table></div>')
    return '<div style="display:flex;flex-wrap:wrap;align-items:flex-start">' + ''.join(cols) + '</div>'


@st.cache_data(max_entries=64)
def movie_png(mid, scale, stokes, log, panel=1.25):
    p, d = load_params().loc[mid], load_movie(mid)
    nf = int(p.nf)
    A, B, C = d['A'][:, stokes], d['B'][stokes], d['C'][:, stokes]
    t, on = d['t_min'], d['n0_on']
    rows = [('A', A), ('B', np.repeat(B[None], nf, 0)), ('C', C)]
    if stokes == 0:
        cmap = 'afmhot'
        vmax_all = max(A.max(), B.max(), C.max())
        def norm(img):
            v = vmax_all if scale == 'shared' else img.max()
            return LogNorm(v * 1e-3, v, clip=True) if log else Normalize(0, v)
    else:                                          # Q, U: signed, symmetric diverging scale
        cmap = 'RdBu_r'
        vmax_all = max(abs(A).max(), abs(B).max(), abs(C).max())
        def norm(img):
            v = vmax_all if scale == 'shared' else abs(img).max()
            return Normalize(-v, v)
    fig, ax = plt.subplots(3, nf, figsize=(panel * nf, panel * 3 + 0.4), squeeze=False,
                           gridspec_kw=dict(wspace=0.03, hspace=0.03))
    flux = {'A': d['FA'], 'B': np.full(nf, float(p.FB)), 'C': d['FC']}
    for r, (name, M) in enumerate(rows):
        nrm = norm(M)
        for k in range(nf):
            a = ax[r, k]
            a.imshow(M[k].T, origin='lower', extent=EXT, cmap=cmap, norm=nrm, interpolation='nearest')
            a.set_xticks([]); a.set_yticks([])
            if stokes == 0:
                a.text(0.03, 0.03, f'{flux[name][k]:.2f}', color='w', fontsize=7, transform=a.transAxes)
            if r == 0:
                a.set_title(f'{k}: {t[k]:.0f} min', fontsize=8, color='red' if on[k] else 'k')
        ax[r, 0].set_ylabel(name, fontsize=12, rotation=0, labelpad=12, va='center')
    fig.suptitle(f'movie {mid}   Stokes {"IQU"[stokes]}   ({scale} scale{", log" if log and stokes == 0 else ""})'
                 f'   in_10k = {bool(p.in_10k)}', fontsize=11, y=1.0)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(max_entries=16)
def corner_png(cols, bins, panel=1.15):
    df = load_params()
    n = len(cols)
    X = {c: np.log10(df[c].astype(float)) if c in LOG else df[c].astype(float) for c in cols}
    lab = {c: f'log10 {c}' if c in LOG else c for c in cols}
    E = {c: np.arange(X[c].min() - 0.5, X[c].max() + 1.5) if c in DISCRETE else np.histogram_bin_edges(X[c], bins)
         for c in cols}
    fig, ax = plt.subplots(n, n, figsize=(panel * n, panel * n), squeeze=False, gridspec_kw=dict(wspace=0.06, hspace=0.06))
    for i, ci in enumerate(cols):
        for j, cj in enumerate(cols):
            a = ax[i, j]
            if j > i:
                a.axis('off'); continue
            if i == j:
                a.hist(X[ci], E[ci], color='C0', alpha=0.75)
                a.set_yticks([])
            else:
                a.hist2d(X[cj], X[ci], [E[cj], E[ci]], cmap='inferno')   # black = empty bin
                a.set_ylim(E[ci][0], E[ci][-1])
            a.set_xlim(E[cj][0], E[cj][-1])
            a.tick_params(labelsize=6)
            if i < n - 1: a.set_xticklabels([])
            else: a.set_xlabel(lab[cj], fontsize=8)
            if j > 0 or i == 0: a.set_yticklabels([])
            else: a.set_ylabel(lab[ci], fontsize=8)
    fig.suptitle(f'{len(df)} movies', fontsize=12, y=0.92)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()


def viewer():
    st.markdown(
        '**A** = disk + spot (top row), **B** = disk only (middle; one image per movie, repeated under every frame), '
        '**C** = spot only (bottom). Frame titles: time in minutes; a **red** title marks the n=0 window (the spot\'s direct '
        'image, one orbit). Per-frame fluxes (Jy) are printed in each panel. Images: 128 x 128 px over 32 M (1 M = 5.19 uas).')
    df = load_params()
    ids = df.index.tolist()
    if 'mid' not in st.session_state:
        st.session_state.mid = st.session_state.sl = st.session_state.box = ids[0]

    def _sl():
        st.session_state.mid = st.session_state.box = st.session_state.sl
    def _box():
        if st.session_state.box in df.index:
            st.session_state.mid = st.session_state.sl = st.session_state.box

    c1, c2 = st.columns([5, 1])
    c1.select_slider('movie id', options=ids, key='sl', on_change=_sl)
    c2.number_input('go to id', min_value=0, max_value=int(max(ids)), step=1, key='box', on_change=_box)
    if st.session_state.box not in df.index:
        c2.caption(f'id {st.session_state.box} is not in the dataset')
        if exists(f'dataset/failures/{st.session_state.box}.txt'):
            st.warning(f'id {st.session_state.box} failed: ' + open(fetch(f'dataset/failures/{st.session_state.box}.txt')).read().strip())
    c1, c2, c3, _ = st.columns([1, 1, 1, 3])
    scale = c1.radio('scale', ['shared', 'per row'], horizontal=True)
    stokes = 'IQU'.index(c2.radio('Stokes', ['I', 'Q', 'U'], horizontal=True))
    log = c3.checkbox('log (3 decades)', disabled=stokes != 0)

    mid = st.session_state.mid
    st.html(params_html(df.loc[mid]))
    with st.spinner(f'loading movie {mid}'):
        try:
            png = movie_png(mid, scale, stokes, log)
        except (FileNotFoundError, EntryNotFoundError):
            st.info(f'movie {mid} is not online yet (the upload may still be running)'); return
    st.image(png, width='stretch')


def distributions():
    st.markdown('Diagonal: histogram of each parameter; below it: 2-D histograms of every pair (inferno: black = no movies, '
                'yellow = most). Log-uniform parameters and the flux ratios are shown as log10.')
    with st.form('corner'):
        cols = st.multiselect('parameters', ALL, default=DEFAULT)
        bins = st.slider('bins', 10, 60, 30, step=5)
        go = st.form_submit_button('plot', type='primary')
    if go or 'corner_cols' in st.session_state:
        if go:
            st.session_state.corner_cols, st.session_state.corner_bins = cols, bins
        cols = [c for c in ALL if c in st.session_state.corner_cols]
        if len(cols) < 2:
            st.warning('select at least 2 parameters'); return
        with st.spinner('plotting'):
            st.image(corner_png(tuple(cols), st.session_state.corner_bins))


st.title('ipole_abc ADAF hot-spot movies (v2)')
st.caption('Sgr A*, 230 GHz, Kerr; each movie rendered as A = disk + hot spot, B = disk only, C = hot spot only.')
ML_PAGES = ['ML · overview and runs', 'ML 1 · task 1 scores', 'ML 1 · task 1 movies and angles', 'ML 2 · task 2 parameters',
            'ML 3 · blurred movies']
page = st.sidebar.radio('page', ['Movie viewer', 'Parameter distributions'] + ML_PAGES)
if page == 'Movie viewer':
    viewer()
elif page == 'Parameter distributions':
    distributions()
else:
    with st.spinner('loading the ML results (first visit only)'):
        import ml
    ml.PAGES[page]()
