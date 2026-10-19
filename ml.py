"""ML results pages of the app: port of viewer.ipynb Sections 0-3 (cells 5-63) on the files of the HF repo (store.py).

Imported on the first visit of an ML page; the module-level code below is the notebook's setup cell (loads the test results of every
finished run). Figures are drawn by the fig_* functions (the notebook's plotting code) and cached as PNGs.
"""
import io, json, os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import streamlit as st
import dpa_lib as dpa
from store import exists, fetch, read_csv, npy_rows, npy_rows_multi

MD = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'nb_text.json'), encoding='utf-8'))
FOV = 32.0
EXT = [-FOV / 2, FOV / 2, -FOV / 2, FOV / 2]


def md(i, *subs):
    s = MD[str(i)]
    for a, b in subs:
        s = s.replace(a, b)
    st.markdown(s)


def _png(fig, dpi=100):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(max_entries=400, ttl=6 * 3600, show_spinner=False)
def _fig_png(name, args):
    return _png(globals()[name](*args))


def plot(fn, *args):
    """draw fig_<name>(*args) (cached by its arguments)"""
    with st.spinner('plotting'):
        st.image(_fig_png(fn.__name__, args))


def table(sty):
    st.dataframe(sty)


# ======================================================================================= cell 7: setup
TAGS = ['none', 'ngeht_n0', 'eht_n0']                       # data versions: unblurred, ngEHT beam, EHT beam (no noise)
TAG_LAB = {'none': 'unblurred', 'ngeht_n0': 'ngEHT (8/3 M)', 'eht_n0': 'EHT (4 M)'}
ddir = lambda tag: 'ml/data' if tag == 'none' else f'ml/data_obs/{tag}'
MOV = {tag: read_csv(f'{ddir(tag)}/movies.csv') for tag in TAGS}     # same movies, split and frames; only 'scale' differs
mov = MOV['none']; fra = read_csv('ml/data/frames.csv')
mov['cos_inc'] = np.cos(np.radians(mov.inc))
row_of_id = {int(i): m for m, i in enumerate(mov.id)}
te = mov[mov.split == 'test']
test_g = np.concatenate([o + np.arange(n) for o, n in zip(te.off, te['T'])])
row_of = {int(g): i for i, g in enumerate(test_g)}            # global frame index -> row of test_pred.npy and test_arrays/<tag>/{A,C}.npy
test_m = te.index.to_numpy()                                  # movie rows of the test movies (row order of test_B.npy and B.npy)

METHOD = {1: '1 single frame', 2: '2 frame + neighbours', 3: '3 whole movie (3D)', 4: '4 disk subtraction', 5: '5 disk subtr. + neighbours'}
MCOL = {1: 'tab:blue', 2: 'tab:green', 3: 'tab:orange', 4: 'tab:purple', 5: 'tab:red'}
TCOL = {'none': 'k', 'ngeht_n0': 'tab:blue', 'eht_n0': 'tab:red'}
run1 = lambda o, s, tag='none': f'opt{o}_{s}' + ('' if tag == 'none' else f'_{tag}')
def parse1(r):                                                # 'opt2_I_eht_n0' -> (2, 'I', 'eht_n0')
    p = r.split('_'); return int(p[0][3]), p[1], '_'.join(p[2:]) or 'none'
ls_of = lambda r: '-' if r.split('_')[1] == 'I' else '--'     # solid = input I, dashed = input IQU
T1 = [run1(o, s, tag) for tag in TAGS for o in (1, 2, 3, 4, 5) for s in ('I', 'IQU')]

# ---- run status: test results online = finished; otherwise the state on the cluster when the data were staged (run_status.csv)
RUN_STATUS = read_csv('ml/run_status.csv').set_index('run').status.to_dict()
def status(run):
    d, f = ('ml/results', 'test_frames.csv') if run.startswith('opt') else ('ml/results_params', 'pred_test.csv')
    if exists(f'{d}/{run}/{f}'): return 'finished'
    s = RUN_STATUS.get(run, 'not run')
    return 'not online yet' if s == 'finished' else s
STATUS_COLOR = {'finished': '#c7e9c0', 'running': '#fdd0a2', 'queued': '#fff3b0', 'stopped': '#e0e0e0', 'not run': '#ffffff',
                'not online yet': '#fdd0a2'}
def status_grid(runs, row_of_, col_of_, rows, cols):
    g = pd.DataFrame('', index=rows, columns=cols)
    for r in runs: g.loc[row_of_(r), col_of_(r)] = status(r)
    return g.style.map(lambda s: f'background-color: {STATUS_COLOR.get(s, "#ffffff")}; color: black')

# ---- task-1 test results, one row per test frame (+ pixel errors in physical units, from nb_cache/)
def load_t1(r):
    t = read_csv(f'ml/results/{r}/test_frames.csv')
    t['flux_err'] = np.abs(np.log(t.flux_pred.clip(1e-12) / t.flux_true.clip(1e-12)))
    t['pos'] = t.k - mov.spot_first.to_numpy()[t.movie]                  # frame number inside the window
    px = read_csv(f'ml/nb_cache/pix_{r}.csv'); assert (px.g.to_numpy() == t.g.to_numpy()).all()
    t['mae'], t['zero'], t['dF'] = px.mae.to_numpy() * 1e6, px.zero.to_numpy() * 1e6, px.dF.to_numpy() * 1e3   # µJy/px, µJy/px, mJy
    return t
done1 = [r for r in T1 if status(r) == 'finished' and exists(f'ml/nb_cache/pix_{r}.csv')]
with ThreadPoolExecutor(8) as _ex:
    T = dict(zip(done1, _ex.map(load_t1, done1)))


# ======================================================================================= cell 12: scores
def scores1(t):
    n1 = t[~t.has_n0]
    return {'relL2 median': t.relL2.median(), 'relL2 p90': t.relL2.quantile(.9), 'relL2 with n=0': t[t.has_n0].relL2.median(),
            'relL2 only n≥1': n1.relL2.median(), 'flux err median': t.flux_err.median(), 'flux err only n≥1': n1.flux_err.median(),
            'frames relL2<0.05': (t.relL2 < 0.05).mean(), 'MAE [µJy/px]': t.mae.mean(), 'MAE / empty': t.mae.sum() / t.zero.sum(),
            '|ΔF| median [mJy]': t.dF.abs().median()}
FMT1 = {'frames relL2<0.05': '{:.0%}', 'MAE / empty': '{:.1%}', 'MAE [µJy/px]': '{:.3f}', '|ΔF| median [mJy]': '{:.2f}'}
def style_scores(tab, sc):
    bad = [c for c in sc if c != 'frames relL2<0.05']
    return tab.style.format({c: FMT1.get(c, '{:.3f}') for c in sc}, na_rep='') \
              .background_gradient(cmap='RdYlGn_r', subset=bad).background_gradient(cmap='RdYlGn', subset=['frames relL2<0.05'])
R0 = [r for r in T if parse1(r)[2] == 'none']                         # finished unblurred runs
tab1 = pd.DataFrame({r: {'method': METHOD[parse1(r)[0]], 'input': parse1(r)[1], **scores1(T[r])} for r in R0}).T
SC1 = list(tab1.columns[2:]); tab1[SC1] = tab1[SC1].astype(float)
t0 = T[R0[0]]


# ======================================================================================= cell 14: bars
def fig_bars():
    scores = ['relL2 median', 'relL2 p90', 'relL2 with n=0', 'relL2 only n≥1', 'flux err median', 'MAE [µJy/px]']
    fig, ax = plt.subplots(1, len(scores), figsize=(3.9 * len(scores), 3.7), squeeze=False); ax = ax[0]
    for a_, sc in zip(ax, scores):
        for o in (1, 2, 3, 4, 5):
            for j, s in enumerate(('I', 'IQU')):
                r = run1(o, s)
                if r in tab1.index:
                    a_.bar(o + (j - .5) * .38, tab1.loc[r, sc], .36, color=MCOL[o], hatch='//' if s == 'IQU' else None,
                           edgecolor='k', lw=.5)
        a_.set_xticks([1, 2, 3, 4, 5]); a_.set_xticklabels(['single\nframe', 'frame +\nneigh.', 'whole\nmovie 3D', 'disk\nsubtract', 'disk sub.\n+ neigh.'], fontsize=8)
        a_.set_title(sc, fontsize=10)
    ax[0].legend([plt.Rectangle((0, 0), 1, 1, fc='0.8', ec='k', lw=.5, hatch=h) for h in (None, '//')], ['input I', 'input IQU'], fontsize=8)
    plt.tight_layout(); return fig


# ======================================================================================= cell 16: bootstrap in µJy/px
def boot_means(runs, col='mae', B=2000, seed=0):
    # bootstrap of the frame-averaged `col` over test movies: (B, len(runs)) array, plus the point estimates
    S = pd.DataFrame({r: T[r].groupby('movie')[col].sum() for r in runs}); N = T[runs[0]].groupby('movie').size()
    Wb = np.random.default_rng(seed).multinomial(len(N), np.full(len(N), 1 / len(N)), size=B).astype(float)
    return (Wb @ S.to_numpy()) / (Wb @ N.to_numpy())[:, None], S.sum().to_numpy() / N.sum()

@st.cache_data(show_spinner=False)
def delta_table(pairs):
    rows = {}
    for lab, ra, rb in pairs:                                 # rb - ra
        bt, pt = boot_means([ra, rb]); d = bt[:, 1] - bt[:, 0]; lo, hi = np.quantile(d, [.025, .975])
        rows[lab] = {'MAE ref [µJy/px]': pt[0], 'MAE [µJy/px]': pt[1], 'Δ [µJy/px]': pt[1] - pt[0], 'Δ 95% lo': lo, 'Δ 95% hi': hi,
                     'Δ [%]': 100 * (pt[1] / pt[0] - 1), 'Δ summed per frame [mJy]': (pt[1] - pt[0]) * 128 ** 2 * 1e-3,
                     'significant': '*' if lo > 0 or hi < 0 else ''}
    return pd.DataFrame(rows).T

def fig_delta_matrix(runs, title):
    runs = list(runs)
    bt, pt = boot_means(runs); n = len(runs); M = np.full((n, n), np.nan); sig = np.zeros((n, n), bool)
    for i in range(n):
        for j in range(n):
            if i != j:
                d = bt[:, i] - bt[:, j]; M[i, j] = pt[i] - pt[j]; lo, hi = np.quantile(d, [.025, .975]); sig[i, j] = lo > 0 or hi < 0
    v = np.nanmax(np.abs(M)) if n > 1 else 1
    fig, a_ = plt.subplots(figsize=(1.15 * n + 2.5, 0.9 * n + 1.2))
    im = a_.imshow(M, cmap='RdYlGn_r', norm=mcolors.TwoSlopeNorm(0, -v, v))
    for i in range(n):
        for j in range(n):
            if i != j: a_.text(j, i, f'{M[i, j]:+.3f}' + ('*' if sig[i, j] else ''), ha='center', va='center', fontsize=8)
    a_.set_xticks(range(n)); a_.set_xticklabels(runs, rotation=45, ha='right'); a_.set_yticks(range(n)); a_.set_yticklabels(runs)
    a_.set_title(title, fontsize=10); plt.colorbar(im, label='µJy / px'); plt.tight_layout(); return fig


# ======================================================================================= cell 18: head-to-head
def fig_head_to_head(runs):
    n = len(runs); M = np.full((n, n), np.nan)
    for i, r1 in enumerate(runs):
        for j, r2 in enumerate(runs):
            if i != j: M[i, j] = (T[r1].relL2.to_numpy() < T[r2].relL2.to_numpy()).mean()
    fig, a_ = plt.subplots(figsize=(1.1 * n + 2, 0.9 * n + 1))
    im = a_.imshow(M, cmap='RdYlGn', vmin=0.2, vmax=0.8)
    for i in range(n):
        for j in range(n):
            if i != j: a_.text(j, i, f'{M[i, j]:.0%}', ha='center', va='center', fontsize=9)
    a_.set_xticks(range(n)); a_.set_xticklabels(runs, rotation=45, ha='right'); a_.set_yticks(range(n)); a_.set_yticklabels(runs)
    a_.set_title('share of test frames where row beats column (relL2)'); plt.colorbar(im); plt.tight_layout(); return fig


# ======================================================================================= cell 20: errors against parameters
# x-axis parameters: name -> (label, plotted as log10)
XPAR = {'logR': ('log R = log10 <F_C>/F_B', False), 'share': ('spot share <F_C>/<F_A>', False), 'FA_mean': ('<F_A> [Jy]', False),
        'a': ('spin a', False), 'inc': ('inclination [deg]', False), 'r_spot': ('r_spot [M]', False), 'th_spot': ('θ_spot [deg]', False),
        'diam': ('spot diameter [M]', False), 'K': ('K (× Kepler)', False), 'f_n': ('f_n', True), 'f_T': ('f_T', True), 'f_B': ('f_B', True),
        'ne5': ('n_e,5 [cm⁻³]', True), 'th5': ('Θ_e,5', True), 'b5': ('B_5 [G]', True), 'ne_slope': ('p_n (density slope)', False),
        'P_min': ('period P [min]', True), 'cadence_min': ('cadence [min]', True), 'T': ('frames in the window T', False),
        'inside_isco': ('spot inside the ISCO (0/1)', False)}
xlab = lambda p: ('log10 ' if XPAR[p][1] else '') + XPAR[p][0]
def xval(p, movie_rows):
    v = mov[p].to_numpy()[np.asarray(movie_rows)].astype(float)
    return np.log10(v) if XPAR[p][1] else v
def binned(x, y, nb=8):
    # median of y in bins of x with equal counts (or per value, for discrete x); x position = median x of the bin
    x, y = np.asarray(x, float), np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) == 0: return np.array([]), np.array([])
    if len(np.unique(x)) <= 12:
        g = pd.Series(y).groupby(x).median(); return g.index.to_numpy(), g.to_numpy()
    e = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1))); b = np.clip(np.digitize(x, e[1:-1]), 0, len(e) - 2)
    g = pd.DataFrame({'b': b, 'x': x, 'y': y}).groupby('b'); return g.x.median().to_numpy(), g.y.median().to_numpy()

MET1 = {'relL2, all frames': ('relL2', None), 'relL2, frames with n=0': ('relL2', True), 'relL2, only-n≥1 frames': ('relL2', False),
        'flux error |ln F̂/F|': ('flux_err', None), 'error per pixel [µJy/px]': ('mae', None)}
def sel(t, has): return t if has is None else t[t.has_n0 == has]
def style1(r): o, s, tag = parse1(r); return dict(c=MCOL[o], ls=ls_of(r), marker='o', ms=3, label=r)

def fig_metric_vs(p, metrics, runs):
    fig, ax = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 3.9), squeeze=False); ax = ax[0]
    for a_, mt in zip(ax, metrics):
        col, has = MET1[mt]
        for r in runs:
            t = sel(T[r], has); a_.plot(*binned(xval(p, t.movie), t[col]), **style1(r))
        a_.set_yscale('log'); a_.set_xlabel(xlab(p)); a_.set_title(f'median {mt}', fontsize=10); a_.grid(alpha=.3, which='both')
    ax[0].legend(fontsize=7, ncol=2); plt.tight_layout(); return fig


# ======================================================================================= cell 22: every parameter
def fig_param_grid(runs, metric, params, title='', ncol=5):
    params = list(params); col, has = MET1[metric]; nr = -(-len(params) // ncol)
    fig, ax = plt.subplots(nr, ncol, figsize=(3.7 * ncol, 2.9 * nr + 0.6), squeeze=False)
    for a_, p in zip(ax.flat, params):
        for r in runs:
            t = sel(T[r], has); a_.plot(*binned(xval(p, t.movie), t[col]), **style1(r))
        a_.set_yscale('log'); a_.set_xlabel(xlab(p), fontsize=8); a_.tick_params(labelsize=7); a_.grid(alpha=.3, which='both')
    for a_ in ax.flat[len(params):]: a_.axis('off')
    for a_ in ax[:, 0]: a_.set_ylabel(f'median {metric}', fontsize=8)
    h, l = ax.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=min(len(l), 8), fontsize=8, frameon=False, title=title, title_fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (2.9 * nr + 0.6))); return fig


# ======================================================================================= cell 24: along the movie
def fig_along():
    fig, a_ = plt.subplots(figsize=(10, 4.2)); a2 = a_.twinx()
    share_n1 = t0.groupby('pos').has_n0.apply(lambda x: (~x).mean())
    a2.bar(share_n1.index, share_n1.values, color='0.85', zorder=0); a2.set_ylabel('share of frames with only n≥1 light'); a2.set_ylim(0, 1.05)
    a_.set_zorder(1); a_.patch.set_visible(False)
    for r in R0:
        g = T[r].groupby('pos').relL2.median(); a_.plot(g.index, g.values, **style1(r))
    a_.set_yscale('log'); a_.set_xlabel('frame position in the window'); a_.set_ylabel('median relL2'); a_.legend(fontsize=8, ncol=2)
    return fig


# ======================================================================================= cell 26: IQU against I
def iqu_pairs(tag):
    return [o for o in (1, 2, 3, 4, 5) if run1(o, 'I', tag) in T and run1(o, 'IQU', tag) in T]

def fig_iqu_vs_i(tag='none'):
    pairs = iqu_pairs(tag)
    fig, ax = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 4.4), squeeze=False); ax = ax[0]
    for a_, o in zip(ax, pairs):
        x, y = T[run1(o, 'I', tag)].relL2.to_numpy(), T[run1(o, 'IQU', tag)].relL2.to_numpy()
        a_.hist2d(np.log10(x), np.log10(y), 60, range=[[-3, 0.5], [-3, 0.5]], cmap='magma_r', cmin=1)
        a_.plot([-3, .5], [-3, .5], 'c-')
        a_.set_title(f'{METHOD[o]}\nIQU better on {(y < x).mean():.0%} of frames, median ratio {np.median(y / x):.2f}', fontsize=10)
        a_.set_xlabel('log10 relL2, input I'); a_.set_ylabel('log10 relL2, input IQU')
    fig.suptitle(TAG_LAB[tag], fontsize=11); plt.tight_layout(); return fig


# ======================================================================================= cell 28: the test movies
def soft(x, vmax, stretch='asinh'):
    if stretch == 'linear': return np.clip(x / vmax, 0, 1)
    return np.arcsinh(np.clip(x, 0, None) / (0.01 * vmax)) / np.arcsinh(100)    # 1 % of vmax -> ~0.18 of the colour range

MOVIE_IDS = [int(r.id) for r in te.sort_values('share').itertuples()]
MOVIE_LAB = {int(r.id): f'{int(r.id)}   share {r.share:.2f}, log R {r.logR:+.2f}, inc {r.inc:.0f}°, a {r.a:+.2f}' for r in te.itertuples()}
MID0 = int(te.id[(te.share - te.share.median()).abs().idxmin()])          # default: the test movie with the median spot share

def frames_of(movie_id):
    """movie row, number of window frames, global index of its first frame, row of that frame in the test arrays"""
    m = row_of_id[movie_id]; Tm, off = int(mov['T'][m]), int(mov.off[m])
    return m, Tm, off, row_of[off]

def picked(tag, c):
    # finished runs among the ticked methods / inputs, and the ticked ones that are missing (with their status)
    want = [run1(o, s, tag) for o in (1, 2, 3, 4, 5) if c.get(f'opt{o}') for s in ('I', 'IQU') if c.get(s)]
    miss = [r for r in want if r not in T]
    if miss: st.caption('not available: ' + ', '.join(f'{r} ({status(r)})' for r in miss))
    return tuple(r for r in want if r in T)

def fig_movie(movie_id, tag, runs, n_frames=7, stretch='asinh'):
    m, Tm, off, i0 = frames_of(movie_id)
    js = np.unique(np.linspace(0, Tm - 1, n_frames).round().astype(int))
    A_, C_, *P = npy_rows_multi((f'ml/test_arrays/{tag}/A.npy', f'ml/test_arrays/{tag}/C.npy')
                                + tuple(f'ml/results/{r}/test_pred.npy' for r in runs), i0, i0 + Tm)
    vC = float(np.max(C_))
    nr = 2 + len(runs); fig, ax = plt.subplots(nr, len(js), figsize=(1.9 * len(js), 1.95 * nr), squeeze=False)
    for c, j in enumerate(js):
        g = off + j; ct = C_[j].astype(float); f = fra.iloc[g]
        ax[0, c].imshow(soft(A_[j, 0].astype(float), float(A_[j, 0].max()), stretch).T, origin='lower', cmap='afmhot', vmin=0, vmax=1)
        ax[0, c].set_title(f'frame {int(f.k)}, {f.t_min:.0f} min  ' + ('n=0' if f.has_n0 else 'n≥1'), fontsize=7)
        ax[1, c].imshow(soft(ct, vC, stretch).T, origin='lower', cmap='afmhot', vmin=0, vmax=1)
        for k, r in enumerate(runs):
            p = P[k][j].astype(float)
            ax[2 + k, c].imshow(soft(p, vC, stretch).T, origin='lower', cmap='afmhot', vmin=0, vmax=1)
            ax[2 + k, c].text(2, 124, f'relL2 {np.linalg.norm(p - ct) / np.linalg.norm(ct):.3f}', color='w', fontsize=7, va='top')
    for k, lab in enumerate(['A (input)', 'true C'] + [f'pred. {r}' for r in runs]): ax[k, 0].set_ylabel(lab, fontsize=8)
    for a_ in ax.flat: a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f'{TAG_LAB[tag]} — test movie {movie_id}: share {mov.share[m]:.2f}, log R {mov.logR[m]:.2f}, <F_A> {mov.FA_mean[m]:.2f} Jy, '
                 f'inc {mov.inc[m]:.0f}°, a {mov.a[m]:.2f}, θ_spot {mov.th_spot[m]:.0f}°, r_spot {mov.r_spot[m]:.1f} M, {Tm} frames, '
                 f'cadence {mov.cadence_min[m]:.1f} min', fontsize=9)
    plt.tight_layout(); return fig

def controls(key, tag='none'):
    """the notebook's shared controls of Sections 1.6-1.9 (key 'p1') and 3.5-3.6 (key 'p3')"""
    c = {'movie': st.select_slider('test movie (ordered by spot share)', options=MOVIE_IDS, value=MID0, format_func=MOVIE_LAB.get,
                                   key=f'{key}_movie')}
    a, b, d = st.columns(3)
    c['tag'] = a.selectbox('data', TAGS, index=TAGS.index(tag), format_func=TAG_LAB.get, key=f'{key}_tag')
    c['n_frames'] = b.slider('frames', 3, 19, 7, key=f'{key}_nf')
    c['stretch'] = d.selectbox('scale', ['asinh', 'linear'], key=f'{key}_stretch')
    cols = st.columns([1.2, 1.4, 1.4, 1.3, 1.6, 0.9, 1])
    for o in (1, 2, 3, 4, 5):
        c[f'opt{o}'] = cols[o - 1].checkbox(METHOD[o], True, key=f'{key}_opt{o}')
    c['I'] = cols[5].checkbox('input I', True, key=f'{key}_I')
    c['IQU'] = cols[6].checkbox('input IQU', False, key=f'{key}_IQU')
    return c


# ======================================================================================= cell 30: light curves
def fig_lc(movie, tag, runs):
    m = row_of_id[movie]; f = fra[fra.movie == m]; s = MOV[tag].scale[m]
    nz = lambda y: y.where(y > 0)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    ax[0].plot(f.t_min, nz(f.FC), 'k-', lw=2.5, label='truth (n=0 + n≥1)'); ax[0].plot(f.t_min, nz(f.FC_n0), 'k:', label='truth, n=0')
    ax[0].plot(f.t_min, nz(f.FC_n1), 'k-.', label='truth, n≥1')
    for r in runs:
        tm = T[r][T[r].movie == m]; st_ = style1(r)
        ax[0].plot(tm.t_min, nz(tm.flux_pred * s), **st_); ax[1].plot(tm.t_min, tm.dF, **st_)
    ax[0].set_yscale('log'); ax[0].set_ylabel('spot flux [Jy]'); ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3, which='both')
    ax[1].axhline(0, c='k', lw=.8); ax[1].set_ylabel('ΔF = F̂ − F [mJy]'); ax[1].set_xlabel('time [min]'); ax[1].grid(alpha=.3)
    ax[0].set_title(f'{TAG_LAB[tag]} — test movie {movie}: share {mov.share[m]:.2f}, log R {mov.logR[m]:.2f}', fontsize=10)
    plt.tight_layout(); return fig


# ======================================================================================= cell 32: the recovered disk
def runs_B(tag, runs=None):
    return tuple(r for r in (runs if runs is not None else T) if parse1(r)[0] in (4, 5) and parse1(r)[2] == tag
                 and exists(f'ml/results/{r}/test_B.npy'))

def fig_disk(movie, tag, rr):
    m = row_of_id[movie]; s = MOV[tag].scale[m]; k = int(np.searchsorted(test_m, m))
    b = npy_rows(f'ml/test_arrays/{tag}/B.npy', k, k + 1)[0, 0].astype(float); vb = b.max()
    BH = npy_rows_multi(tuple(f'ml/results/{r}/test_B.npy' for r in rr), k, k + 1)
    fig, ax = plt.subplots(len(rr), 3, figsize=(9.6, 3.2 * len(rr)), squeeze=False)
    for i, r in enumerate(rr):
        bh = BH[i][0].astype(float); res = bh - b
        ax[i, 0].imshow(soft(b, vb).T, origin='lower', cmap='afmhot', vmin=0, vmax=1, extent=EXT); ax[i, 0].set_title('true disk B', fontsize=9)
        ax[i, 1].imshow(soft(bh, vb).T, origin='lower', cmap='afmhot', vmin=0, vmax=1, extent=EXT)
        ax[i, 1].set_title(f'{r}: B̂, relL2 {np.linalg.norm(res) / np.linalg.norm(b):.3f}', fontsize=9)
        ax[i, 2].imshow((np.sign(res) * np.arcsinh(np.abs(res) / (0.01 * vb)) / np.arcsinh(100)).T, origin='lower', cmap='RdBu_r', vmin=-1, vmax=1, extent=EXT)
        ax[i, 2].set_title(f'B̂ − B: {np.abs(res).mean() * s * 1e6:.2f} µJy/px, Σ = {res.sum() * s * 1e3:+.1f} mJy', fontsize=9)
        ax[i, 0].set_ylabel(r, fontsize=8)
    for a_ in ax.flat: a_.set_xticks([]); a_.set_yticks([])
    fig.suptitle(f'{TAG_LAB[tag]} — test movie {movie}: F_B = {mov.FB[m]:.2f} Jy, share {mov.share[m]:.2f}', fontsize=10)
    plt.tight_layout(); return fig


# ======================================================================================= cells 34-40: position angles
@lru_cache(None)
def pa_data():
    PA = read_csv('ml/dpa/pa_test.csv')
    need = [r for r in done1 if parse1(r)[2] == 'none']
    RUNS_PA = [r for r in need if r in set(PA.src)]
    BEST = min(RUNS_PA, key=lambda r: tab1.loc[r, 'relL2 median'])
    return PA, RUNS_PA, BEST

def cn0(i):
    """exact direct (n=0) image of the window frames, movie-normalised like C (from the dataset's Cn0)"""
    r = mov.iloc[row_of_id[i]]
    with np.load(fetch(f'dataset/samples/{i}.npz')) as z:
        return z['Cn0'][int(r.spot_first):int(r.spot_last) + 1].astype(float) / r.scale

def draw_seg(a_, img, m0, m1, vmax, cx, cy, half=12):
    a_.imshow(soft(img, vmax).T, origin='lower', cmap='afmhot', vmin=0, vmax=1, extent=EXT)
    a_.plot(cx, cy, c='c', lw=0.4, alpha=0.6)
    for msk, col, mk in ((m0, 'cyan', 'x'), (m1, 'lime', '+')):
        if msk.sum() > 0:
            a_.contour(dpa.XC, dpa.XC, (msk > 0).T.astype(float), [0.5], colors=col, linewidths=0.8)
            F, x, y, pa = dpa.centroid_pa(msk)
            a_.plot([0, x], [0, y], c=col, lw=0.6); a_.plot(x, y, mk, c=col, ms=9, mew=2)
    a_.set_xlim(-half, half); a_.set_ylim(-half, half); a_.set_xticks([]); a_.set_yticks([])
SEG_LEGEND = ([Line2D([], [], c='cyan', marker='x', mew=2), Line2D([], [], c='lime', marker='+', mew=2), Line2D([], [], c='c', lw=0.6)],
              ['n = 0 part / centroid', 'n ≥ 1 part / centroid', 'critical curve'])

def fig_seg(movie_id, run, n_frames=7):
    m, Tm, off, i0 = frames_of(movie_id); r = mov.iloc[m]
    C_, P_ = (x.astype(float) for x in npy_rows_multi(('ml/test_arrays/none/C.npy', f'ml/results/{run}/test_pred.npy'), i0, i0 + Tm))
    n0 = cn0(movie_id)
    cx, cy = dpa.curve_xy(r.a, r.inc); dmap = dpa.dist_map(cx, cy)
    js = np.unique(np.linspace(0, Tm - 1, n_frames).round().astype(int)); vC = C_.max()
    rows_ = [('exact split', C_), ('true C, segmented', C_), (f'{run}, segmented', P_)]
    fig, ax = plt.subplots(3, len(js), figsize=(2.1 * len(js), 6.6), squeeze=False)
    for i, (lab, imgs) in enumerate(rows_):
        for c, j in enumerate(js):
            m0, m1 = dpa.exact_split(n0[j], imgs[j]) if i == 0 else dpa.segment(imgs[j], dmap)[:2]
            draw_seg(ax[i, c], imgs[j], m0, m1, vC, cx, cy)
            if i == 0: ax[i, c].set_title(f'frame {j}  (t = {j * r.cadence_min:.0f} min)', fontsize=8)
        ax[i, 0].set_ylabel(lab, fontsize=8)
    fig.legend(*SEG_LEGEND, loc='lower center', ncol=3, fontsize=8, frameon=False)
    fig.suptitle(f'test movie {movie_id}: share {r.share:.2f}, log R {r.logR:.2f}, inc {r.inc:.0f}°, a {r.a:.2f}, r_spot {r.r_spot:.1f} M  '
                 f'(central 24 M × 24 M, soft-log scale)', fontsize=10)
    plt.tight_layout(rect=(0, 0.04, 1, 1)); return fig

def pa_series(d, src, comp):
    s = d[(d.src == src) & (d.comp == comp)].set_index('k'); return s.PA, s.R

def fig_dpa(movie_id, runs):
    # exact truth black; segmented true C grey dashed; segmented predictions in colour; every curve unwrapped onto the truth
    PA = pa_data()[0]; d = PA[PA.id == movie_id]
    m = row_of_id[movie_id]; cad = mov.cadence_min[m]; srcs = ['exact', 'GT'] + list(runs)
    fig, ax = plt.subplots(1, 3, figsize=(17, 3.8))
    for a_, what in zip(ax, ('PA₀  (n = 0 centroid)', 'PA₁  (n ≥ 1 centroid)', 'DPA = PA₁ − PA₀')):
        ref = None
        for s in srcs:
            p0, _ = pa_series(d, s, 'n0'); p1, R1 = pa_series(d, s, 'n1')
            y = (p0 if what.startswith('PA₀') else p1 if what.startswith('PA₁') else dpa.wrap(p1 - p0)).reindex(range(int(d.k.max()) + 1))
            if ref is None:
                v = y.to_numpy(float).copy(); ok = np.isfinite(v); v[ok] = np.degrees(np.unwrap(np.radians(v[ok]))); y = ref = pd.Series(v, y.index)
            else:
                y = pd.Series(np.where(ref.notna(), ref + dpa.wrap(y - ref), y), y.index)
            st_ = dict(c='k', lw=2.5, ls='-') if s == 'exact' else dict(c='0.55', lw=1.5, ls='--') if s == 'GT' else dict(c=MCOL[parse1(s)[0]], lw=1.2, ls=ls_of(s))
            st_['zorder'] = 3 if s in ('exact', 'GT') else 2
            a_.plot(y.index * cad, y, marker='o', ms=4, label={'exact': 'exact (dataset)', 'GT': 'true C, segmented'}.get(s, s), **st_)
            if not what.startswith('PA₀'):
                bad = (R1 < 2).reindex(y.index).fillna(False).astype(bool)
                a_.plot(y.index[bad] * cad, y[bad], 'o', mfc='w', mec=st_['c'], ms=7)
        a_.set_title(what); a_.set_xlabel('time since the spot appears [min]'); a_.set_ylabel('angle [deg]'); a_.grid(alpha=0.3)
    ax[0].legend(fontsize=7)
    fig.suptitle(f'test movie {movie_id}: share {mov.share[m]:.2f}, log R {mov.logR[m]:.2f}, inc {mov.inc[m]:.0f}°, a {mov.a[m]:.2f}', fontsize=10)
    plt.tight_layout(); return fig

@lru_cache(None)
def pa_frames_all():
    PA, RUNS_PA, _ = pa_data()
    WP = PA.pivot_table(index=['id', 'k', 'comp'], columns='src', values=['PA', 'R', 'F'], aggfunc='first')
    WP.columns = [f'{v}_{s}' for v, s in WP.columns]; WP = WP.reset_index()
    P0, P1 = WP[WP.comp == 'n0'].set_index(['id', 'k']), WP[WP.comp == 'n1'].set_index(['id', 'k'])

    def pa_frames(src, ref='exact'):
        # per test frame: PA errors of the two parts, true and predicted DPA, DPA error (NaN where undefined)
        out = pd.DataFrame(index=P0.index)
        ok0 = (P0[f'F_{ref}'] > 0) & (P0[f'F_{src}'] > 0); ok1 = (P1[f'F_{ref}'] > 0) & (P1[f'F_{src}'] > 0) & (P1[f'R_{ref}'] >= 2)
        out['e0'] = np.abs(dpa.wrap(P0[f'PA_{src}'] - P0[f'PA_{ref}'])).where(ok0)
        out['e1'] = np.abs(dpa.wrap(P1[f'PA_{src}'] - P1[f'PA_{ref}'])).where(ok1)
        out['dpa_true'] = dpa.wrap(P1[f'PA_{ref}'] - P0[f'PA_{ref}']).where(P1[f'R_{ref}'] >= 2)
        out['dpa_pred'] = dpa.wrap(P1[f'PA_{src}'] - P0[f'PA_{src}'])
        out['eD'] = np.abs(dpa.wrap(out.dpa_pred - out.dpa_true))
        for c, P_ in (('0', P0), ('1', P1)):
            out[f'has{c}_t'], out[f'has{c}_p'] = P_[f'F_{ref}'] > 0, P_[f'F_{src}'] > 0
        out = out.reset_index(); out['movie'] = out.id.map(row_of_id); return out
    return {'true C, segmented (floor)': pa_frames('GT')} | {r: pa_frames(r) for r in RUNS_PA}

def pa_summary(f):
    r = {}
    for c, e in (('n0', 'e0'), ('n1', 'e1')):
        ht, hp = f[f'has{c[1]}_t'], f[f'has{c[1]}_p']
        r[f'{c} |ΔPA| med'] = f[e].median(); r[f'{c} |ΔPA| p90'] = f[e].quantile(.9)
        r[f'{c} missed %'] = 100 * (ht & ~hp).sum() / max(ht.sum(), 1); r[f'{c} invented %'] = 100 * (~ht & hp).sum() / max((~ht).sum(), 1)
    r['DPA |Δ| med'] = f.eD.median(); r['DPA |Δ| p90'] = f.eD.quantile(.9); r['DPA frames'] = int(f.eD.notna().sum())
    return r

def fig_dpa_hist():
    PF = pa_frames_all()
    nc = 4; fig, ax = plt.subplots(-(-len(PF) // nc), nc, figsize=(4.1 * nc, 4 * -(-len(PF) // nc)), squeeze=False)
    for a_ in ax.flat[len(PF):]: a_.axis('off')
    for a_, (k, f) in zip(ax.flat, PF.items()):
        ok = f.dpa_true.notna() & f.dpa_pred.notna()
        a_.hist2d(f.dpa_true[ok], f.dpa_pred[ok], 72, range=[[-180, 180], [-180, 180]], cmap='magma_r', cmin=1, norm=mcolors.LogNorm())
        a_.plot([-180, 180], [-180, 180], 'c-', lw=.8)
        a_.set_title(f'{k}\nmedian |ΔDPA| {f.eD.median():.2f}°, within 10°: {(f.eD[ok] < 10).mean():.0%}', fontsize=9)
        a_.set_xlabel('exact DPA [deg]'); a_.set_ylabel('recovered DPA [deg]')
    plt.tight_layout(); return fig

def fig_pa_params():
    PPAR = ['logR', 'share', 'inc', 'a', 'r_spot', 'th_spot', 'diam', 'K']
    fig, ax = plt.subplots(3, len(PPAR), figsize=(3.2 * len(PPAR), 8.4), squeeze=False)
    for k, f in pa_frames_all().items():
        st_ = dict(c='0.55', ls='--', lw=1.5, label=k) if k.startswith('true') else style1(k)
        for i, e in enumerate(('e0', 'e1', 'eD')):
            for j, p in enumerate(PPAR): ax[i, j].plot(*binned(xval(p, f.movie), f[e]), **st_)
    for i, lab in enumerate(('|ΔPA₀| [deg]', '|ΔPA₁| [deg]', '|ΔDPA| [deg]')):
        ax[i, 0].set_ylabel(f'median {lab}')
        for j, p in enumerate(PPAR):
            ax[i, j].set_yscale('log'); ax[i, j].grid(alpha=.3, which='both'); ax[i, j].tick_params(labelsize=7)
            if i == 2: ax[i, j].set_xlabel(xlab(p), fontsize=8)
    ax[0, 0].legend(fontsize=7); plt.tight_layout(); return fig


# ======================================================================================= cells 42-46: task 2
RP = 'ml/results_params'
TGT = {'inc': 'inclination [deg]', 'a': 'spin a', 'th_spot': 'θ_spot [deg]', 'r_spot': 'r_spot [M]', 'logP': 'period: |P̂/P − 1|'}
SET2 = [(arch, st_, tag) for tag in TAGS for arch in ('pool', 'xattn') for st_ in ('I', 'IQU')]
base2 = lambda arch, st_, tag: f'{arch}_{st_}' + ('' if tag == 'none' else f'_{tag}')
KEYS2 = [base2(*k) for k in SET2]

@lru_cache(None)
def task2():
    DEC = open(fetch(f'{RP}/decision.txt')).read() if exists(f'{RP}/decision.txt') else ''
    DESIGN = 'separate' if 'decision: SEPARATE' in DEC else 'joint'

    def load_t2(arch, st_, tag):
        # one DataFrame per setting (movie, id, <t>_true/_pred/_sig for the targets that exist), and the runs it came from
        b = base2(arch, st_, tag)
        joint = f'{b}_rP'; sep = [f'{b}_t{n}' for n in TGT]
        order = [[joint], sep] if DESIGN == 'joint' else [sep, [joint]]
        for runs in order:
            ok = [r for r in runs if exists(f'{RP}/{r}/pred_test.csv')]
            if ok:
                d = read_csv(f'{RP}/{ok[0]}/pred_test.csv')
                for r in ok[1:]: d = d.merge(read_csv(f'{RP}/{r}/pred_test.csv').drop(columns='id'), on='movie')
                return d, ok
        return None, []

    P2 = {}
    for a_, s_, t_ in SET2:
        d, src = load_t2(a_, s_, t_)
        if d is not None: P2[base2(a_, s_, t_)] = d
    return DEC, DESIGN, P2

def t2_err(d, n):
    return np.abs(10 ** (d.logP_pred - d.logP_true) - 1) if n == 'logP' else np.abs(d[n + '_pred'] - d[n + '_true'])

MK2 = {'I': dict(marker='o'), 'IQU': dict(marker='o', mfc='w')}
def style2(k):
    p = k.split('_'); tag = '_'.join(p[2:]) or 'none'
    return dict(c=TCOL[tag], ls='-' if p[0] == 'pool' else '--', ms=4, lw=1.2, label=k, **MK2[p[1]])

def fig_t2_grid(keys, params):
    P2 = task2()[2]
    params = list(params); nt = len(TGT)
    fig, ax = plt.subplots(len(params), nt, figsize=(3.6 * nt, 2.5 * len(params) + 0.8), squeeze=False)
    for k in keys:
        d = P2[k]; mr = d.movie.to_numpy()
        for j, n in enumerate(TGT):
            if n + '_pred' not in d: continue
            e = t2_err(d, n).to_numpy()
            for i, p in enumerate(params): ax[i, j].plot(*binned(xval(p, mr), e, 8), **style2(k))
    for j, n in enumerate(TGT): ax[0, j].set_title(TGT[n], fontsize=10)
    for i, p in enumerate(params):
        for j in range(nt):
            a_ = ax[i, j]; a_.set_yscale('log'); a_.grid(alpha=.3, which='both'); a_.tick_params(labelsize=7); a_.set_xlabel(xlab(p), fontsize=8)
        ax[i, 0].set_ylabel('median |error|', fontsize=8)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', ncol=6, fontsize=8, frameon=False)
    plt.tight_layout(rect=(0, 0, 1, 1 - 0.8 / (2.5 * len(params) + 0.8))); return fig


# ======================================================================================= cells 48-62: blurred movies
RB = [r for r in T if parse1(r)[2] != 'none']
_rows = {}
for r in T1:
    o, s, tag = parse1(r)
    _rows[r] = {'method': METHOD[o], 'input': s, 'data': TAG_LAB[tag], 'status': status(r), **(scores1(T[r]) if r in T else {})}
tabO = pd.DataFrame(_rows).T
for c in SC1: tabO[c] = tabO[c].astype(float) if c in tabO else np.nan
tabO = tabO.loc[sorted(tabO.index, key=lambda r: (parse1(r)[0], parse1(r)[1], TAGS.index(parse1(r)[2])))]

def fig_31():
    m = row_of_id[MID0]; f = fra[fra.movie == m]; g = int(f.index[f.has_n0][len(f.index[f.has_n0]) // 2])
    fig, ax = plt.subplots(3, 3, figsize=(8.4, 8.4))
    for j, tag in enumerate(TAGS):
        a = npy_rows(f'ml/test_arrays/{tag}/A.npy', row_of[g], row_of[g] + 1)[0, 0].astype(float)
        c = npy_rows(f'ml/test_arrays/{tag}/C.npy', row_of[g], row_of[g] + 1)[0].astype(float)
        for i, (img, st_) in enumerate(((a, 'asinh'), (a, 'linear'), (c, 'asinh'))):
            ax[i, j].imshow(soft(img, img.max(), st_).T, origin='lower', cmap='afmhot', vmin=0, vmax=1, extent=EXT)
            ax[i, j].set_title(f"{'AAC'[i]}, {TAG_LAB[tag]} ({st_})", fontsize=8); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
    fig.suptitle(f'test movie {MID0}, frame {int(fra.k[g])}: share {mov.share[m]:.2f}, log R {mov.logR[m]:.2f}, inc {mov.inc[m]:.0f}°', fontsize=10)
    plt.tight_layout(); return fig

def fig_32():
    pan = ['relL2 median', 'relL2 p90', 'relL2 only n≥1', 'flux err median', 'MAE [µJy/px]', '|ΔF| median [mJy]']
    fig, ax = plt.subplots(2, len(pan), figsize=(4.3 * len(pan), 6.6), squeeze=False)
    for i, s in enumerate(('I', 'IQU')):
        for a_, sc in zip(ax[i], pan):
            v = np.array([[tabO.loc[run1(o, s, tag), sc] for tag in TAGS] for o in (1, 2, 3, 4, 5)], float); ok = np.isfinite(v)
            if ok.any(): a_.imshow(v, cmap='RdYlGn_r', norm=mcolors.LogNorm(np.nanmin(v), max(np.nanmax(v), np.nanmin(v) * 1.01)), aspect='auto')
            else: a_.imshow(np.zeros_like(v), cmap='Greys', vmin=0, vmax=1, aspect='auto')
            for (p, q) in zip(*np.nonzero(ok)): a_.text(q, p, f'{v[p, q]:.3g}', ha='center', va='center', fontsize=8)
            for (p, q) in zip(*np.nonzero(~ok)): a_.text(q, p, status(run1(p + 1, s, TAGS[q])), ha='center', va='center', fontsize=7, c='0.5')
            a_.set_xticks(range(3)); a_.set_xticklabels([TAG_LAB[t] for t in TAGS], fontsize=8)
            a_.set_yticks(range(5)); a_.set_yticklabels([METHOD[o] for o in (1, 2, 3, 4, 5)] if a_ is ax[i, 0] else [], fontsize=8)
            a_.set_title(f'{sc}, input {s}', fontsize=9)
    plt.tight_layout(); return fig

def style3(r): o, s, tag = parse1(r); return dict(c=TCOL[tag], ls=ls_of(r), marker='o', ms=3, label=f'{TAG_LAB[tag]}, {s}')
def fig_survive(p):
    meths = [o for o in (1, 2, 3, 4, 5) if any(parse1(r)[0] == o for r in RB)]
    mets = ['relL2, frames with n=0', 'relL2, only-n≥1 frames', 'error per pixel [µJy/px]']
    fig, ax = plt.subplots(3, len(meths), figsize=(4.6 * len(meths), 10), squeeze=False, sharey='row')
    for j, o in enumerate(meths):
        for r in [run1(o, s, tag) for tag in TAGS for s in ('I', 'IQU')]:
            if r not in T: continue
            for i, mt in enumerate(mets):
                col, has = MET1[mt]; t = sel(T[r], has); ax[i, j].plot(*binned(xval(p, t.movie), t[col]), **style3(r))
        ax[0, j].set_title(METHOD[o], fontsize=10)
        for a_ in ax[:, j]: a_.set_yscale('log'); a_.set_xlabel(xlab(p), fontsize=8); a_.grid(alpha=.3, which='both')
    for i, mt in enumerate(mets): ax[i, 0].set_ylabel(f'median {mt}', fontsize=8)
    ax[0, 0].legend(fontsize=7); plt.tight_layout(); return fig

def fig_lc_blur(movie, method, stokes):
    m = row_of_id[movie]; f = fra[fra.movie == m]; nz = lambda y: y.where(y > 0)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw=dict(height_ratios=[2, 1]))
    ax[0].plot(f.t_min, nz(f.FC), 'k-', lw=2.5, label='truth')
    for tag in TAGS:
        r = run1(method, stokes, tag)
        if r not in T: continue
        tm = T[r][T[r].movie == m]; kw = dict(c='0.6', lw=3) if tag == 'none' else dict(c=TCOL[tag])
        ax[0].plot(tm.t_min, nz(tm.flux_pred * MOV[tag].scale[m]), marker='.', label=TAG_LAB[tag], **kw); ax[1].plot(tm.t_min, tm.dF, marker='.', **kw)
    ax[0].set_yscale('log'); ax[0].set_ylabel('spot flux [Jy]'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which='both')
    ax[1].axhline(0, c='k', lw=.8); ax[1].set_ylabel('ΔF [mJy]'); ax[1].set_xlabel('time [min]'); ax[1].grid(alpha=.3)
    ax[0].set_title(f'{METHOD[method]}, input {stokes} — test movie {movie}: share {mov.share[m]:.2f}, log R {mov.logR[m]:.2f}', fontsize=10)
    plt.tight_layout(); return fig


# ======================================================================================= pages
DELTA_FMT = {c: '{:+.4f}' for c in ['Δ [µJy/px]', 'Δ 95% lo', 'Δ 95% hi']}


def page_overview():
    md(5, ('Section numbers follow the v1 notebook. Runs still training show up as empty rows / cells marked with their status; re-running the\n'
           'cells fills them in (the notebook updates its own caches of pixel errors and centroid angles).\n'
           'Sliders and check boxes need a live kernel; a static run (`NB_HEADLESS=1`) shows one default movie instead.',
           'Section numbers follow the v1 notebook. Each part has its own page (sidebar).'))
    md(6)
    md(8)
    st.caption(f'task-1 runs with test results: {len(done1)} of {len(T1)}')
    table(status_grid(T1, lambda r: '_'.join(r.split('_')[:2]), lambda r: TAG_LAB[parse1(r)[2]],
                      [f'opt{o}_{s}' for o in (1, 2, 3, 4, 5) for s in ('I', 'IQU')], [TAG_LAB[t] for t in TAGS]))


def page_scores():
    md(10)
    md(11)
    st.caption(f'{len(t0)} test frames from {t0.movie.nunique()} test movies; {(~t0.has_n0).mean():.0%} contain only n≥1 light.  '
               f'Truth per pixel: mean |C| = {t0.zero.mean():.2f} µJy/px; mean spot flux per frame = '
               f'{(t0.flux_true * mov.scale.to_numpy()[t0.movie]).mean() * 1e3:.0f} mJy')
    table(style_scores(tab1, SC1))
    md(13)
    plot(fig_bars)
    md(15)
    pairs = tuple((METHOD[o], run1(o, 'I'), run1(o, 'IQU')) for o in (1, 2, 3, 4, 5) if run1(o, 'I') in T and run1(o, 'IQU') in T)
    if pairs:
        table(delta_table(pairs).rename(columns={'MAE ref [µJy/px]': 'MAE input I', 'MAE [µJy/px]': 'MAE input IQU'})
              .style.format(DELTA_FMT | {'MAE input I': '{:.4f}', 'MAE input IQU': '{:.4f}', 'Δ [%]': '{:+.1f}', 'Δ summed per frame [mJy]': '{:+.2f}'}))
    if len(R0) > 1:
        plot(fig_delta_matrix, tuple(R0), 'MAE(row) − MAE(column) [µJy/px], unblurred test frames  (* = 95 % bootstrap interval excludes 0)')
    md(17)
    plot(fig_head_to_head, tuple(R0))
    md(19)
    plot(fig_metric_vs, 'logR', ('relL2, frames with n=0', 'relL2, only-n≥1 frames', 'flux error |ln F̂/F|', 'error per pixel [µJy/px]'), tuple(R0))
    md(21, ('\n(static view: relL2 over all frames and the error per pixel, unblurred).', '.'))
    a, b = st.columns(2)
    metric = a.selectbox('error', list(MET1), key='g_m')
    tag = b.selectbox('data', TAGS, format_func=TAG_LAB.get, key='g_t')
    params = st.multiselect('parameters', list(XPAR), default=list(XPAR), format_func=xlab, key='g_p')
    runs = tuple(r for r in T if parse1(r)[2] == tag)
    if not runs: st.info(f'no finished runs for {TAG_LAB[tag]}')
    elif params: plot(fig_param_grid, runs, metric, tuple(params), f'{metric} — {TAG_LAB[tag]}')
    md(23)
    plot(fig_along)
    md(25)
    if iqu_pairs('none'): plot(fig_iqu_vs_i, 'none')
    else: st.info('need both I and IQU runs of at least one method')


def page_movies():
    md(27, ('Pick a test movie with the slider', 'Pick a test movie with the slider below'))
    c = controls('p1')
    movie, tag = c['movie'], c['tag']
    runs = picked(tag, c)
    try:
        plot(fig_movie, movie, tag, runs, c['n_frames'], c['stretch'])
    except Exception as e:
        st.info(f'could not read the arrays of this movie ({type(e).__name__}: {e}); some files may still be uploading')
    md(29)
    plot(fig_lc, movie, tag, runs)
    md(31)
    rr = runs_B(tag, runs)
    if rr:
        plot(fig_disk, movie, tag, rr)
    else:
        st.info('tick a finished method-4 or method-5 network for this data version')
    if exists('ml/web/B_summary.csv'):
        bs = read_csv('ml/web/B_summary.csv', index_col='run')
        bs.insert(0, 'data', bs.pop('tag').map(TAG_LAB))
        table(bs.style.format({c_: '{:.3f}' for c_ in ['relL2 of B̂ (median)', '|B̂−B| [µJy/px] (median)']}
                              | {'Σ(B̂−B) [mJy] (median)': '{:+.2f}', 'mean B [µJy/px]': '{:.1f}'}))
    md(33)
    PA, RUNS_PA, BEST = pa_data()
    st.caption(f'networks in the PA table: {", ".join(RUNS_PA)} | best unblurred network (median relL2): {BEST}')
    a, b = st.columns([1, 2])
    net = a.selectbox('network', RUNS_PA, index=RUNS_PA.index(BEST), key='pa_net')
    all_nets = b.checkbox('all networks in the PA plot', False, key='pa_all')
    st.caption(f'movie and number of frames from the controls of Section 1.6 above (test movie {movie})')
    try:
        plot(fig_seg, movie, net, c['n_frames'])
    except Exception as e:
        st.info(f'could not read this movie ({type(e).__name__}); its dataset file may still be uploading')
    plot(fig_dpa, movie, tuple(RUNS_PA) if all_nets else (net,))
    md(35)
    with st.spinner('computing the angle errors of all test frames'):
        PF = pa_frames_all()
    f_ = PF['true C, segmented (floor)']
    st.caption(f'{len(f_)} test frames; truth: n=0 present in {f_.has0_t.sum()}, n≥1 present in {f_.has1_t.sum()}, '
               f'DPA defined (both present, R̄₁ ≥ 2 M) in {f_.dpa_true.notna().sum()}.  Angles in degrees.')
    table(pd.DataFrame({k: pa_summary(v) for k, v in PF.items()}).T.round(2))
    md(37)
    plot(fig_dpa_hist)
    md(39)
    plot(fig_pa_params)


def page_task2():
    md(41)
    DEC, DESIGN, P2 = task2()
    st.caption('task-2 design: ' + (DESIGN if DEC else 'not decided yet (comparison runs pending); showing whatever exists')
               + f' | settings with test results: {len(P2)} of {len(SET2)}')
    table(status_grid([f'{base2(*k)}_rP' for k in SET2], lambda r: '_'.join(r.split('_')[:2]), lambda r: TAG_LAB['_'.join(r.split('_')[2:-1]) or 'none'],
                      [f'{a}_{s}' for a in ('pool', 'xattn') for s in ('I', 'IQU')], [TAG_LAB[t] for t in TAGS]))
    md(43)
    rows = {}
    for k, d in P2.items():
        row = {}
        for n in TGT:
            if n + '_pred' not in d: continue
            e = d[n + '_pred'] - d[n + '_true']
            if n == 'logP':
                row[(n, 'RMSE [min]')] = np.sqrt(((10 ** d.logP_pred - 10 ** d.logP_true) ** 2).mean()); row[(n, 'median |P̂/P−1|')] = t2_err(d, n).median()
            else:
                row[(n, 'RMSE')] = np.sqrt((e ** 2).mean())
            row[(n, 'R²')] = 1 - (e ** 2).sum() / ((d[n + '_true'] - d[n + '_true'].mean()) ** 2).sum()
            row[(n, '1σ cov.')] = (e.abs() < d[n + '_sig']).mean()
        rows[k] = row
    if rows:
        tab2 = pd.DataFrame(rows).T
        fmt = {f'{n} {c}': '{:.0%}' if c == '1σ cov.' else '{:.4f}' if c in ('R²', 'median |P̂/P−1|') else '{:.3g}' for n, c in tab2.columns}
        tab2.columns = [f'{n} {c}' for n, c in tab2.columns]
        table(tab2.style.format(fmt))
    else:
        st.info('no task-2 test results yet')
    md(45, ('Untick settings / pick parameters in a live kernel.', 'Untick settings / pick parameters below.'))
    keys = []
    for t in TAGS:
        cols = st.columns([1.3] + [1] * 4)
        cols[0].markdown(f'**{TAG_LAB[t]}**')
        for col, k in zip(cols[1:], [k for k in KEYS2 if ('_'.join(k.split('_')[2:]) or 'none') == t]):
            if col.checkbox(k, True, key=f't2_{k}', disabled=k not in P2): keys.append(k)
    params = st.multiselect('parameters', list(XPAR), default=list(XPAR), format_func=xlab, key='t2_p')
    keys = tuple(k for k in keys if k in P2)
    if not keys: st.info('no task-2 test results for the chosen settings')
    elif params: plot(fig_t2_grid, keys, tuple(params))


def page_blur():
    md(47)
    st.caption(f'blurred runs with test results: {len(RB)} of {len([r for r in T1 if parse1(r)[2] != "none"])}')
    table(style_scores(tabO, SC1))
    md(49)
    plot(fig_31)
    md(51)
    plot(fig_32)
    md(53)
    pairs = tuple((f'{run1(o, s)} → {TAG_LAB[tag]}', run1(o, s), run1(o, s, tag)) for o in (1, 2, 3, 4, 5) for s in ('I', 'IQU') for tag in TAGS[1:]
                  if run1(o, s) in T and run1(o, s, tag) in T)
    if pairs:
        dt = delta_table(pairs).copy()
        dt['|ΔF| med, unblurred [mJy]'] = [T[a].dF.abs().median() for _, a, _ in pairs]; dt['|ΔF| med, blurred [mJy]'] = [T[b].dF.abs().median() for _, _, b in pairs]
        table(dt.rename(columns={'MAE ref [µJy/px]': 'MAE unblurred', 'MAE [µJy/px]': 'MAE blurred'})
              .style.format(DELTA_FMT | {'MAE unblurred': '{:.4f}', 'MAE blurred': '{:.4f}', 'Δ [%]': '{:+.1f}',
                                         'Δ summed per frame [mJy]': '{:+.2f}', '|ΔF| med, unblurred [mJy]': '{:.2f}', '|ΔF| med, blurred [mJy]': '{:.2f}'}))
    else:
        st.info('no method has both an unblurred and a blurred result yet')
    for tag in TAGS[1:]:
        rr = tuple(r for r in T if parse1(r)[2] == tag)
        if len(rr) > 1: plot(fig_delta_matrix, rr, f'MAE(row) − MAE(column) [µJy/px], {TAG_LAB[tag]}  (* = 95 % bootstrap interval excludes 0)')
    md(55)
    p = st.selectbox('x axis', list(XPAR), format_func=xlab, key='s3')
    if RB: plot(fig_survive, p)
    else: st.info('no blurred results yet')
    md(57)
    rows = []
    for o in (1, 2, 3, 4, 5):
        for tag in TAGS:
            rI, rQ = run1(o, 'I', tag), run1(o, 'IQU', tag)
            if rI not in T or rQ not in T: continue
            x, y = T[rI].relL2.to_numpy(), T[rQ].relL2.to_numpy(); dt = delta_table((('', rI, rQ),)).iloc[0]
            rows.append({'method': METHOD[o], 'data': TAG_LAB[tag], 'IQU better on': (y < x).mean(), 'median relL2(IQU)/relL2(I)': np.median(y / x),
                         'relL2 median, I': np.median(x), 'relL2 median, IQU': np.median(y), 'ΔMAE [µJy/px]': dt['Δ [µJy/px]'],
                         '95% interval': f"[{dt['Δ 95% lo']:+.4f}, {dt['Δ 95% hi']:+.4f}]{dt['significant']}"})
    if rows:
        table(pd.DataFrame(rows).style.format({'IQU better on': '{:.0%}', 'median relL2(IQU)/relL2(I)': '{:.2f}', 'relL2 median, I': '{:.3f}',
                                               'relL2 median, IQU': '{:.3f}', 'ΔMAE [µJy/px]': '{:+.4f}'}))
    else:
        st.info('no I / IQU pairs yet')
    md(59)
    c = controls('p3', 'eht_n0')
    try:
        plot(fig_movie, c['movie'], c['tag'], picked(c['tag'], c), c['n_frames'], c['stretch'])
    except Exception as e:
        st.info(f'could not read the arrays of this movie ({type(e).__name__}: {e}); some files may still be uploading')
    md(61)
    a, b = st.columns(2)
    method = a.selectbox('method', (1, 2, 3, 4, 5), index=4, format_func=METHOD.get, key='lc3_m')
    stokes = b.selectbox('input', ('I', 'IQU'), key='lc3_s')
    miss = [f'{run1(method, stokes, t)}: {status(run1(method, stokes, t))}' for t in TAGS if run1(method, stokes, t) not in T]
    if miss: st.caption(' | '.join(miss))
    plot(fig_lc_blur, c['movie'], method, stokes)
    md(63)


PAGES = {'ML · overview and runs': page_overview, 'ML 1 · task 1 scores': page_scores, 'ML 1 · task 1 movies and angles': page_movies,
         'ML 2 · task 2 parameters': page_task2, 'ML 3 · blurred movies': page_blur}
