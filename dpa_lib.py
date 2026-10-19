"""Pure geometry / segmentation helpers of ipole_abc_ml_v2/code/dpa.py (copied verbatim; the data-loading parts of dpa.py
read the cluster paths and are replaced by store.py / ml.py here)."""
import numpy as np
import pandas as pd


# ---------- Kerr critical curve (M = 1) and image geometry ----------
NPX, FOV = 128, 32.0
DX = FOV / NPX
XC = (np.arange(NPX) + 0.5) * DX - FOV / 2          # pixel-centre coordinates in M, image centre = black hole
X, Y = np.meshgrid(XC, XC, indexing='ij')            # arrays are [x, y]


def critical_curve(a, inc_deg, n=4000):
    """Bardeen critical curve (alpha, beta) in M: alpha along the projected spin-perpendicular axis."""
    a = a if abs(a) > 1e-3 else np.copysign(1e-3, a if a != 0 else 1)
    rp = 2 * (1 + np.cos(2 / 3 * np.arccos(-abs(a)))); rm = 2 * (1 + np.cos(2 / 3 * np.arccos(abs(a))))
    r = np.linspace(rp, rm, n)
    xi = (r**2 * (3 - r) - a**2 * (r + 1)) / (a * (r - 1))
    eta = r**3 * (4 * a**2 - r * (r - 3)**2) / (a**2 * (r - 1)**2)
    i = np.radians(inc_deg)
    b2 = eta + a**2 * np.cos(i)**2 - xi**2 / np.tan(i)**2
    ok = b2 >= 0
    al, be = -xi[ok] / np.sin(i), np.sqrt(b2[ok])
    return np.r_[al, al[::-1]], np.r_[be, -be[::-1]]


def curve_xy(a, inc_deg, orient=(0, 1, 1)):
    """critical curve in image (x, y); orient = (swap, sx, sy) maps (alpha, beta) to the ipole image axes."""
    al, be = critical_curve(a, inc_deg)
    swap, sx, sy = orient
    x, y = (be, al) if swap else (al, be)
    return sx * x, sy * y


def dist_map(cx, cy):
    """distance (M) of every pixel from the curve."""
    from scipy.spatial import cKDTree
    d, _ = cKDTree(np.c_[cx, cy]).query(np.c_[X.ravel(), Y.ravel()])
    return d.reshape(NPX, NPX)


# ---------- segmentation of a spot image into n=0 and n>=1 ----------
def components(img, tau=3e-3, grow=1):
    """connected components of img > tau * max(img) (8-connected, after growing the mask by `grow` px so dotted
    arcs stay one component); returns label map (0 = background) restricted to the thresholded pixels."""
    from scipy import ndimage as nd
    m = img > tau * img.max() if img.max() > 0 else np.zeros(img.shape, bool)
    lab, n = nd.label(nd.binary_dilation(m, iterations=grow) if grow else m, structure=np.ones((3, 3)))
    return np.where(m, lab, 0), n


def comp_table(img, dmap, tau=3e-3, grow=1):
    """one row per component: flux, flux-weighted mean distance to the critical curve d_bar (M), area (px)."""
    lab, n = components(img, tau, grow)
    rows = []
    for k in range(1, n + 1):
        w = np.where(lab == k, np.clip(img, 0, None), 0); F = w.sum()
        if F > 0:
            rows.append(dict(k=k, F=F, d_bar=(w * dmap).sum() / F, area=int((lab == k).sum())))
    t = pd.DataFrame(rows)
    if len(t):
        t['F_rel'] = t.F / t.F.max()
    return t, lab


D_MIN, AREA_MIN, F_MIN = 0.3, 30, 0.02


def segment(img, dmap, tau=3e-3):
    """split img into (n0, n1, lab, is0): the n=0 image is the single connected component lying farthest from the
    critical curve (flux-weighted mean distance d_bar), provided d_bar > D_MIN (M), it covers > AREA_MIN px and holds
    > F_MIN of the brightest component's flux; every other component is n>=1 (rule tuned on v1)."""
    t, lab = comp_table(img, dmap, tau)
    is0 = {}
    if len(t):
        ok = (t.F_rel > F_MIN) & (t.d_bar > D_MIN) & (t.area > AREA_MIN)
        k0 = t.k[t.d_bar.where(ok).idxmax()] if ok.any() else -1
        is0 = {int(k): k == k0 for k in t.k}
    im = np.clip(img, 0, None)
    n0 = np.isin(lab, [k for k, v in is0.items() if v]); n1 = np.isin(lab, [k for k, v in is0.items() if not v])
    return im * n0, im * n1, lab, is0


def exact_split(n0, c, tau=3e-3):
    """exact (n0, n1) images from the stored direct image, with the same brightness cut as segment()."""
    cut = tau * c.max(); n1 = c - n0
    return np.where(n0 > cut, n0, 0), np.where(n1 > cut, n1, 0)


def centroid_pa(img):
    """flux, centroid (x, y) in M and position angle PA (deg, counter-clockwise from 'up' in imshow(img.T,
    origin='lower')) of an image; NaNs if empty."""
    F = img.sum()
    if F <= 0:
        return F, np.nan, np.nan, np.nan
    x, y = (img * X).sum() / F, (img * Y).sum() / F
    return F, x, y, np.degrees(np.arctan2(-x, y))


def wrap(d):
    return (d + 180) % 360 - 180          # keeps pandas Series as Series
