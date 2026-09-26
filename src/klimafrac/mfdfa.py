"""
Multifractal detrended fluctuation analysis (Kantelhardt et al. 2002).

Every analysis imports the estimator from here, so there is one
implementation to check. For a series x of length N:

  profile   Y(k) = sum_{i<=k} (x_i - mean(x))
  segments  2 * floor(N/s) non-overlapping segments of length s, scanned from
            both ends; a polynomial trend of order 1 is removed in each and
            F^2(v, s) is the mean squared residual
  moments   F_q(s) = { mean_v [F^2(v,s)]^(q/2) }^(1/q),  q != 0
  scaling   h(q) is the least-squares slope of log F_q(s) on log s over
            up to 20 log-spaced integer scales from s_min = 10 to s_max
  spectrum  tau(q) = q h(q) - 1, alpha = d tau / d q, f = q alpha - tau

Flat segments. Where every return inside a segment is identical -- in a
calendar-time series of a thinly traded asset, a stretch with no trade -- the
detrended residual is exactly zero. For q > 0 such a segment contributes an
exact zero to the mean. For q < 0 its moment is infinite and F_q is undefined,
so flat segments are excluded from the negative moments. They are detected
exactly from the returns, not from the computed residual, which in floating
point is round-off of order 1e-30 rather than zero; keeping those round-off
values instead (`exclude_flat=False`) makes h(q < 0) a function of machine
precision, and is available only to demonstrate that.

Two q ranges:
  QS      -5 to 5 in steps of 0.5, excluding 0
  QS_POS  0.5 to 5; the positive-side width uses only these
"""

import numpy as np

QS = np.concatenate([np.arange(-5, 0, 0.5), np.arange(0.5, 5.5, 0.5)])
QS_POS = np.arange(0.5, 5.5, 0.5)
S_MIN = 10
N_SCALES = 20
POLY_ORDER = 1


def scales(n: int, s_min: int, s_max: int, count: int = N_SCALES) -> np.ndarray:
    if s_max <= s_min:
        return np.array([], dtype=int)
    s = np.unique(np.round(np.logspace(np.log10(s_min), np.log10(s_max), count)).astype(int))
    return s[s >= s_min]


def segment_fluctuations(x: np.ndarray, profile: np.ndarray, s: int,
                         order: int = POLY_ORDER) -> tuple[np.ndarray, np.ndarray] | None:
    """F^2 per segment (both scan directions) and a mask of flat segments."""
    n = len(profile)
    n_seg = n // s
    if n_seg < 4:
        return None
    t = np.arange(s)
    f2, flat = [], []
    for offset in (0, n - n_seg * s):
        seg = profile[offset:offset + n_seg * s].reshape(n_seg, s)
        coef = np.polynomial.polynomial.polyfit(t, seg.T, order)
        resid = seg - np.polynomial.polynomial.polyval(t, coef)
        f2.append(np.mean(resid ** 2, axis=1))
        steps = x[offset:offset + n_seg * s].reshape(n_seg, s)[:, 1:]
        flat.append(np.ptp(steps, axis=1) == 0)
    return np.concatenate(f2), np.concatenate(flat)


def mfdfa(x, s_min: int = S_MIN, s_max: int | None = None, qs=QS,
          order: int = POLY_ORDER, exclude_flat: bool = True):
    """Generalised Hurst exponents. Returns (scales, h), or (None, None)."""
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains non-finite values")
    n = len(x)
    s_max = n // 4 if s_max is None else s_max
    profile = np.cumsum(x - x.mean())
    qs = np.asarray(qs, dtype=float)

    used, fq = [], []
    for s in scales(n, s_min, s_max):
        res = segment_fluctuations(x, profile, int(s), order)
        if res is None:
            continue
        f2, flat = res
        if exclude_flat:
            f2 = np.where(flat, 0.0, f2)
        live = f2[f2 > 0]
        if len(live) < 4:
            continue
        row = []
        for q in qs:
            if q > 0:
                row.append(np.mean(f2 ** (q / 2.0)) ** (1.0 / q))
            elif q < 0:
                row.append(np.mean(live ** (q / 2.0)) ** (1.0 / q))
            else:
                row.append(np.exp(0.5 * np.mean(np.log(live))))
        used.append(s)
        fq.append(row)

    if len(used) < 5:
        return None, None
    used = np.array(used)
    logs = np.log(used)
    fq = np.array(fq)
    h = np.array([np.polyfit(logs, np.log(fq[:, i]), 1)[0] for i in range(len(qs))])
    return used, h


def spectrum(qs, h):
    """Legendre transform. Returns (alpha, f(alpha), width, alpha0)."""
    qs = np.asarray(qs, dtype=float)
    tau = qs * h - 1.0
    alpha = np.gradient(tau, qs)
    falpha = qs * alpha - tau
    return alpha, falpha, float(alpha.max() - alpha.min()), float(alpha[np.argmax(falpha)])


def full_spectrum(x, s_max_div: int, exclude_flat: bool = True) -> dict | None:
    """Width and peak over the full q range, with h(2) and h(-5)."""
    x = np.asarray(x, dtype=float)
    _, h = mfdfa(x, s_max=len(x) // s_max_div, qs=QS, exclude_flat=exclude_flat)
    if h is None:
        return None
    _, _, width, alpha0 = spectrum(QS, h)
    return {"n": len(x), "width": width, "alpha0": alpha0,
            "h2": float(h[QS == 2.0][0]), "h_neg5": float(h[0])}


def positive_spectrum(x, s_max_div: int, min_n: int = 400) -> dict | None:
    """h(1), h(2), h(5) and the width of the spectrum over q in [0.5, 5]."""
    x = np.asarray(x, dtype=float)
    if len(x) < min_n:
        return None
    _, h = mfdfa(x, s_max=len(x) // s_max_div, qs=QS_POS)
    if h is None:
        return None
    alpha = np.gradient(QS_POS * h - 1.0, QS_POS)
    return {"n": len(x),
            "h1": float(h[QS_POS == 1.0][0]),
            "h2": float(h[QS_POS == 2.0][0]),
            "h5": float(h[-1]),
            "width_pos": float(alpha.max() - alpha.min())}


# --- surrogates ------------------------------------------------------------

def phase_randomise(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same power spectrum, Gaussianised distribution."""
    f = np.fft.rfft(x)
    phases = rng.uniform(0, 2 * np.pi, len(f))
    phases[0] = 0.0
    return np.fft.irfft(np.abs(f) * np.exp(1j * phases), n=len(x))


# --- validation ------------------------------------------------------------

def validate(n: int = 2000, reps: int = 20, s_max_div: int = 10, tol: float = 0.05,
             verbose: bool = True) -> dict:
    """h(2) must recover the Hurst exponent of synthetic fBm."""
    from klimafrac.synthetic import fbm

    out = {}
    for hurst in (0.3, 0.5, 0.7):
        est = []
        for i in range(reps):
            _, h = mfdfa(fbm(n, hurst, np.random.default_rng(100 + i)),
                         s_max=n // s_max_div, qs=np.array([2.0]))
            est.append(h[0])
        out[hurst] = float(np.mean(est))
        if verbose:
            ok = abs(out[hurst] - hurst) < tol
            print(f"  H={hurst}: h(2) = {out[hurst]:.3f}  {'PASS' if ok else 'FAIL'}")
    if not all(abs(v - k) < tol for k, v in out.items()):
        raise AssertionError(f"estimator validation failed: {out}")
    return out


if __name__ == "__main__":
    # Writes the recovered exponents to data/results/mfdfa_validation.json.
    from klimafrac.results import numbers

    design = {"n": 2000, "reps": 20, "s_max_div": 10, "tol": 0.05}
    recovered = validate(**design)
    print("wrote", numbers("mfdfa_validation",
                           {f"h2_at_H_{h}": v for h, v in recovered.items()}
                           | {f"design_{k}": v for k, v in design.items()}))
