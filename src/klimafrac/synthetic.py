"""
Processes with known scaling, for validating the estimator and for simulating
size and power.

`fbm` gives monofractal increments of fractional Brownian motion. `mrw` gives
increments of the multifractal random walk of Bacry, Delour and Muzy (2001):

    X_i = eps_i * exp(omega_i)

with eps i.i.d. standard Gaussian and omega a stationary Gaussian process with

    Cov(omega_i, omega_j) = lambda2 * ln(L / (|i - j| + 1))   for |i - j| < L
                            0                                  otherwise

and mean -Var(omega)/2, so that E[exp(omega)] = 1 and the scale of X does not
depend on lambda2. lambda2 = 0 is a monofractal random walk; the spectrum
widens with lambda2. L is the integral scale, fixed by the caller rather than
tied to the series length, so that series of different lengths are draws from
the same process.

Both Gaussian processes are generated exactly by circulant embedding
(Davies and Harte 1987). `check_embedding` verifies the variance and
covariance of the generated omega against the target.
"""

import numpy as np

INTEGRAL_SCALE = 4096


def _circulant_gaussian(cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A stationary Gaussian series of len(cov) with autocovariance `cov`."""
    n = len(cov)
    circ = np.concatenate([cov, [0.0] if n > 1 else [], cov[:0:-1]])
    eig = np.real(np.fft.fft(circ))
    if eig.min() < -1e-8 * eig.max():
        raise ValueError("covariance is not embeddable at this length")
    eig = np.clip(eig, 0.0, None)
    m = len(circ)
    z = rng.normal(size=m) + 1j * rng.normal(size=m)
    # E|FFT(sqrt(eig/m) z)_k|^2 = 2 C(0); the real part carries C(0).
    return np.real(np.fft.fft(np.sqrt(eig / m) * z))[:n]


def fbm(n: int, hurst: float, rng: np.random.Generator) -> np.ndarray:
    """Increments of fractional Brownian motion with unit variance."""
    k = np.arange(n, dtype=float)
    g = 0.5 * (np.abs(k - 1) ** (2 * hurst) - 2 * k ** (2 * hurst)
               + (k + 1) ** (2 * hurst))
    circ = np.concatenate([g, g[-2:0:-1]])
    eig = np.clip(np.real(np.fft.fft(circ)), 0.0, None)
    m = len(circ)
    z = rng.normal(size=m) + 1j * rng.normal(size=m)
    return np.real(np.fft.fft(np.sqrt(eig / m) * z))[:n]


def omega_covariance(n: int, lambda2: float, integral_scale: int = INTEGRAL_SCALE) -> np.ndarray:
    lags = np.arange(n, dtype=float)
    cov = lambda2 * np.log(integral_scale / (lags + 1.0))
    return np.where(lags < integral_scale, np.maximum(cov, 0.0), 0.0)


def mrw(n: int, lambda2: float, rng: np.random.Generator,
        integral_scale: int = INTEGRAL_SCALE, sigma: float = 1.0) -> np.ndarray:
    """Multifractal random walk increments."""
    eps = rng.normal(0.0, sigma, n)
    if lambda2 <= 0:
        return eps
    cov = omega_covariance(n, lambda2, integral_scale)
    omega = _circulant_gaussian(cov, rng) - cov[0] / 2.0
    return eps * np.exp(omega)


def check_embedding(n: int = 20_000, lambda2: float = 0.10, reps: int = 200,
                    lags=(0, 10, 100, 1000), seed: int = 0) -> dict:
    """Empirical / target autocovariance of omega at a few lags; all near 1."""
    cov = omega_covariance(n, lambda2)
    rng = np.random.default_rng(seed)
    acc = np.zeros(len(lags))
    for _ in range(reps):
        w = _circulant_gaussian(cov, rng)
        acc += [np.mean(w[: n - k] * w[k:]) for k in lags]
    return {k: float(v / reps / cov[k]) for k, v in zip(lags, acc)}
