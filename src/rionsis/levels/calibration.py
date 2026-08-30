"""Independent calibration of the single-ion power ``A1``.

The exact integer staircase fit (``rionsis.levels.staircase``) needs the
mean power increment of a single stored ion, ``A1``, as an *input*. This
module estimates ``A1`` from the same background-subtracted power traces
used everywhere else, but without ever running the staircase fit itself:
it fits a Poisson(ion count)-times-Gaussian(bin noise) mixture model
directly to the raw power distribution in each time bin.

Because the mixture model's ``mode='exp'`` fit also estimates a decay
rate for the mean ion occupancy, this calibration produces a *second*,
fully independent half-life estimate as a by-product -- useful as a
cross-check against the primary staircase + truncated-geometric-MLE
result in ``rionsis.halflife``, but not a substitute for it (see
``A1Calibration.halflife_ms_byproduct``).

A single-ion power A1 obtained here only applies to the exact spectral
pipeline (STFT window, bandwidth, noise normalization) it was calibrated
on; recompute it whenever those choices change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.special import gammaln

#: Default grid of fixed A1 values for the profile-likelihood scan.
DEFAULT_PROFILE_GRID: npt.NDArray[np.float64] = np.array(
    [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.8, 2.2]
)


def _log_sum_exp(a: npt.NDArray[np.float64], axis: int) -> npt.NDArray[np.float64]:
    mx = np.max(a, axis=axis, keepdims=True)
    return (mx + np.log(np.sum(np.exp(a - mx), axis=axis, keepdims=True))).squeeze(axis)


def _fit_variance_fixed_point(
    r_n: npt.NDArray[np.float64],
    s_n: npt.NDArray[np.float64],
    n_squared: npt.NDArray[np.float64],
    a: float,
    c: float,
    n_fixed_point_iter: int = 8,
) -> tuple[float, float]:
    """Minimize ``sum_N [S_N/u_N + R_N*log(u_N)]`` with ``u_N = a + c*N^2``."""
    for _ in range(n_fixed_point_iter):
        u = np.maximum(a + c * n_squared, 1e-12)
        w = 1.0 / (u * u)
        w_r, w_s = w * r_n, w * s_n
        matrix = np.array(
            [
                [np.sum(w_r), np.sum(w_r * n_squared)],
                [np.sum(w_r * n_squared), np.sum(w_r * n_squared * n_squared)],
            ]
        )
        rhs = np.array([np.sum(w_s), np.sum(w_s * n_squared)])
        try:
            sol = np.linalg.solve(matrix, rhs)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(sol).all() or sol[0] <= 0:
            break
        a, c = float(sol[0]), max(float(sol[1]), 0.0)
    return max(a, 1e-12), max(c, 0.0)


def _poisson_exp_decay_mle(n_t: npt.NDArray[np.float64], t_s: npt.NDArray[np.float64], n_files: int) -> tuple[float, float]:
    """Poisson MLE of ``mu_t = mu0 * exp(-lam*t)`` from expected ion counts ``n_t``."""
    total = float(np.sum(n_t))
    if total <= 0:
        return 1e-9, 0.0
    target = float(np.sum(n_t * t_s) / total)

    def imbalance(lam: float) -> float:
        w = np.exp(-lam * t_s)
        return float(np.sum(t_s * w) / np.sum(w)) - target

    lo, hi = 1e-4, 5.0e3
    if imbalance(lo) < 0:
        lam = lo
    elif imbalance(hi) > 0:
        lam = hi
    else:
        lam = float(brentq(imbalance, lo, hi, xtol=1e-9, rtol=1e-13, maxiter=300))
    mu0 = total / (n_files * float(np.sum(np.exp(-lam * t_s))))
    return max(mu0, 1e-9), lam


@dataclass(frozen=True)
class MixtureFitResult:
    """Result of one Poisson-Gaussian mixture EM fit (see ``fit_power_mixture``)."""

    a1: float
    b: float
    s0: float
    k: float
    mu: npt.NDArray[np.float64]
    mu0: float
    lam: float
    nll: float
    n_iter: int
    mode: str
    halflife_ms: float


def fit_power_mixture(
    x: npt.ArrayLike,
    t_s: npt.ArrayLike,
    *,
    n_max: int = 22,
    mode: str = "exp",
    a1_fixed: float | None = None,
    init: dict | None = None,
    n_iter: int = 800,
    tol: float = 1e-11,
) -> MixtureFitResult:
    """EM fit of a Poisson(ion count) x Gaussian(bin noise) power mixture.

    Parameters
    ----------
    x : array_like, shape (n_time_bins, n_files)
        Background-subtracted power, transposed so time bins are rows.
    t_s : array_like, shape (n_time_bins,)
        Time axis in seconds.
    n_max : int, default 22
        Largest ion occupancy state considered.
    mode : {"exp", "free"}
        ``"exp"``: ``mu_t = mu0*exp(-lam*t)``, a two-parameter decay law.
        ``"free"``: ``mu_t`` is free in every time bin; used only as a
        robustness check and is substantially less well identified.
    a1_fixed : float, optional
        If given, hold ``A1`` fixed at this value (for profile-likelihood
        scans) instead of fitting it.
    init : dict, optional
        Initial guesses for ``A1``, ``b``, ``s0``, ``k``.
    n_iter : int, default 800
        Maximum EM iterations.
    tol : float, default 1e-11
        Relative convergence tolerance on the negative log-likelihood.

    Returns
    -------
    MixtureFitResult

    Raises
    ------
    ValueError
        If ``x`` is not a finite 2D array with at least 3 files, or
        ``t_s`` does not match ``x``'s time axis.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or x.shape[1] < 3 or np.any(~np.isfinite(x)):
        raise ValueError("x must be a finite 2D array (time bins x files).")
    n_time, n_files = x.shape
    t_s = np.asarray(t_s, dtype=float)
    if t_s.shape != (n_time,):
        raise ValueError("t_s must be a time axis of length x.shape[0].")

    n_grid = np.arange(int(n_max) + 1, dtype=float)
    n_squared = n_grid * n_grid
    log_fact = gammaln(n_grid + 1.0)
    m_t = x.mean(axis=1)
    init = dict(init or {})
    a1 = float(init.get("A1", max(1e-3, 0.25 * (np.max(m_t) - np.min(m_t)))))
    if a1_fixed is not None:
        a1 = float(a1_fixed)
    b = float(init.get("b", 0.0))
    s0 = float(init.get("s0", max(1e-3, float(np.std(x[m_t <= np.quantile(m_t, 0.3)])))))
    k = float(init.get("k", 0.3))
    a, c = s0 * s0, k * k
    mu = np.clip((m_t - b) / max(a1, 1e-6), 1e-9, float(n_max))
    mu0, lam = float(mu[0]), 0.0
    previous, nll, used = np.inf, np.nan, 0
    for used in range(1, int(n_iter) + 1):
        s2 = np.maximum(a + c * n_squared, 1e-12)
        log_w = np.log(np.maximum(mu, 1e-12))[:, None] * n_grid - mu[:, None] - log_fact[None, :]
        log_w -= _log_sum_exp(log_w, 1)[:, None]
        z = x[:, :, None] - (b + a1 * n_grid)
        ll = log_w[:, None, :] - 0.5 * z * z / s2 - 0.5 * np.log(2.0 * np.pi * s2)
        total = _log_sum_exp(ll, 2)
        nll = float(-total.sum())
        r = np.exp(ll - total[:, :, None])
        n_t = np.einsum("tik,k->t", r, n_grid)
        if mode == "exp":
            mu0, lam = _poisson_exp_decay_mle(n_t, t_s, n_files)
            mu = np.clip(mu0 * np.exp(-lam * t_s), 1e-9, float(n_max))
        else:
            mu = np.clip(n_t / n_files, 1e-9, float(n_max))
        w = r / s2
        s_w = w.sum()
        s_wn = np.einsum("tik,k->", w, n_grid)
        s_wnn = np.einsum("tik,k->", w, n_squared)
        wx = w * x[:, :, None]
        s_wx = wx.sum()
        s_wnx = np.einsum("tik,k->", wx, n_grid)
        if a1_fixed is None:
            det = s_w * s_wnn - s_wn * s_wn
            if abs(det) > 1e-12:
                b = (s_wx * s_wnn - s_wn * s_wnx) / det
                a1 = max((s_w * s_wnx - s_wn * s_wx) / det, 1e-4)
        else:
            b = (s_wx - a1 * s_wn) / s_w
        residual = x[:, :, None] - (b + a1 * n_grid)
        a, c = _fit_variance_fixed_point(r.sum(axis=(0, 1)), (r * residual * residual).sum(axis=(0, 1)), n_squared, a, c)
        if abs(previous - nll) < tol * max(1.0, abs(nll)):
            break
        previous = nll
    halflife_ms = float(np.log(2) / lam * 1000.0) if lam > 0 else np.nan
    return MixtureFitResult(
        a1=float(a1), b=float(b), s0=float(np.sqrt(a)), k=float(np.sqrt(c)),
        mu=mu.copy(), mu0=float(mu0), lam=float(lam), nll=float(nll),
        n_iter=int(used), mode=str(mode), halflife_ms=halflife_ms,
    )


def moment_a1(y: npt.ArrayLike) -> tuple[float, float, float, float, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Method-of-moments estimate of ``A1`` from ``Var(power) = c0 + c1*m + c2*m^2``.

    Parameters
    ----------
    y : array_like, shape (n_files, n_time_bins)
        Background-subtracted power traces.

    Returns
    -------
    a1_moment, c0, c1, c2, mean_per_bin, variance_per_bin
        ``a1_moment = c1 / (1 + c2)``.
    """
    y = np.asarray(y, dtype=float)
    m, v = y.mean(axis=0), y.var(axis=0, ddof=1)
    design = np.vstack([np.ones_like(m), m, m * m]).T
    c0, c1, c2 = np.linalg.lstsq(design, v, rcond=None)[0]
    return float(c1 / (1.0 + c2)), float(c0), float(c1), float(c2), m, v


def _simulate_mixture_bootstrap(
    a1: float, mu0: float, lam: float, s0: float, k: float, b: float,
    n_files: int, t_s: npt.NDArray[np.float64], rng: np.random.Generator,
) -> npt.NDArray[np.float64]:
    """Poisson injection + independent exponential decay + iid Gaussian noise.

    A deliberately simpler generative model than
    ``rionsis.simulation.decay_simulation.simulate_decay_traces``: no STFT
    window response and no empirical noise library. It exists only to
    parametric-bootstrap the bias and uncertainty of the mixture EM
    estimator itself, not to validate the full analysis pipeline.
    """
    n0 = rng.poisson(mu0, int(n_files))
    out = np.empty((int(n_files), len(t_s)), dtype=float)
    for i, count in enumerate(n0):
        lifetimes = rng.exponential(1.0 / lam, int(count)) if lam > 0 else np.full(int(count), np.inf)
        occupancy = (lifetimes[:, None] > t_s[None, :]).sum(axis=0).astype(float)
        out[i] = b + a1 * occupancy + np.sqrt(s0**2 + (k * occupancy) ** 2) * rng.standard_normal(len(t_s))
    return out


@dataclass(frozen=True)
class A1Calibration:
    """Independent single-ion power calibration, without a staircase fit.

    Attributes
    ----------
    a1 : float
        Bias-corrected calibrated single-ion power. This is the value to
        feed into ``rionsis.levels.staircase``.
    sd : float
        Parametric-bootstrap standard deviation of ``a1``.
    ci68 : (float, float)
        68% parametric-bootstrap interval for ``a1``.
    mc_bias : float
        Fractional bias of the raw EM fit relative to the bootstrap
        median, i.e. ``a1 = fit_exp.a1 / (1 + mc_bias)``.
    a1_moment : float
        Method-of-moments estimate (diagnostic only).
    fit_exp : MixtureFitResult
        Full mixture EM fit with ``mode="exp"`` (the fit ``a1`` is derived
        from).
    fit_free : MixtureFitResult
        Robustness check with ``mode="free"``; systematically biased low,
        used only as a lower bound.
    profile_grid : ndarray
        Fixed-``A1`` values used for the profile-likelihood scan.
    profile_nll : ndarray
        Negative log-likelihood at each ``profile_grid`` point.
    halflife_ms_byproduct : float
        Half-life implied by ``fit_exp``'s decay rate. An independent
        cross-check estimate, not the primary half-life measurement (see
        ``rionsis.halflife``).
    n_files : int
        Number of files used for the calibration.
    """

    a1: float
    sd: float
    ci68: tuple[float, float]
    mc_bias: float
    a1_moment: float
    fit_exp: MixtureFitResult
    fit_free: MixtureFitResult
    profile_grid: npt.NDArray[np.float64]
    profile_nll: npt.NDArray[np.float64]
    halflife_ms_byproduct: float
    n_files: int
    external_a1: float | None = None
    external_a1_within_ci68: bool | None = field(default=None)


def calibrate_a1(
    y_files_e: npt.ArrayLike,
    dt_e: float,
    *,
    n_max: int = 22,
    profile_grid: npt.ArrayLike = DEFAULT_PROFILE_GRID,
    mc_reps: int = 24,
    em_iter: int = 800,
    file_idx: npt.ArrayLike | None = None,
    seed: int | np.random.Generator = 20260828,
    external_a1: float | None = None,
    sigma_hint: float | None = None,
) -> A1Calibration:
    """Calibrate the single-ion power ``A1`` from raw power traces.

    Parameters
    ----------
    y_files_e : array_like, shape (n_files, n_time_bins)
        Background-subtracted, noise-normalized power traces (e.g.
        ``GroundNoiseNormalization.y_files``).
    dt_e : float
        Time-bin width in seconds.
    n_max : int, default 22
        Largest ion occupancy state considered by the mixture fit.
    profile_grid : array_like, default DEFAULT_PROFILE_GRID
        Fixed ``A1`` values for the profile-likelihood scan.
    mc_reps : int, default 24
        Number of parametric-bootstrap replicates used to bias-correct
        and quantify the uncertainty of ``A1``. Time bins within a file
        are correlated, so the likelihood-curvature interval would be too
        optimistic; the bootstrap is used instead.
    em_iter : int, default 800
        Maximum EM iterations for the primary fit (profile and bootstrap
        refits use ``max(300, em_iter // 2)``).
    file_idx : array_like of int, optional
        Subset of files to use. Defaults to all files.
    seed : int or numpy.random.Generator, default 20260828
        Seed (or generator) for the parametric bootstrap. Explicit by
        design: no hidden global RNG state.
    external_a1 : float, optional
        A previously used/fixed A1 value to cross-check against the 68%
        calibration band (purely diagnostic; does not affect the fit).
    sigma_hint : float, optional
        Initial guess for the bin-noise floor ``s0`` (e.g. the reference
        noise level from ``normalize_ground_noise``). If omitted,
        ``fit_power_mixture`` falls back to a data-driven guess from the
        low-occupancy tail of ``y_files_e``.

    Returns
    -------
    A1Calibration
    """
    y_files_e = np.asarray(y_files_e, dtype=float)
    idx = np.arange(len(y_files_e)) if file_idx is None else np.asarray(file_idx, dtype=int)
    y = y_files_e[idx]
    t_s = np.arange(y.shape[1], dtype=float) * float(dt_e)
    x = y.T

    a1_moment, _c0, _c1, _c2, _m, _v = moment_a1(y)
    init = {"A1": float(np.clip(a1_moment, 0.2, 3.0)), "k": 0.3}
    if sigma_hint is not None:
        init["s0"] = float(sigma_hint)

    fit_exp = fit_power_mixture(x, t_s, n_max=n_max, mode="exp", init=init, n_iter=em_iter)
    fit_free = fit_power_mixture(x, t_s, n_max=n_max, mode="free", init=init, n_iter=em_iter)

    profile_grid = np.asarray(profile_grid, dtype=float)
    profile_nll = np.array(
        [
            fit_power_mixture(
                x, t_s, n_max=int(max(14, 3.0 * np.max(_m) / value + 5)),
                mode="exp", a1_fixed=float(value), init=init, n_iter=max(300, em_iter // 2),
            ).nll
            for value in profile_grid
        ],
        dtype=float,
    )

    rng = np.random.default_rng(seed)
    mc_values = np.array(
        [
            fit_power_mixture(
                _simulate_mixture_bootstrap(
                    fit_exp.a1, fit_exp.mu0, fit_exp.lam, fit_exp.s0, fit_exp.k, fit_exp.b, len(y), t_s, rng
                ).T,
                t_s, n_max=n_max, mode="exp", init=init, n_iter=max(300, em_iter // 2),
            ).a1
            for _ in range(int(mc_reps))
        ],
        dtype=float,
    )
    mc_bias = float(np.median(mc_values) / fit_exp.a1 - 1.0)
    a1_calibrated = float(fit_exp.a1 / (1.0 + mc_bias))
    sd = float(np.std(mc_values, ddof=1))
    ci68 = (
        float(np.percentile(mc_values, 16) / (1.0 + mc_bias)),
        float(np.percentile(mc_values, 84) / (1.0 + mc_bias)),
    )

    within_ci68 = None
    if external_a1 is not None:
        within_ci68 = bool(ci68[0] <= float(external_a1) <= ci68[1])

    return A1Calibration(
        a1=a1_calibrated, sd=sd, ci68=ci68, mc_bias=mc_bias, a1_moment=a1_moment,
        fit_exp=fit_exp, fit_free=fit_free, profile_grid=profile_grid, profile_nll=profile_nll,
        halflife_ms_byproduct=fit_exp.halflife_ms, n_files=int(len(y)),
        external_a1=None if external_a1 is None else float(external_a1),
        external_a1_within_ci68=within_ci68,
    )
