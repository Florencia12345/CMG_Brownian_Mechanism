import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, special, stats, integrate
import math
import time

"""
Dwell time fitting with the probability distribution, 
optimisted with MLE 
"""

def Q_reg(t, T_FNA, N_steps):
    """Q(t) = (t/T_FNA)^(N-1) / (1 + (t/T_FNA)^(N-1))"""
    T = np.clip(T_FNA, 1e-10, None)
    x = (t / T) ** max(1, N_steps - 1)
    return x / (1.0 + x)

def P_Nnt(t, f_FNA, T_FNA, N_steps):
    """
    P_Nnt(t) = (f_FNA^N / (T_FNA * (N-1)!)) * (t*N/T_FNA)^(N-1) * exp(-t*N/T_FNA)
    """
    N = N_steps
    t = np.asarray(t, dtype=float)
    T = np.clip(T_FNA, 1e-10, None)
    x = (t * N) / T
    out = np.zeros_like(t)
    mask = x > 1e-300
    if not np.any(mask):
        return out
    f = max(f_FNA, 1e-300)
    # log(coef) = N*ln(f_FNA) - ln(T_FNA) - ln((N-1)!)
    # log_coef = N * math.log(f) - math.log(T) - math.lgamma(N)
    log_coef = math.log(f * N) - math.log(T) - math.lgamma(N)

    log_term = (N - 1) * np.log(x[mask]) - x[mask]
    out[mask] = np.exp(log_coef + log_term)
    return np.clip(out, 0, 1e300)

# ----------------------------
# pdf distribution and components
def _paper_components(t, weights, T_FNA, T_SNA, T_VSNA, N_steps):
    """
    Return individual components of the paper PDF for visualization:
    g (gamma P_Nnt), Q, s1 (SNA), s2 (VSNA), p_llp, and the full pdf = g + Q*(s1 + s2 + p_llp).
    """
    a = np.asarray(weights, dtype=float)
    a = np.maximum(a, 0.0)
    s = np.sum(a)
    if s <= 0:
        s = 1.0
    f = a / s
    f_FNA, f_SNA, f_VSNA, f_LLP = f

    t = np.asarray(t, dtype=float)
    g1 = P_Nnt(t, f_FNA, T_FNA, N_steps)
    g = g1
    Q = Q_reg(t, T_FNA, N_steps)
    sigma_step = max(0.05 * T_FNA, 0.1)
    step = 0.5 * (1.0 + np.tanh((t - T_FNA) / sigma_step))
    s1 = step * (f_SNA / T_SNA) * np.exp(-np.maximum(t - T_FNA, 0) / T_SNA)
    s2 = step * (f_VSNA / T_VSNA) * np.exp(-np.maximum(t - T_FNA, 0) / T_VSNA)
    sigma_llp = max(0.15 * T_FNA, 0.2)
    ramp_llp = 0.5 * (1.0 + np.tanh((t - 0.5 * T_FNA) / sigma_llp))
    p_llp = ramp_llp * f_LLP * math.sqrt(1.0 + T_FNA) / (2.0 * (1.0 + t) ** 1.5)
    pdf = g + Q * (s1 + s2 + p_llp)
    pdf = np.maximum(pdf, 1e-300)
    return g, Q, s1, s2, p_llp, pdf


def paper_pdf(t, weights, T_FNA, T_SNA, T_VSNA, N_steps):
    """
    Full probability function: P(t) = P_Nnt(t) + Q(t) * ( SNA + VSNA + LLP ).
    weights: length-4 array; constraint f_FNA + f_SNA + f_VSNA + f_LLP = 1 via normalization: f = weights / sum(weights).
    """
    _, _, _, _, _, pdf = _paper_components(t, weights, T_FNA, T_SNA, T_VSNA, N_steps)
    return pdf

def paper_pdf_gamma(t, weights, T_FNA, T_SNA, T_VSNA, N_steps):
    """
    Full probability function: P(t) = P_Nnt(t) + Q(t) * ( SNA + VSNA + LLP ).
    weights: length-4 array; constraint f_FNA + f_SNA + f_VSNA + f_LLP = 1 via normalization: f = weights / sum(weights).
    """
    a = np.asarray(weights, dtype=float)
    a = np.maximum(a, 0.0)
    s = np.sum(a)
    if s <= 0:
        s = 1.0
    f = a / s
    f_FNA, f_SNA, f_VSNA, f_LLP = f

    t = np.asarray(t, dtype=float)
    # P_Nnt: gamma term
    g1 = P_Nnt(t, f_FNA, T_FNA, N_steps)
    # g2 = P_Nnt(t, f_SNA, T_SNA, N_steps)
    
    g = g1 # sum of two gamma terms
    # Q(t)
    Q = Q_reg(t, T_FNA, N_steps)
    # SNA / VSNA: shifted exponentials. Use a SMOOTH step at T_FNA to avoid a discontinuity (jump) in the PDF.
    # Smooth step: ramp in over ~0.2*T_FNA so the curve is continuous on log-log plots.
    sigma_step = max(0.05 * T_FNA, 0.1)
    step = 0.5 * (1.0 + np.tanh((t - T_FNA) / sigma_step))
    s1 = step * (f_SNA / T_SNA) * np.exp(-np.maximum(t - T_FNA, 0) / T_SNA)
    s2 = step * (f_VSNA / T_VSNA) * np.exp(-np.maximum(t - T_FNA, 0) / T_VSNA)
    # LLP: f_LLP * sqrt(1 + T_FNA) / (2 * (1 + t/1s)^(3/2)), 1s = 1
    # Ramp LLP in at t ~ 0.5*T_FNA so it doesn't create a short-time hump then dip (Q*LLP peaks
    # at small t then falls while gamma is still small; ramping LLP avoids that bend).
    sigma_llp = max(0.15 * T_FNA, 0.2)
    ramp_llp = 0.5 * (1.0 + np.tanh((t - 0.5 * T_FNA) / sigma_llp))
    p_llp = ramp_llp * f_LLP * math.sqrt(1.0 + T_FNA) / (2.0 * (1.0 + t) ** 1.5)
    # P(t) = P_Nnt + Q * (SNA + VSNA + LLP)
    pdf = g + Q * (s1 + s2 + p_llp)
    pdf = np.maximum(pdf, 1e-300)
    return pdf

'''
# ----------------------------
# Fit  model to the FIRST-PASSAGE DENSITY (not raw dwell times). 
#  this part is wrong, fit to raw dwell times 
# ----------------------------
'''
def empirical_first_passage_density(dwell_times, n_bins=80, log_bins=False):
    """Compute empirical first-passage density (histogram, density=True). Returns bin_centers, density_vals.
    If log_bins=True, use log-spaced bin edges so the tail (long times) gets more bins and the fit can capture it."""
    dt = np.asarray(dwell_times, dtype=float)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        return np.array([]), np.array([])
    t_min, t_max = np.min(dt) * 0.99, np.max(dt) * 1.01
    t_min = max(t_min, 1e-9)
    if log_bins:
        edges = np.logspace(np.log10(t_min), np.log10(t_max), n_bins + 1)
        counts, _ = np.histogram(dt, bins=edges, density=True)
        centers = np.sqrt(edges[:-1] * edges[1:])  # geometric center for log scale
    else:
        edges = np.linspace(t_min, t_max, n_bins + 1)
        counts, _ = np.histogram(dt, bins=edges, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, counts


def _residual_density(params_vec, t_centers, density_emp, N_steps, log_scale=True, normalize_model=True, tail_weight=False):
    """
    Residuals for fit to first-passage density.
    params_vec: [f_FNA, f_SNA, f_VSNA, f_LLP] (normalized to sum 1 in paper_pdf), then T_FNA, T_SNA, T_VSNA.
    If tail_weight=True, residuals are scaled so that long-time (tail) points contribute more to the loss.
    """
    weights = np.maximum(params_vec[:4], 0.0)
    T_FNA, T_SNA, T_VSNA = params_vec[4], params_vec[5], params_vec[6]
    if (T_FNA <= 0) or (T_SNA <= 0) or (T_VSNA <= 0):
        return np.full_like(density_emp, 1e10)
    model_vals = paper_pdf(t_centers, weights, T_FNA, T_SNA, T_VSNA, N_steps)
    if normalize_model and len(t_centers) > 1:
        area = np.trapz(model_vals, t_centers)
        if area > 1e-300:
            model_vals = model_vals / area
    eps = 1e-12
    if log_scale:
        log_emp = np.log(np.maximum(density_emp, eps))
        log_model = np.log(np.maximum(model_vals, eps))
        r = log_emp - log_model
    else:
        r = density_emp - model_vals
    if tail_weight and len(t_centers) > 1:
        t_med = np.median(t_centers)
        w = 1.0 + np.log1p(np.maximum(t_centers, 1e-9) / max(t_med, 1e-9))
        r = r * np.sqrt(w)
    return r

def fit_paper_model_to_first_passage_density(
    dwell_times, N_steps=3, n_bins=80, initial=None, bounds=None, method='L-BFGS-B', verbose=False, n_restarts=200,
    log_bins=True, tail_weight=True
):
    
    dwell_times = np.asarray(dwell_times, dtype=float)
    dwell_times = dwell_times[np.isfinite(dwell_times) & (dwell_times > 0)]
    n_raw = dwell_times.size
    if n_raw == 0:
        raise ValueError("No valid dwell times provided")

    t_centers, density_emp = empirical_first_passage_density(dwell_times, n_bins=n_bins, log_bins=log_bins)
    if t_centers.size == 0:
        raise ValueError("Could not compute empirical first-passage density")
    n = len(t_centers)  # number of points in density curve

    # Single default init and bounds (no data-driven inits)
    default_init = np.array([0.25, 0.25, 0.25, 0.25, 5.0, 5.0, 10.0])
    default_bounds = [(1e-6, None)] * 4 + [(1e-3, 1e4)] * 3
    bounds = bounds if bounds is not None else default_bounds
    inits = [initial] if initial is not None else [default_init]

    def loss(params_vec):
        r = _residual_density(params_vec, t_centers, density_emp, N_steps, log_scale=True, normalize_model=True, tail_weight=tail_weight)
        return np.sum(r ** 2)

    best_rss = np.inf
    best_res = None
    t0 = time.time()
    for x0 in inits[:n_restarts]:
        try:
            res = optimize.minimize(
                loss,
                x0=x0,
                method=method,
                bounds=bounds,
                options={'maxiter': 15000, 'ftol': 1e-9, 'gtol': 1e-6},
            )
            if res.fun < best_rss and np.isfinite(res.fun):
                best_rss = res.fun
                best_res = res
        except Exception:
            continue
    t1 = time.time()

    if best_res is None:
        raise RuntimeError("All density-fit restarts failed")
    res = best_res
    rss = res.fun
    success = res.success
    # Log-likelihood under Gaussian errors: sigma^2 = RSS/n (MLE), logL = -n/2*(1 + log(2*pi*RSS/n))
    sigma2 = max(rss / n, 1e-300)
    logL = -0.5 * n * (1.0 + math.log(2 * math.pi * sigma2))
    k_params = 7
    bic = k_params * math.log(n) - 2.0 * logL

    w = np.maximum(res.x[:4], 0.0)
    f_vals = w / (np.sum(w) + 1e-300)
    param_dict = {
        'f_FNA': float(f_vals[0]),
        'f_SNA': float(f_vals[1]),
        'f_VSNA': float(f_vals[2]),
        'f_LLP': float(f_vals[3]),
        'T_FNA': float(res.x[4]),
        'T_SNA': float(res.x[5]),
        'T_VSNA': float(res.x[6]),
    }
    msg = res.message if hasattr(res, 'message') else getattr(res, 'message', '')
    out = {
        'params': param_dict,
        'logL': float(logL),
        'rss': float(rss),
        'bic': float(bic),
        'success': success,
        'opt': res,
        'time_sec': t1 - t0,
        'convergence_message': str(msg),
        'n_bins': n,
        'n_dwell_times': n_raw,
    }
    if verbose:
        print("density fit success:", success, "RSS:", rss, "logL:", logL, "BIC:", bic, "time(s):", t1 - t0)
    return out


# ----------------------------
# MLE fit: maximize LL = sum ln P(t_i) over raw dwell times {t_i}
def _paper_pdf_norm(params_vec, N_steps, t_max=1e5):
    """Normalizing constant Z = int_0^t_max paper_pdf(t) dt for use in MLE."""
    weights = np.maximum(params_vec[:4], 0.0)
    T_FNA, T_SNA, T_VSNA = params_vec[4], params_vec[5], params_vec[6]
    if (T_FNA <= 0) or (T_SNA <= 0) or (T_VSNA <= 0):
        return np.nan

    def _integrand(t):
        return paper_pdf(np.atleast_1d(float(t)), weights, T_FNA, T_SNA, T_VSNA, N_steps).item()

    z, _ = integrate.quad(_integrand, 1e-9, min(t_max, 1e5), limit=300)
    return z

# this is a better loss function 
def neg_log_likelihood(params_vec, dwell_times, N_steps, head_weight = False, head_sigma_log=15):
    """
    Negative log-likelihood for MLE: -LL = -sum ln P(t_i).
    P(t) = paper_pdf(t) / Z so that P integrates to 1; then LL = sum ln P(t_i) = sum ln paper_pdf(t_i) - n*ln(Z).
    """
    dwell_times = np.asarray(dwell_times, dtype=float)
    dwell_times = dwell_times[np.isfinite(dwell_times) & (dwell_times > 0)]
    # print(mean(well_times))
    n = dwell_times.size
    if n == 0:
        return 0.0
    weights = np.maximum(params_vec[:4], 0.0)
    T_FNA, T_SNA, T_VSNA = params_vec[4], params_vec[5], params_vec[6]
    if (T_FNA <= 0) or (T_SNA <= 0) or (T_VSNA <= 0):
        return 1e300
    p = paper_pdf(dwell_times, weights, T_FNA, T_SNA, T_VSNA, N_steps)
    t_max = max(10.0 * np.max(dwell_times), 1e3)
    z = _paper_pdf_norm(params_vec, N_steps, t_max=t_max)
    if z <= 0 or not np.isfinite(z):
        return 1e300
    if head_weight:
        # Weight by Gaussian in log(t) centered at empirical "bump" (log-median as proxy for mode)
        log_t = np.log(np.maximum(dwell_times, 1e-9))
        log_t_center = np.median(log_t)
        print('t_center', np.exp(log_t_center))
        w = np.exp(-0.5 * ((log_t - log_t_center) / head_sigma_log) ** 2)
        w = np.maximum(w, 0.1)
        nll = - np.sum(w * np.log(np.maximum(p, 1e-300))) + np.sum(w) * np.log(z)
        print('nll', nll, 'z', z)
    else:
        # nll = -np.sum(np.log(np.maximum(p, 1e-300))) + n * np.log(z)
        nll = -np.sum(np.log(np.maximum(p, 1e-300)))
        # print('nll', nll, 'z', z)
    return float(nll)

# ----------------------------BFGSL-BFGS-B
# MLE fitting 
def fit_paper_model_MLE(dwell_times, N_steps=3, initial=None, bounds=None, method='L-BFGS-B', verbose=False, n_restarts=20):
    """
    Fit the paper PDF to experimentally collected dwell times {t_i} by maximizing the log-likelihood
    LL = sum_i ln P(t_i). Minimizes negative log-likelihood via scipy.optimize.minimize. Returns best-fit params, logL, BIC.
    """
    dwell_times = np.asarray(dwell_times, dtype=float)
    dwell_times = dwell_times[np.isfinite(dwell_times) & (dwell_times > 0)]
    n = dwell_times.size
    if n == 0:
        raise ValueError("No dwell times provided")

    default_init = np.array([0.25, 0.25, 0.25, 0.25, 5.0, 5.0, 10.0])
    default_bounds = [(1e-6, None)] * 4 + [(1e-3, 1e4)] * 3
    bounds = bounds if bounds is not None else default_bounds
    inits = [initial] if initial is not None else [default_init]

    best_nll = np.inf
    best_res = None
    t0 = time.time()
    for x0 in inits[:n_restarts]:
        try:
            res = optimize.minimize(
                neg_log_likelihood,
                x0=x0,
                            args=(dwell_times, N_steps),
                method=method,
                bounds=bounds,
                options={'maxiter': 20000, 'ftol': 1e-9, 'gtol': 1e-6},
            )
            if res.fun < best_nll and np.isfinite(res.fun):
                best_nll = res.fun
                best_res = res
        except Exception:
            continue
    t1 = time.time()

    if best_res is None:
        raise RuntimeError("All MLE restarts failed")
    res = best_res
    rss = res.fun
    success = res.success
    nll = res.fun
    logL = -nll
    k_params = 7
    bic = k_params * np.log(n) - 2.0 * logL

    w = np.maximum(res.x[:4], 0.0)
    f_vals = w / (np.sum(w) + 1e-300)
    param_dict = {
        'f_FNA': float(f_vals[0]),
        'f_SNA': float(f_vals[1]),
        'f_VSNA': float(f_vals[2]),
        'f_LLP': float(f_vals[3]),
        'T_FNA': float(res.x[4]),
        'T_SNA': float(res.x[5]),
        'T_VSNA': float(res.x[6]),
    }
    msg = res.message if hasattr(res, 'message') else getattr(res, 'message', '')
    out = {
        'params': param_dict,
        'rss': float(rss),
        'logL': float(logL),
        'nll': float(nll),
        'bic': float(bic),
        'success': success,
        'opt': res,
        'time_sec': t1 - t0,
        'convergence_message': str(msg),
        'n_dwell_times': n,
    }
    if verbose:
        print("MLE opt success:", success, "logL:", logL, "BIC:", bic, "time(s):", t1 - t0)
    return out


def fit_two_dwelltime_files_MLE(
    path1: str,
    path2: str,
    N_steps: int = 20,
    skiprows: int = 1,
    save_plot: bool = True,
    output_dir: str ='/Users/vivian/Desktop/Undergrad Study/Part C Project/code/fit_dwelltime_output',
    verbose: bool = True,
):
    """
    Convenience helper: load two dwell_times text files, combine the dwell times,
    and run the MLE fit on the pooled data.

    Parameters
    ----------
    path1, path2 : str
        Paths to text files containing dwell times (one per line, optional header).
    N_steps : int
        Shape parameter for the gamma-like first-passage term in the paper PDF.
    skiprows : int
        Number of header rows to skip when reading with numpy.loadtxt.
    save_plot : bool
        If True, save an overlay plot of empirical PDF + fitted model.
    output_dir : str or None
        Directory to save the plot. If None, uses a 'fit_dwelltime_output' folder
        next to this script.
    verbose : bool
        Print basic fit information.

    Returns
    -------
    fit_out : dict
        The dictionary returned by fit_paper_model_MLE for the pooled dwell times.
    """
    import os

    def _load(path: str) -> np.ndarray:
        try:
            arr = np.loadtxt(path, skiprows=skiprows)
        except Exception:
            arr = np.loadtxt(path)
        arr = np.asarray(arr, dtype=float).ravel()
        arr = arr[np.isfinite(arr) & (arr > 0)]
        return arr

    dt1 = _load(path1)
    dt2 = _load(path2)
    pooled = np.concatenate([dt1, dt2])
    if pooled.size == 0:
        raise ValueError("No valid dwell times after combining the two files.")

    fit_out = fit_paper_model_MLE(pooled, N_steps=N_steps, verbose=verbose)

    if save_plot:
        if output_dir is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(script_dir, "fit_dwelltime_output")
        os.makedirs(output_dir, exist_ok=True)

        base1 = os.path.splitext(os.path.basename(path1))[0]
        base2 = os.path.splitext(os.path.basename(path2))[0]
        tag = f"{base1}__{base2}"
        plot_path = os.path.join(output_dir, f"mle_fit_pooled_{tag}.png")
        plot_fit(pooled, fit_out, N_steps=N_steps, show=False, save_path=plot_path)
        if verbose:
            print("Saved pooled MLE fit overlay:", plot_path)

    return fit_out
# ----------------------------
# Bootstrap uncertainties (for density fit: resample dwell times, recompute density, refit)
def bootstrap_density_fit(dwell_times, N_steps=3, n_bins=80, n_boot=100, random_seed=0, **fit_kwargs):
    """Bootstrap: resample dwell times, compute empirical first-passage density, fit paper model. Report std per parameter."""
    rng = np.random.default_rng(random_seed)
    dwell_times = np.asarray(dwell_times)
    dwell_times = dwell_times[np.isfinite(dwell_times) & (dwell_times > 0)]
    boots = []
    for i in range(n_boot):
        sample = rng.choice(dwell_times, size=len(dwell_times), replace=True)
        try:
            out = fit_paper_model_to_first_passage_density(
                sample, N_steps=N_steps, n_bins=n_bins, verbose=False, **fit_kwargs
            )
            boots.append(out['params'])
        except Exception as e:
            print("bootstrap density fit failed:", e)
            continue
    if not boots:
        raise RuntimeError("No successful bootstrap fits; cannot compute stats")
    keys = list(boots[0].keys())
    med = {k: np.median([b[k] for b in boots]) for k in keys}
    mean = {k: np.mean([b[k] for b in boots]) for k in keys}
    std = {k: np.std([b[k] for b in boots]) for k in keys}
    return {'boots': boots, 'median': med, 'mean': mean, 'std': std, 'n_success': len(boots)}

# ----------------------------
# Plot helpers
def plot_fit(dwell_times, fit_out, N_steps=3, nbins=200, show=True, save_path=None):
    """
    PDF overlay (log-y) and Survival overlay (log-y). Optionally save to save_path.
    """
    dt = np.asarray(dwell_times)
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if dt.size == 0:
        raise ValueError("No valid dwell times to plot")
    t_lo = max(1e-9, np.min(dt) * 0.9)
    t_hi = np.max(dt) * 1.1
    xs = np.logspace(np.log10(t_lo), np.log10(t_hi), 400)
    weights = np.array([fit_out['params']['f_FNA'], fit_out['params']['f_SNA'],
                        fit_out['params']['f_VSNA'], fit_out['params']['f_LLP']])
    # Compute components and total PDF for dashed component plots
    g_comp, Q_comp, s1_comp, s2_comp, p_llp_comp, pdf_raw = _paper_components(
        xs, weights, fit_out['params']['T_FNA'], fit_out['params']['T_SNA'],
        fit_out['params']['T_VSNA'], N_steps
    )
    area = np.trapz(pdf_raw, xs)
    if area > 1e-300:
        pdf_vals = pdf_raw / area
        g_comp = g_comp / area
        q_s1 = (Q_comp * s1_comp) / area
        q_s2 = (Q_comp * s2_comp) / area
        q_llp = (Q_comp * p_llp_comp) / area
    else:
        pdf_vals = pdf_raw
        q_s1 = Q_comp * s1_comp
        q_s2 = Q_comp * s2_comp
        q_llp = Q_comp * p_llp_comp
    edges = np.logspace(np.log10(t_lo), np.log10(t_hi), nbins + 1)
    counts, _ = np.histogram(dt, bins=edges, density=True)
    print('counts', counts)
    print('sum counts * bin width', np.sum(counts * np.diff(edges)))
    centers = np.sqrt(edges[:-1] * edges[1:])
    sorted_dt = np.sort(dt)
    n = len(sorted_dt)
    surv = 1.0 - np.arange(1, n + 1) / (n + 1.0)

    dx = np.diff(xs)
    trapz = (pdf_vals[:-1] + pdf_vals[1:]) * 0.5 * dx
    cdf_vals = np.concatenate([[0.0], np.cumsum(trapz)])
    total = cdf_vals[-1]
    if total > 1e-300:
        cdf_vals = cdf_vals / total
    S_model = 1.0 - cdf_vals

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Left: survival S(t), log-y
    ax1.scatter(sorted_dt, surv, label='empirical survival', s=30)
    ax1.plot(xs, S_model, label='model S(t)', lw=2, color='C1')
    ax1.set_yscale('log')
    ax1.set_xlabel('Passage time (s)')
    ax1.set_ylabel('Survival S(t)')
    ax1.legend()
    ax1.grid(True, which='both', ls='--', alpha=0.4)

    # Right: first-passage density (PDF), log-log
    ax2.scatter(centers, counts, label='empirical first-passage density', s=30)
    # ax2.plot(xs, pdf_vals, label='paper-model PDF', lw=2, color='C1')
    # ax2.plot(xs[:200], g_comp[:200], '--', lw=1.5, color='C2', label='g (gamma P_Nnt)')
    # ax2.plot(xs[100:350], q_s1[100:350], '--', lw=1.5, color='C3', label='Q·SNA')
    # ax2.plot(xs[150:350], q_s2[150:350], '--', lw=1.5, color='C4', label='Q·VSNA')

    # n_bin250 high_pause4 & 2
    ax2.plot(xs, pdf_vals, label='paper-model PDF', lw=2, color='C1')
    ax2.plot(xs[:50], g_comp[:50], '--', lw=1.5, color='C2', label='g (gamma P_Nnt)')
    ax2.plot(xs[100:250], q_s1[100:250], '--', lw=1.5, color='C3', label='Q·SNA')
    
    # n_bin250 high_pause_3 & High_Pausing
    # ax2.plot(xs, pdf_vals, label='paper-model PDF', lw=2, color='C1')
    # ax2.plot(xs[:150], g_comp[:150], '--', lw=1.5, color='C2', label='g (gamma P_Nnt)')
    # ax2.plot(xs[100:350], q_s1[100:350], '--', lw=1.5, color='C3', label='Q·SNA')

   # n_bin250 Balanced
    # ax2.plot(xs, pdf_vals, label='paper-model PDF', lw=2, color='C1')
    # ax2.plot(xs[0:150], g_comp[0:150], '--', lw=1.5, color='C2', label='g (gamma P_Nnt)')
    # ax2.plot(xs[150:400], q_s1[150:400], '--', lw=1.5, color='C3', label='Q·SNA')


    #     ax2.plot(xs, pdf_vals, label='paper-model PDF', lw=2, color='C1')
    # ax2.plot(xs[:50], g_comp[:50], '--', lw=1.5, color='C2', label='g (gamma P_Nnt)')
    # ax2.plot(xs[50:250], q_s1[50:250], '--', lw=1.5, color='C3', label='Q·SNA')
    ax2.plot(xs[100:350], q_s2[100:350], '--', lw=1.5, color='C4', label='Q·VSNA')
    # ax2.plot(xs[250:450], q_llp[250:450], '--', lw=1.5, color='C5', label='Q·LLP')
    ax2.set_yscale('log')
    ax2.set_xscale('log')
    ax2.set_xlabel('Passage time (s)')
    ax2.set_ylabel('First-passage density (PDF)')
    ax2.legend()
    ax2.grid(True, which='both', ls='--', alpha=0.4)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
    if show:
        plt.show()
    return fig


def write_summary_page(out_path, fit_out, bootstrap_std, N_steps, n_data, convergence_warning=None):
    """Write one-page text summary: fitted params, bootstrap std, logL, BIC, convergence."""
    lines = [
        "Dwell-time paper model – fit to first-passage density",
        "=" * 50,
        "",
        "Fit target: empirical first-passage density (histogram of dwell times), not raw dwell times.",
        "Data: n = {} dwell times -> binned density, then paper PDF fitted to that density curve.".format(n_data),
        "Model: paper PDF with N_steps = {} (gamma shape).".format(N_steps),
        "Fitted parameters (k=7): softmax weights f_FNA, f_SNA, f_VSNA, f_LLP; T_FNA, T_SNA, T_VSNA.",
        "",
        "Fitted parameter values:",
    ]
    for k, v in fit_out['params'].items():
        std_str = ""
        if bootstrap_std and k in bootstrap_std:
            std_str = "  (bootstrap std = {:.6g})".format(bootstrap_std[k])
        lines.append("  {} = {:.6g}{}".format(k, v, std_str))
    lines.extend([
        "",
        "Log-likelihood (logL) = {:.4f}".format(fit_out['logL']),
        "BIC = {:.4f}".format(fit_out['bic']),
    ])
    if fit_out.get('rss') is not None:
        lines.append("RSS (residual sum of squares vs density) = {:.6g}".format(fit_out['rss']))
    lines.extend([
        "Optimizer success: {}".format(fit_out['success']),
        "Convergence message: {}".format(fit_out.get('convergence_message', '')),
    ])
    if convergence_warning:
        lines.append("WARNING: {}".format(convergence_warning))
    lines.append("")
    with open(out_path, 'w') as f:
        f.write("\n".join(lines))

# ----------------------------
# Example usage and pipeline
# ----------------------------
if __name__ == "__main__":
    import os
    import glob

    script_dir = os.path.dirname(os.path.abspath(__file__))

    out_dir = os.path.join(script_dir, "fit_dwelltime_output")
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "mle_fit_summary.txt")
    # Determine which data paths have already been processed (to skip on reruns)
    processed_paths = set()
    if os.path.exists(summary_path):
        with open(summary_path, "r") as f:
            for line in f:
                if not line.strip() or line.startswith("nbin_dir"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 3:
                    processed_paths.add(parts[2])
    else:
        # Create file with header if it doesn't exist yet
        with open(summary_path, "w") as f:
            f.write(
                "nbin_dir\trun_name\tdata_path\tN_steps\t"
                "f_FNA\tf_SNA\tf_VSNA\tf_LLP\t"
                "T_FNA\tT_SNA\tT_VSNA\t"
                "logL\tnll\tBIC\tsuccess\topt_time_sec\n"
            )

    # Where simulation outputs live, e.g. sim_traces_nbin_200/run_High_Unwinding/dwell_times_all.txt
    # pattern = os.path.join(script_dir, "sim_traces_nbin_*", "run_*", "dwell_times_all.txt")
    # pattern = os.path.join(script_dir, "sim_traces_nbin*", "run_*", "dwell_times_all.txt")
    pattern = os.path.join(script_dir, "dwell_time_all", "dwell_times_all*.txt")

    dwell_files = sorted(glob.glob(pattern))

    if not dwell_files:
        raise RuntimeError(f"No dwell_times_all.txt files found under {pattern}")

    N_steps_main = 20

    for path in dwell_files:
        print("\n==============================")
        print("Loading dwell times from:", path)

        # Skip files that are already in the summary
        if path in processed_paths:
            print("  Already processed (found in mle_fit_summary.txt), skipping.")
            continue

        try:
            dwell_times = np.loadtxt(path, skiprows=1)
        except Exception as e:
            print("  Failed to load, skipping:", e)
            continue

        dwell_times = np.asarray(dwell_times, dtype=float).ravel()
        dwell_times = dwell_times[np.isfinite(dwell_times) & (dwell_times > 0)]
        if dwell_times.size == 0:
            print("  No valid dwell times, skipping.")
            continue

        # Parse nbin and run name from path
        run_dir = os.path.dirname(path)
        nbin_dir = os.path.basename(os.path.dirname(run_dir))  # e.g. sim_traces_nbin_200
        run_name = os.path.basename(run_dir)                   # e.g. run_High_Unwinding
        tag = f"{nbin_dir}_{run_name}"

        # Fit by MLE
        try:
            fit_out = fit_paper_model_MLE(dwell_times, N_steps=N_steps_main, verbose=True)
        except Exception as e:
            print("  MLE fit failed for", tag, ":", e)
            continue

        print("  Best-fit params:", fit_out["params"])
        print("  logL:", fit_out["logL"], "nll:", fit_out["nll"], "BIC:", fit_out["bic"])

        # Save overlay plot with identifying name
        plot_name = f"mle_fit_{tag}.png"
        plot_path = os.path.join(out_dir, plot_name)
        plot_fit(dwell_times, fit_out, N_steps=N_steps_main, show=False, save_path=plot_path)
        print("  Saved MLE fit overlay:", plot_path)

        # Append summary row
        p = fit_out["params"]
        with open(summary_path, "a") as f:
            f.write(
                f"{nbin_dir}\t{run_name}\t{path}\t{N_steps_main}\t"
                f"{p['f_FNA']:.6g}\t{p['f_SNA']:.6g}\t{p['f_VSNA']:.6g}\t{p['f_LLP']:.6g}\t"
                f"{p['T_FNA']:.6g}\t{p['T_SNA']:.6g}\t{p['T_VSNA']:.6g}\t"
                f"{fit_out['logL']:.6f}\t{fit_out['nll']:.6f}\t{fit_out['bic']:.6f}\t"
                f"{fit_out['success']}\t{fit_out['time_sec']:.4f}\n"
            )
    fit_out = fit_two_dwelltime_files_MLE('/Users/vivian/Desktop/Undergrad Study/Part C Project/code/dwell_time_all/dwell_times_all_250_Really_High_Pause.txt', '/Users/vivian/Desktop/Undergrad Study/Part C Project/code/dwell_time_all/dwell_times_all_250_High_Pausing.txt')
    # Append summary row
    p = fit_out["params"]
    with open(summary_path, "a") as f:
        f.write(
            f"{nbin_dir}\t{run_name}\t{path}\t{N_steps_main}\t"
            f"{p['f_FNA']:.6g}\t{p['f_SNA']:.6g}\t{p['f_VSNA']:.6g}\t{p['f_LLP']:.6g}\t"
            f"{p['T_FNA']:.6g}\t{p['T_SNA']:.6g}\t{p['T_VSNA']:.6g}\t"
            f"{fit_out['logL']:.6f}\t{fit_out['nll']:.6f}\t{fit_out['bic']:.6f}\t"
            f"{fit_out['success']}\t{fit_out['time_sec']:.4f}\n"
        )

    print("\nFinished MLE fits. Summary written to:", summary_path)