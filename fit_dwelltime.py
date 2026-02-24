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
    log_coef = N * math.log(f) - math.log(T) - math.lgamma(N)
    log_term = (N - 1) * np.log(x[mask]) - x[mask]
    out[mask] = np.exp(log_coef + log_term)
    return np.clip(out, 0, 1e300)

# ----------------------------
# pdf distribution 
def paper_pdf(t, weights, T_FNA, T_SNA, T_VSNA, N_steps):
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
    g2 = P_Nnt(t, f_SNA, T_SNA, N_steps)
    g = g1 + g2  # sum of two gamma terms
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
def neg_log_likelihood(params_vec, dwell_times, N_steps):
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
    nll = -np.sum(np.log(np.maximum(p, 1e-300))) + n * np.log(z)
    return float(nll)

# ----------------------------
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
    pdf_vals = paper_pdf(xs, weights, fit_out['params']['T_FNA'],
                         fit_out['params']['T_SNA'], fit_out['params']['T_VSNA'], N_steps)
    # Normalize model PDF over plot range so scale matches empirical density and survival is valid
    area = np.trapz(pdf_vals, xs)
    if area > 1e-300:
        pdf_vals = pdf_vals / area
    edges = np.logspace(np.log10(t_lo), np.log10(t_hi), nbins + 1)
    counts, _ = np.histogram(dt, bins=edges, density=True)
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
    ax2.plot(xs, pdf_vals, label='paper-model PDF', lw=2, color='C1')
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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # default_path = os.path.join(script_dir, "sim_traces", "run_Balanced", "dwell_times_all.txt")
    # default_path = os.path.join(script_dir, "sim_traces", "run_High_Unwinding", "dwell_times_all.txt")
    default_path = os.path.join(script_dir, "sim_traces", "run_High_Pausing", "dwell_times_all.txt")

    if os.path.isfile(default_path):
        dwell_times = np.loadtxt(default_path, skiprows=1)
        dwell_times = dwell_times[np.isfinite(dwell_times) & (dwell_times > 4)]
    else:
        rng = np.random.default_rng(1)
        g = rng.gamma(3.0, 0.5, size=300)
        e = 5.0 + rng.exponential(3.0, size=200)
        dwell_times = np.concatenate([g, e])
    dwell_times = np.asarray(dwell_times, dtype=float).ravel()
    dwell_times = dwell_times[np.isfinite(dwell_times) & (dwell_times > 0)]
    if dwell_times.size == 0:
        raise ValueError("No valid dwell times (check path or use synthetic data)")

    out_dir = os.path.join(script_dir, "fit_dwelltime_output")
    os.makedirs(out_dir, exist_ok=True)
    '''
    N_steps_main = 12
    n_bins = 80
    n_boot = 100

    # Fit paper model to the empirical first-passage density (not to raw dwell times)
    res = fit_paper_model_to_first_passage_density(
        dwell_times, N_steps=N_steps_main, n_bins=n_bins, verbose=True, n_restarts=200
    )
    print("Best-fit params:", res['params'])
    print("RSS:", res['rss'], "logL:", res['logL'], "BIC:", res['bic'])

    # Bootstrap: resample dwell times -> empirical density -> refit; report std per parameter
    bs = bootstrap_density_fit(dwell_times, N_steps=N_steps_main, n_bins=n_bins, n_boot=n_boot)
    bootstrap_std = bs['std']
    print("Bootstrap std:", bootstrap_std)

    # Save: PDF overlay, Survival overlay, one-page summary
    overlay_path = os.path.join(out_dir, "dwell_fit_PDF_and_survival_overlay.png")
    plot_fit(dwell_times, res, N_steps=N_steps_main, nbins=200, show=False, save_path=overlay_path)
    print("Saved overlay (PDF + Survival):", overlay_path)

    convergence_warning = None if res['success'] else "Optimizer did not report success."
    summary_path = os.path.join(out_dir, "dwell_fit_summary.txt")
    write_summary_page(summary_path, res, bootstrap_std, N_steps_main, dwell_times.size, convergence_warning)
    print("Saved summary:", summary_path)

    # Sensitivity analysis: N_steps = 2, 3, 4, 5; report how T_FNA changes (density fit)
    sensitivity = []
    print("\nSensitivity analysis (N_steps vs T_FNA), density fit:")
    print("-" * 40)
    for N in range(2, 30):
        try:
            r = fit_paper_model_to_first_passage_density(
                dwell_times, N_steps=N, n_bins=n_bins, verbose=False, n_restarts=200
            )
            T_FNA = r['params']['T_FNA']
            T_SNA = r['params']['T_SNA']
            T_VSNA = r['params']['T_VSNA']
            sensitivity.append((N, T_FNA, T_SNA, T_VSNA, r['logL'], r['bic']))
            print("  N_steps = {}  ->  T_FNA = {:.4f}   logL = {:.2f}   BIC = {:.2f}".format(N, T_FNA, r['logL'], r['bic']))
        except Exception as e:
            print("  N_steps = {}  ->  failed: {}".format(N, e))
    sens_path = os.path.join(out_dir, "sensitivity_N_steps.txt")
    with open(sens_path, 'w') as f:
        f.write("Sensitivity: fitted T_FNA vs N_steps\n")
        f.write("N_steps  T_FNA    T_SNA    T_VSNA    logL      BIC\n")
        for N, T_FNA, T_SNA, T_VSNA, logL, bic in sensitivity:
            f.write("{}        {:.6g}  {:.6g} {:.6g}  {:.4f}  {:.4f}\n".format(N, T_FNA, T_SNA, T_VSNA, logL, bic))
    print("Saved sensitivity table:", sens_path)

    plot_fit(dwell_times, res, N_steps=N_steps_main, show=True)
    '''

    fit_out = fit_paper_model_MLE(dwell_times, N_steps=5, verbose=True)
    # fit_out['logL'] is the maximized log-likelihood
    plot_fit(dwell_times, fit_out, N_steps=5, save_path='mle_fit.png')
    print("Saved MLE fit overlay:", 'mle_fit.png')