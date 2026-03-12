'''
This code contains the helper functions that are called in cmg_brownian_rachet_improved.py 
Functions:
    1. parse_file_like_original --> read and parse data files
    2. detect_noisy_tail_index --> detect noisy tails and crop them for experimental data processing
    3. denoiszation --> denoising function 
    4. plot_dwell_time_survival + plot_first_passage_density --> dwell_time_analysis function 
    5. I did try using Curzor to generate a few fitting methods, including biased random walk, gamma, etc. Doesn't reall work here. 
        Better fitting: fit_dwelltime.py 
'''

import os
import glob
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

try:
    from scipy.signal import medfilt, savgol_filter
    from scipy import stats as scipy_stats
    from scipy.optimize import minimize
    _HAS_SCIPY = True
    _HAS_OPTIMIZE = True
except Exception:
    _HAS_SCIPY = False
    _HAS_OPTIMIZE = False
    scipy_stats = None

# VIVIAN_DIR = Path("./sim_traces")
# PLOT_DIR = VIVIAN_DIR / "dwell_time_plots"
# PLOT_DIR.mkdir(parents=True, exist_ok=True)
VIVIAN_DIR = Path("/Users/vivian/Desktop/Undergrad Study/Part C Project/code/vivian/")
PLOT_DIR = VIVIAN_DIR / "dwell_time_plots"

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.figsize": (6.5, 4.0),
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "lines.linewidth": 1,
    }
)


def parse_file_like_original(path):
    """
    Parse the file
    - pad shorter rows to the max number of columns with 'nan'
    - detect header by attempting float(columns[0][0])
    Returns: headers (list of str), data_cols (list of lists of floats)
    """
    with open(path, "r") as f:
        # split each non-empty line on ANY amount of whitespace
        lines = [line.strip().split() for line in f if line.strip()]

    if not lines:
        return None, None  # empty file

    # find max number of columns
    max_len = max(len(line) for line in lines)

    # pad shorter lines with 'nan' to create rectangular structure
    for line in lines:
        while len(line) < max_len:
            line.append("nan")

    # transpose rows -> columns
    columns = list(zip(*lines))  # tuple per column

    # detect header: try converting first value to float
    try:
        float(columns[0][0])
        has_header = False
    except Exception:
        has_header = True

    if has_header:
        headers = [c[0] for c in columns]
        # convert remaining rows (skip header row)
        data_cols = []
        for c in columns:
            col_vals = []
            for v in c[1:]:
                try:
                    col_vals.append(float(v))
                except Exception:
                    col_vals.append(np.nan)
            data_cols.append(col_vals)
    else:
        headers = [f"col{i+1}" for i in range(len(columns))]
        data_cols = []
        for c in columns:
            col_vals = []
            for v in c:
                try:
                    col_vals.append(float(v))
                except Exception:
                    col_vals.append(np.nan)
            data_cols.append(col_vals)

    # At this point, data_cols is a list of lists (each same length)
    return headers, data_cols

# -----------------------------------------------------------
# Crop the noisey tail
def detect_noisy_tail_index(y,
                            deriv_window=20,
                            deriv_thr_mul=8.0,
                            deriv_min_consec=25,
                            median_window=30,
                            median_drop_frac=0.5,
                            median_min_consec=15,
                            margin=2):
    """
    Detect the index where a noisy / dropped tail begins in series y.
    Return an integer index 'cut' such that y[:cut] is the 'clean' part
    If no cut is detected, returns len(y) (no cropping).
    """
    y = np.array(y, dtype=float)
    n = len(y)
    if n < 2:
        return n

    # Interpolate NaNs for detection purposes only
    mask = np.isfinite(y)
    if not np.any(mask):
        return n
    if np.sum(mask) < n:
        idx = np.arange(n)
        y_interp = np.copy(y)
        y_interp[~mask] = np.interp(idx[~mask], idx[mask], y[mask])
    else:
        y_interp = y

    # ---------- 1) Derivative / spike detector ----------
    dy = np.abs(np.diff(y_interp, prepend=y_interp[0]))
    if deriv_window > 1:
        kernel = np.ones(deriv_window) / deriv_window
        dy_smooth = np.convolve(dy, kernel, mode="same")
    else:
        dy_smooth = dy

    base_end = max(int(n * 0.1), deriv_window * 2)
    base_vals = dy_smooth[:base_end] if base_end > 0 else dy_smooth
    base_med = np.median(base_vals)
    base_std = np.std(base_vals) if np.std(base_vals) > 0 else 1e-9
    deriv_thr = base_med + deriv_thr_mul * base_std

    above = dy_smooth > deriv_thr
    consec = 0
    cut_idx_deriv = None
    for i, val in enumerate(above):
        if val:
            consec += 1
            if consec >= deriv_min_consec:
                cut_idx_deriv = i - consec + 1
                break
        else:
            consec = 0

    # ---------- 2) Rolling median drop detector ----------
    if median_window < 3:
        median_window = 3
    half = median_window // 2
    med = np.full(n, np.nan)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        med[i] = np.median(y_interp[a:b])

    base_median_val = np.median(med[:base_end]) if base_end > 0 else np.median(med)
    median_thr_val = base_median_val * median_drop_frac

    below = med < median_thr_val
    consec = 0
    cut_idx_med = None
    for i, val in enumerate(below):
        if val:
            consec += 1
            if consec >= median_min_consec:
                cut_idx_med = i - consec + 1
                break
        else:
            consec = 0

    candidates = [idx for idx in (cut_idx_deriv, cut_idx_med) if idx is not None]
    if not candidates:
        return n

    cut = min(candidates)
    cut = max(0, cut - margin)
    if cut < 10:
        cut = 10 if cut > 0 else 0
    if cut >= n:
        return n
    return cut

# -----------------------------------------------------------
# denoise function
def _interpolate_nans_1d(arr):
    """Linear interpolate NaNs in a 1D array; returns new np.array of floats."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    mask = np.isfinite(arr)
    if mask.all():
        return arr
    if mask.sum() == 0:
        return arr  # all NaN, nothing to do
    idx = np.arange(n)
    arr_interp = arr.copy()
    arr_interp[~mask] = np.interp(idx[~mask], idx[mask], arr[mask])
    return arr_interp

def _moving_median(arr, kernel):
    """Simple moving median fallback (O(n*kernel)). Kernel must be odd."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if kernel <= 1:
        return arr.copy()
    half = kernel // 2
    out = np.full(n, np.nan)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = np.median(arr[a:b])
    return out

def _moving_mean(arr, kernel):
    """Simple moving average fallback."""
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if kernel <= 1:
        return arr.copy()
    half = kernel // 2
    out = np.full(n, np.nan)
    for i in range(n):
        a = max(0, i - half)
        b = min(n, i + half + 1)
        out[i] = np.mean(arr[a:b])
    return out

def denoiszation(x, y, headers, filepath, 
            interpolate_nans=True,
            median_kernel=3,
            savgol_window=150,
            savgol_polyorder=2,
            aggressive_guard=True,
            aggressive_thresh_mult=19.0,
            save = True, 
            show = False
            ):
    """
    Denoise a 1D signal y (optionally using x for length checks).
    
    Steps:
      1. Interpolate NaNs for filtering stability
      2. Small median filter to remove isolated spikes (kernel=3).
      3. Gentel Savitzky-Golay smoothing to preserve peaks/phase.
      5. Aggressive-guard: if smoothing changed many points massively, re-run with milder smoothing.
    """
    info = {
        "used_scipy": _HAS_SCIPY,
        "median_kernel": median_kernel,
        "savgol_window": savgol_window,
        "savgol_polyorder": savgol_polyorder,
        "aggressive_guard_triggered": False,
        "notes": ""
    }

    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return y.copy(), info

    # Keep track of original NaN mask and restore later
    nan_mask = ~np.isfinite(y)

    # 1) Interpolate NaNs for filter stability
    if interpolate_nans and np.any(nan_mask):
        y_work = _interpolate_nans_1d(y)
    else:
        # if we choose not to interpolate, replace NaNs with 0 to avoid errors (but keep mask)
        y_work = y.copy()
        if np.any(nan_mask):
            y_work[~np.isfinite(y_work)] = 0.0

    # 2) Median (spike) removal
    if median_kernel < 1:
        median_kernel = 1
    if median_kernel % 2 == 0:
        median_kernel += 1  # force odd
    if median_kernel > 1:
        if _HAS_SCIPY:
            try:
                y_med = medfilt(y_work, kernel_size=median_kernel)
            except Exception:
                y_med = _moving_median(y_work, median_kernel)
        else:
            y_med = _moving_median(y_work, median_kernel)
    else:
        y_med = y_work.copy()

    # 3) Gentle smoothing: Savitzky-Golay (preserves peaks), fallback to moving mean
    # Ensure window is odd and less than or equal to length
    if savgol_window % 2 == 0:
        savgol_window += 1
    if savgol_window >= n:
        # choose a smaller odd window
        w = n - 1 if (n - 1) % 2 == 1 else n - 2
        savgol_window = max(3, w)
    try:
        if _HAS_SCIPY:
            y_smooth = savgol_filter(y_med, window_length=savgol_window, polyorder=savgol_polyorder, mode="interp")
        else:
            y_smooth = _moving_mean(y_med, savgol_window)
    except Exception:
        # in case of any runtime issue, fallback to moving mean
        y_smooth = _moving_mean(y_med, savgol_window)

    # 4) Restore NaNs where original had NaNs
    y_denoised = y_smooth.copy()
    if np.any(nan_mask):
        y_denoised[nan_mask] = np.nan

    # 5) Aggressive guard: detect if smoothing changed the signal too much (avoid over-smoothing)
    info["aggressive_guard_triggered"] = False
    if aggressive_guard:
        diff = np.abs(np.nan_to_num(y_denoised) - np.nan_to_num(y))
        # robust statistics: median absolute deviation
        med_diff = np.median(diff)
        mad = np.median(np.abs(diff - med_diff)) if np.isfinite(med_diff) else 0.0
        # fallback small scale
        if mad <= 0:
            mad = np.std(diff) if np.std(diff) > 0 else 1e-9

        # count points that changed by more than aggressive_thresh_mult * mad
        thresh = aggressive_thresh_mult * mad
        changed_count = np.sum(diff > thresh)
        # If more than 1% of points (and at least 5) changed massively, trigger milder smoothing
        if changed_count > max(5, 0.01 * n):
            info["aggressive_guard_triggered"] = True
            info["notes"] += f"aggressive_guard: {changed_count} points > {aggressive_thresh_mult}*MAD. "
            # milder approach: smaller median + smaller smoothing window
            small_med = max(1, (median_kernel // 2) | 1)
            small_win = max(3, (savgol_window // 2) | 1)
            if small_med > 1:
                if _HAS_SCIPY:
                    try:
                        y_med2 = medfilt(y_work, kernel_size=small_med)
                    except Exception:
                        y_med2 = _moving_median(y_work, small_med)
                else:
                    y_med2 = _moving_median(y_work, small_med)
            else:
                y_med2 = y_work.copy()
            if _HAS_SCIPY:
                try:
                    y_smooth2 = savgol_filter(y_med2, window_length=small_win, polyorder=min(2, savgol_polyorder), mode="interp")
                except Exception:
                    y_smooth2 = _moving_mean(y_med2, small_win)
            else:
                y_smooth2 = _moving_mean(y_med2, small_win)
            y_denoised = y_smooth2.copy()
            if np.any(nan_mask):
                y_denoised[nan_mask] = np.nan
            info["notes"] += f"used milder median={small_med}, window={small_win}."

    # final info
    info["used_scipy"] = _HAS_SCIPY
    info["final_median_kernel"] = median_kernel if not info["aggressive_guard_triggered"] else small_med
    info["final_savgol_window"] = savgol_window if not info["aggressive_guard_triggered"] else small_win
    info["final_savgol_polyorder"] = savgol_polyorder if not info["aggressive_guard_triggered"] else min(2, savgol_polyorder)


    plt.figure(figsize=(12, 6))
    plt.plot(x, y_denoised, label=headers[2] if headers and len(headers) > 2 else "col3")
    plt.xlabel(headers[0] if headers else "X")
    plt.ylabel("Values")
    plt.title(f"{Path(filepath).name} — first column as X")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True)
    plt.tight_layout()

    if save:
        outpath = PLOT_DIR / f"{Path(filepath).stem}.png"
        plt.savefig(outpath, dpi=200)
        print(f"Saved plot: {outpath}")
    if show:
        plt.show()
    else:
        plt.close()

    return y_denoised, info, x


# Dwell time analysis-------------------------------
# def _interp_cross_time(x0, y0, x1, y1, level):
#     """Linear interpolation to estimate x where y crosses 'level' between (x0,y0) and (x1,y1)."""
#     if not np.isfinite(y0) or not np.isfinite(y1) or y1 == y0:
#         return x0
#     t = (level - y0) / (y1 - y0)
#     return x0 + t * (x1 - x0)

# ---------------------------------
def _interp_cross_time(x0, y0, x1, y1, level):
    """Linear interpolation time where segment crosses y=level. Returns None if impossible."""
    if not (np.isfinite(y0) and np.isfinite(y1)):
        return None
    dy = y1 - y0
    if dy == 0:
        return None
    t = (level - y0) / dy
    return x0 + t * (x1 - x0)

def plot_dwell_time_survival(dwell_times, filepath, show=False):
    """
    Plot and save dwell-time survival (y-axis log, x-axis linear).
    In log(S) vs t the exponential S(t)=exp(-t/τ) is a line with slope = -1/τ,
    so the fitted slope gives dwell time τ = -1/slope. 
    Feb.20
    Added Overlay fit, 95% CI band,
    and annotate τ ± SE(τ), R², and fit quality on the graph.
    """
    dwell_times = np.asarray(dwell_times, dtype=float)
    dwell_times = dwell_times[np.isfinite(dwell_times) & (dwell_times > 0)]
    if dwell_times.size == 0:
        print("No dwell times to plot survival.")
        return

    sorted_dt = np.sort(dwell_times)
    n = len(sorted_dt)
    survival = 1.0 - np.arange(1, n + 1) / (n + 1.0)
    # print('survival ------------',survival)
    # Avoid log(0)
    log_survival = np.log(np.clip(survival, 1e-15, 1.0))

    # Linear fit: log(S) = slope * t + intercept  =>  slope = -λ = -1/τ, so τ = -1/slope (dwell time)
    if _HAS_SCIPY and scipy_stats is not None and n >= 3:
        try:
            res = scipy_stats.linregress(sorted_dt, log_survival)
            slope, intercept, r_val, p_val, se_slope = res.slope, res.intercept, res.rvalue, res.pvalue, res.stderr
        except Exception:
            slope, intercept, r_val, se_slope = _linear_fit_log_survival(sorted_dt, log_survival, n)
    else:
        slope, intercept, r_val, se_slope = _linear_fit_log_survival(sorted_dt, log_survival, n)

    # Dwell time τ = -1/slope (slope is negative); SE(τ) = |SE(slope)|/slope^2 by delta method
    rate_fit = -slope if slope != 0 else 1.0 / float(np.mean(dwell_times))
    tau_fit = 1.0 / rate_fit if rate_fit > 0 else np.nan
    se_tau = (float(np.abs(se_slope)) / (slope ** 2)) if slope != 0 and np.isfinite(se_slope) else np.nan
    if not np.isfinite(se_tau) or se_tau <= 0:
        se_tau = tau_fit / np.sqrt(n) if n > 0 and np.isfinite(tau_fit) else np.nan
    r_squared = (r_val ** 2) if np.isfinite(r_val) else np.nan

    # Exponential curve and 95% CI band
    t_fit = np.linspace(sorted_dt.min(), sorted_dt.max(), 200)
    survival_exp = np.exp(slope * t_fit + intercept)
    lambda_lo = lambda_hi = rate_fit
    s_lo = s_hi = survival_exp
    if np.isfinite(se_slope) and se_slope > 0 and slope != 0:
        lambda_lo = -(slope + 1.96 * se_slope)
        lambda_hi = -(slope - 1.96 * se_slope)
        if lambda_lo > 0 and lambda_hi > 0:
            s_lo = np.exp(-lambda_lo * t_fit + intercept)
            s_hi = np.exp(-lambda_hi * t_fit + intercept)

    ks_stat, ks_pval = np.nan, np.nan
    if _HAS_SCIPY and scipy_stats is not None and n >= 5:
        try:
            mean_dt = float(np.mean(dwell_times))
            ks_stat, ks_pval = scipy_stats.kstest(dwell_times, scipy_stats.expon(scale=mean_dt).cdf)
        except Exception:
            pass
    mean_dt = float(np.mean(dwell_times))
    median_dt = float(np.median(dwell_times))
    std_dt = float(np.std(dwell_times))

    dot_color = "#5B6498"
    cmap = plt.get_cmap("tab20c")
    col_model = cmap(0)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(
        sorted_dt,
        survival,
        "o",
        markersize=4,
        markerfacecolor="none",
        markeredgecolor=dot_color,
        linewidth=0.6,
        label="Empirical survival",
    )
    if np.isfinite(se_slope) and se_slope > 0 and slope != 0 and lambda_lo > 0 and lambda_hi > 0:
        ax.fill_between(t_fit, np.minimum(s_lo, s_hi), np.maximum(s_lo, s_hi), color=col_model, alpha=0.12, label="95% CI (fit)")
    ax.semilogy(t_fit, survival_exp, "-", color=col_model, linewidth=1.2, label=f"Exponential  (τ = {tau_fit:.3g})")
    ax.set_xlabel("Dwell time Δt", fontsize=10)
    ax.set_ylabel("Survival 1 − CDF", fontsize=10)
    ax.set_title("Dwell-time survival (log y)", fontsize=11)
    leg = ax.legend(loc="upper right", fontsize=8, frameon=True)
    if leg is not None:
        leg.get_frame().set_edgecolor("0.7")
        leg.get_frame().set_linewidth(0.7)

    # On-graph annotation: dwell time τ ± SE, R², fit quality
    text_lines = [
        f"τ (dwell time) = {tau_fit:.4g} ± {se_tau:.4g}",
        f"R² = {r_squared:.4f}" if np.isfinite(r_squared) else "R² = —",
    ]
    if np.isfinite(ks_pval):
        text_lines.append(f"KS p = {ks_pval:.3f}")
    text = "\n".join(text_lines)
    ax.text(0.05, 0.35, text, transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8), family="monospace")

    ax.grid(True, which="major", alpha=0.2, linestyle="--", linewidth=0.4, color="0.7")
    ax.grid(True, which="minor", alpha=0.15, linestyle=":", linewidth=0.3, color="0.8")
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_edgecolor("black")
    ax.set_ylim(bottom=min(0.5 / (n + 1), float(survival.min()) * 0.5))
    fig.tight_layout()

    try:
        out_dir = PLOT_DIR
    except NameError:
        out_dir = Path(filepath).parent / "plots"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outpath = out_dir / f"{Path(filepath).stem}_survival.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved survival plot: {outpath}")
    print("  Dwell time τ (from slope of log S vs t): τ = {:.4g} ± {:.4g}".format(tau_fit, se_tau))
    print("  Fit: R² = {:.4f}".format(r_squared) if np.isfinite(r_squared) else "  Fit: R² = —")
    print("  Statistics: n = {}, mean = {:.4g}, median = {:.4g}, std = {:.4g}".format(n, mean_dt, median_dt, std_dt))
    if np.isfinite(ks_pval):
        print("  KS test (exponential): statistic = {:.4f}, p-value = {:.4f}".format(ks_stat, ks_pval))
    if show:
        plt.show()


def _linear_fit_log_survival(x, y, n):
    """Fallback linear fit for log(survival) vs t; returns slope, intercept, r, se_slope."""
    if n < 2:
        return 0.0, 0.0, np.nan, np.nan
    coef = np.polyfit(x, y, 1)
    slope, intercept = coef[0], coef[1]
    y_pred = slope * x + intercept
    res = y - y_pred
    ss_res = np.sum(res ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r = np.sqrt(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    mse = ss_res / (n - 2) if n > 2 else 0.0
    s_xx = np.sum((x - np.mean(x)) ** 2)
    se_slope = np.sqrt(mse / s_xx) if s_xx > 0 else np.nan
    return slope, intercept, r, se_slope

def plot_first_passage_density(dwell_times, filepath, show=False, n_bins=80):
    """
    Plot and save first-passage time probability density (log-log).
    Overlays empirical histogram and MLE fits (Exponential, Gamma, Weibull, Inverse-Gaussian, Lognormal).
    n_bins : int
        Number of bins for histogram
    """
    dwell_times = np.asarray(dwell_times, dtype=float)
    dwell_times = dwell_times[np.isfinite(dwell_times) & (dwell_times > 0)]

    if dwell_times.size == 0:
        print("No dwell times to plot probability density.")
        return

    t_min = max(dwell_times.min(), 1e-10)
    t_max = dwell_times.max()
    hist, edges = np.histogram(dwell_times, bins=n_bins, density=True)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    mask = hist > 0
    hist = hist[mask]
    bin_centers = bin_centers[mask]

    if hist.size == 0:
        print("No non-empty bins for first-passage density plot.")
        return

    dot_color = "#5B6498"
    cmap = plt.get_cmap("tab20c")
    model_colors = [cmap(0), cmap(6), cmap(10), cmap(5)]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(
        bin_centers,
        hist,
        "o",
        markersize=3.5,
        markerfacecolor="none",
        markeredgecolor=dot_color,
        linewidth=0.6,
        label="Empirical density",
    )

    # Overlay MLE fits (same models as plot_dwell_time_fits), styled with tab20c
    models = _fit_dwell_time_models(dwell_times, n_bootstrap=0)
    t_plot = np.linspace(t_min, t_max, 300)
    for i, m in enumerate(models):
        c = model_colors[i % len(model_colors)]
        try:
            pdf_vals = m["dist"].pdf(t_plot)
            pdf_vals = np.clip(pdf_vals, 1e-20, None)
            ax.loglog(t_plot[:200], pdf_vals[:200], "-", color=c, linewidth=1.2, label=m["name"])
        except Exception:
            pass

    ax.set_xlabel("Passage time Δt", fontsize=10)
    ax.set_ylabel("First-passage density p(Δt)", fontsize=10)
    ax.set_title("First-passage time probability density", fontsize=11)
    leg = ax.legend(loc="upper right", fontsize=8, frameon=True)
    if leg is not None:
        leg.get_frame().set_edgecolor("0.7")
        leg.get_frame().set_linewidth(0.7)
    ax.grid(True, which="major", alpha=0.2, linestyle="--", linewidth=0.4, color="0.7")
    ax.grid(True, which="minor", alpha=0.15, linestyle=":", linewidth=0.3, color="0.8")
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_edgecolor("black")
    fig.tight_layout()

    try:
        out_dir = PLOT_DIR
    except NameError:
        out_dir = Path(filepath).parent / "plots"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outpath = out_dir / f"{Path(filepath).stem}_fp_density.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved first-passage density plot (with model fits): {outpath}")
    if show:
        plt.show()


# --- Dwell-time distribution fits (MLE on raw times): Wald/Inv-Gauss, Lognormal, Gamma, Weibull, optional mixtures ---
# these are baisic fittings for the dwell time, but obviously they are not correct. 
# Refer to the more sophisticated fitting - fit_dwelltime.py 
# can ignore here. Trial functions generated by Cursor and doesn't work
def _wald_pdf(t, mu, lam):
    """Wald / Inverse-Gaussian PDF: first-passage time for drift-diffusion. t, mu, lam > 0."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    mask = t > 0
    if not np.any(mask):
        return out
    tt = np.clip(t[mask], 1e-300, None)
    out[mask] = np.sqrt(lam / (2 * np.pi * tt**3)) * np.exp(-lam * (tt - mu)**2 / (2 * mu**2 * tt))
    return out


def _wald_logpdf(t, mu, lam):
    """Log of Wald PDF. Returns -inf where t <= 0."""
    t = np.asarray(t, dtype=float)
    out = np.full_like(t, -np.inf)
    mask = (t > 0) & (mu > 0) & (lam > 0)
    if not np.any(mask):
        return out
    tt = t[mask]
    out[mask] = 0.5 * (np.log(lam) - np.log(2 * np.pi) - 3 * np.log(tt)) - lam * (tt - mu)**2 / (2 * mu**2 * tt)
    return out


class _WaldDist:
    """Thin wrapper so Wald has .pdf and .sf for plotting. Parameterization: mu (mean), lam (shape)."""
    def __init__(self, mu, lam):
        self.mu = float(mu)
        self.lam = float(lam)
    def pdf(self, x):
        return _wald_pdf(x, self.mu, self.lam)
    def cdf(self, x):
        """Wald CDF: F(x; μ, λ) = Φ(√(λ/x)(x/μ - 1)) + exp(2λ/μ) Φ(-√(λ/x)(x/μ + 1))."""
        x = np.asarray(x, dtype=float)
        out = np.where(x <= 0, 0.0, np.nan)
        if scipy_stats is None:
            return out
        mask = x > 0
        if not np.any(mask):
            return np.where(x <= 0, 0.0, 0.0)
        xt = x[mask]
        rt = np.sqrt(self.lam / np.clip(xt, 1e-300, None))
        z1 = rt * (xt / self.mu - 1)
        z2 = -rt * (xt / self.mu + 1)
        out[mask] = scipy_stats.norm.cdf(z1) + np.exp(2 * self.lam / self.mu) * scipy_stats.norm.cdf(z2)
        out[mask] = np.clip(out[mask], 0, 1)
        return out
    def sf(self, x):
        return 1.0 - self.cdf(x)


def _fit_dwell_time_models(dwell_times, n_bootstrap=1000, random_state=None, fit_mixtures=False):
    """
    Fit candidate models to dwell times using maximum likelihood on raw times (no histogram).

    Models:
    A. Wald / Inverse-Gaussian (drift-diffusion first-passage): params (μ, λ); also report (v, D, L) with L=1.
    B. 1D first-passage PDE: same as Wald for standard drift-diffusion (absorbing boundary).
    C. Lognormal  D. Gamma  E. Weibull
    F. Optional: two-component mixtures (Gamma+Exp, Wald+Exp) if fit_mixtures=True.

    For each model: MLE via scipy.optimize.minimize on negative log-likelihood; log-likelihood, AIC, BIC;
    95%% bootstrap CIs (non-parametric, default 1000 resamples).

    Returns
    -------
    list of dicts: name, dist (.pdf/.sf), params, n_params, loglike, aic, bic, bootstrap_ci (if n_bootstrap>0).
    """
    if not _HAS_SCIPY or scipy_stats is None or not _HAS_OPTIMIZE:
        return []
    rng = np.random.default_rng(random_state)
    data = np.asarray(dwell_times, dtype=float)
    data = data[np.isfinite(data) & (data > 0)]
    n = len(data)
    if n < 2:
        return []
    models = []

    def _bootstrap_ci(boot_params, names, pct_lo=2.5, pct_hi=97.5):
        if not boot_params:
            return {}
        arr = np.array(boot_params)
        return {names[i]: (np.percentile(arr[:, i], pct_lo), np.percentile(arr[:, i], pct_hi)) for i in range(len(names))}

    # --- A. Wald / Inverse-Gaussian (first-passage: drift v, diffusion D, threshold L). μ = L/v, λ = L²/(2D). ---
    def _wald_negll(theta):
        mu, lam = theta[0], theta[1]
        if mu <= 0 or lam <= 0:
            return 1e20
        return -np.sum(_wald_logpdf(data, mu, lam))

    try:
        mu0 = np.mean(data-10)
        lam0 = max(mu0**3 / (np.var(data) + 1e-12), 0.1)
        res = minimize(_wald_negll, [mu0, lam0], method="L-BFGS-B", bounds=[(1e-6, None), (1e-6, None)])
        if res.success:
            mu, lam = res.x[0], res.x[1]
            loglike = -res.fun
            L = 1.0
            v = L / mu
            D = L**2 / (2 * lam)
            dist = _WaldDist(mu, lam)
            k = 2
            aic = 2 * k - 2 * loglike
            bic = k * np.log(n) - 2 * loglike
            ci = {}
            if n_bootstrap > 0:
                boot = []
                for _ in range(n_bootstrap):
                    b = rng.choice(data, size=n, replace=True)
                    r = minimize(lambda th: -np.sum(_wald_logpdf(b, th[0], th[1])), [np.mean(b), max(np.mean(b)**3 / (np.var(b) + 1e-12), 0.1)], method="L-BFGS-B", bounds=[(1e-6, None), (1e-6, None)])
                    if r.success:
                        boot.append(r.x)
                ci = _bootstrap_ci(boot, ["mu", "lam"])
            models.append({
                "name": "Wald (Inv-Gauss)",
                "dist": dist,
                "params": {"mu": mu, "lam": lam, "v (drift)": v, "D (diffusion)": D, "L (threshold)": L},
                "n_params": 2,
                "loglike": loglike,
                "aic": aic,
                "bic": bic,
                "bootstrap_ci": ci if n_bootstrap > 0 else None,
            })
    except Exception:
        pass

    # --- B. 1D first-passage PDE: same as Wald (standard drift-diffusion with absorbing boundary) ---
    # Already fitted as Wald; we add a note in the first model. No separate fit.

    '''
    # --- Exponential: 1 param (scale). MLE = 1/mean. ---
    try:
        scale = float(np.mean(data))
        if scale <= 0:
            raise ValueError("mean <= 0")
        dist = scipy_stats.expon(scale=scale)
        loglike = np.sum(dist.logpdf(data))
        k = 1
        aic = 2 * k - 2 * loglike
        bic = k * np.log(n) - 2 * loglike
        ci = {}
        if n_bootstrap > 0:
            boot = [np.mean(rng.choice(data, size=n, replace=True)) for _ in range(n_bootstrap)]
            ci = {"scale": (np.percentile(boot, 2.5), np.percentile(boot, 97.5))}
        models.append({
            "name": "Exponential",
            "dist": dist,
            "params": {"scale": scale},
            "n_params": 1,
            "loglike": loglike,
            "aic": aic,
            "bic": bic,
            "bootstrap_ci": ci if n_bootstrap > 0 else None,
        })
    except Exception:
        pass
    
    
    '''
    # --- C. Lognormal ---
    def _lognorm_negll(theta):
        s, scale = theta[0], theta[1]
        if s <= 0 or scale <= 0:
            return 1e20
        return -np.sum(scipy_stats.lognorm(s, scale=scale).logpdf(data))

    try:
        s0 = np.sqrt(np.log(1 + np.var(data) / (np.mean(data)**2 + 1e-12)))
        scale0 = np.mean(data)
        res = minimize(_lognorm_negll, [s0, scale0], method="L-BFGS-B", bounds=[(1e-6, None), (1e-6, None)])
        if res.success:
            s, scale = res.x[0], res.x[1]
            dist = scipy_stats.lognorm(s, scale=scale)
            loglike = -res.fun
            k = 2
            aic = 2 * k - 2 * loglike
            bic = k * np.log(n) - 2 * loglike
            ci = {}
            if n_bootstrap > 0:
                boot = []
                for _ in range(n_bootstrap):
                    b = rng.choice(data, size=n, replace=True)
                    sb = np.sqrt(np.log(1 + np.var(b) / (np.mean(b)**2 + 1e-12)))
                    r = minimize(lambda th: -np.sum(scipy_stats.lognorm(th[0], scale=th[1]).logpdf(b)), [sb, np.mean(b)], method="L-BFGS-B", bounds=[(1e-6, None), (1e-6, None)])
                    if r.success:
                        boot.append(r.x)
                ci = _bootstrap_ci(boot, ["s", "scale"])
            models.append({
                "name": "Lognormal",
                "dist": dist,
                "params": {"s": s, "scale": scale},
                "n_params": 2,
                "loglike": loglike,
                "aic": aic,
                "bic": bic,
                "bootstrap_ci": ci if n_bootstrap > 0 else None,
            })
    except Exception:
        pass

    # --- D. Gamma ---
    def _gamma_negll(theta):
        shape, scale = theta[0], theta[1]
        if shape <= 0 or scale <= 0:
            return 1e20
        return -np.sum(scipy_stats.gamma(shape, scale=scale).logpdf(data))

    try:
        mean, var = np.mean(data), np.var(data) + 1e-12
        shape0 = mean**2 / var
        scale0 = var / mean
        res = minimize(_gamma_negll, [shape0, scale0], method="L-BFGS-B", bounds=[(1e-6, None), (1e-6, None)])
        if res.success:
            shape, scale = res.x[0], res.x[1]
            dist = scipy_stats.gamma(shape, scale=scale)
            loglike = -res.fun
            k = 2
            aic = 2 * k - 2 * loglike
            bic = k * np.log(n) - 2 * loglike
            ci = {}
            if n_bootstrap > 0:
                boot = []
                for _ in range(n_bootstrap):
                    b = rng.choice(data, size=n, replace=True)
                    m, v = np.mean(b), np.var(b) + 1e-12
                    r = minimize(lambda th: -np.sum(scipy_stats.gamma(th[0], scale=th[1]).logpdf(b)), [m**2/v, v/m], method="L-BFGS-B", bounds=[(1e-6, None), (1e-6, None)])
                    if r.success:
                        boot.append(r.x)
                ci = _bootstrap_ci(boot, ["shape", "scale"])
            models.append({
                "name": "Gamma",
                "dist": dist,
                "params": {"shape": shape, "scale": scale},
                "n_params": 2,
                "loglike": loglike,
                "aic": aic,
                "bic": bic,
                "bootstrap_ci": ci if n_bootstrap > 0 else None,
            })
    except Exception:
        pass

    # --- E. Weibull ---
    def _weibull_negll(theta):
        c, scale = theta[0], theta[1]
        if c <= 0 or scale <= 0:
            return 1e20
        return -np.sum(scipy_stats.weibull_min(c, scale=scale).logpdf(data))

    try:
        c0 = 1.2
        scale0 = np.mean(data) / np.exp(np.log(np.e) / c0)
        res = minimize(_weibull_negll, [c0, scale0], method="L-BFGS-B", bounds=[(1e-6, None), (1e-6, None)])
        if res.success:
            c, scale = res.x[0], res.x[1]
            dist = scipy_stats.weibull_min(c, scale=scale)
            loglike = -res.fun
            k = 2
            aic = 2 * k - 2 * loglike
            bic = k * np.log(n) - 2 * loglike
            ci = {}
            if n_bootstrap > 0:
                boot = []
                for _ in range(n_bootstrap):
                    b = rng.choice(data, size=n, replace=True)
                    r = minimize(lambda th: -np.sum(scipy_stats.weibull_min(th[0], scale=th[1]).logpdf(b)), [1.2, np.mean(b)], method="L-BFGS-B", bounds=[(1e-6, None), (1e-6, None)])
                    if r.success:
                        boot.append(r.x)
                ci = _bootstrap_ci(boot, ["shape", "scale"])
            models.append({
                "name": "Weibull",
                "dist": dist,
                "params": {"shape": c, "scale": scale},
                "n_params": 2,
                "loglike": loglike,
                "aic": aic,
                "bic": bic,
                "bootstrap_ci": ci if n_bootstrap > 0 else None,
            })
    except Exception:
        pass

    # --- F. Optional: two-component mixtures (Gamma + Exponential, Wald + Exponential) ---
    if fit_mixtures and n >= 10:
        # Gamma + Exponential: f(t) = p * Gamma(t) + (1-p) * Exp(t)
        def _mix_gamma_exp_negll(theta):
            p, sh, sc, exp_scale = theta[0], theta[1], theta[2], theta[3]
            if not (0 < p < 1 and sh > 0 and sc > 0 and exp_scale > 0):
                return 1e20
            g = scipy_stats.gamma(sh, scale=sc).pdf(data)
            e = scipy_stats.expon(scale=exp_scale).pdf(data)
            mix = p * g + (1 - p) * e
            mix = np.clip(mix, 1e-300, None)
            return -np.sum(np.log(mix))

        try:
            p0, exp_scale0 = 0.5, np.mean(data)
            mean, var = np.mean(data), np.var(data) + 1e-12
            sh0, sc0 = mean**2 / var, var / mean
            res = minimize(_mix_gamma_exp_negll, [p0, sh0, sc0, exp_scale0], method="L-BFGS-B",
                          bounds=[(1e-6, 1 - 1e-6), (1e-6, None), (1e-6, None), (1e-6, None)])
            if res.success:
                p, sh, sc, exp_scale = res.x[0], res.x[1], res.x[2], res.x[3]
                g = scipy_stats.gamma(sh, scale=sc).pdf(data)
                e = scipy_stats.expon(scale=exp_scale).pdf(data)
                loglike = np.sum(np.log(np.clip(p * g + (1 - p) * e, 1e-300, None)))
                k = 4
                aic = 2 * k - 2 * loglike
                bic = k * np.log(n) - 2 * loglike
                class _MixGammaExp:
                    def __init__(self, p, sh, sc, es):
                        self.p, self.sh, self.sc, self.es = p, sh, sc, es
                    def pdf(self, x):
                        return self.p * scipy_stats.gamma(self.sh, scale=self.sc).pdf(x) + (1 - self.p) * scipy_stats.expon(scale=self.es).pdf(x)
                    def sf(self, x):
                        return self.p * scipy_stats.gamma(self.sh, scale=self.sc).sf(x) + (1 - self.p) * scipy_stats.expon(scale=self.es).sf(x)
                models.append({
                    "name": "Gamma+Exp mix",
                    "dist": _MixGammaExp(p, sh, sc, exp_scale),
                    "params": {"p": p, "shape": sh, "scale": sc, "exp_scale": exp_scale},
                    "n_params": 4,
                    "loglike": loglike,
                    "aic": aic,
                    "bic": bic,
                    "bootstrap_ci": None,
                })
        except Exception:
            pass

        # Wald + Exponential
        def _mix_wald_exp_negll(theta):
            p, mu, lam, exp_scale = theta[0], theta[1], theta[2], theta[3]
            if not (0 < p < 1 and mu > 0 and lam > 0 and exp_scale > 0):
                return 1e20
            w = _wald_pdf(data, mu, lam)
            e = scipy_stats.expon(scale=exp_scale).pdf(data)
            mix = np.clip(p * w + (1 - p) * e, 1e-300, None)
            return -np.sum(np.log(mix))

        try:
            mu0, lam0 = np.mean(data), max(np.mean(data)**3 / (np.var(data) + 1e-12), 0.1)
            res = minimize(_mix_wald_exp_negll, [0.5, mu0, lam0, np.mean(data)], method="L-BFGS-B",
                          bounds=[(1e-6, 1 - 1e-6), (1e-6, None), (1e-6, None), (1e-6, None)])
            if res.success:
                p, mu, lam, exp_scale = res.x[0], res.x[1], res.x[2], res.x[3]
                w = _wald_pdf(data, mu, lam)
                e = scipy_stats.expon(scale=exp_scale).pdf(data)
                loglike = np.sum(np.log(np.clip(p * w + (1 - p) * e, 1e-300, None)))
                k = 4
                aic = 2 * k - 2 * loglike
                bic = k * np.log(n) - 2 * loglike
                class _MixWaldExp:
                    def __init__(self, p, mu, lam, es):
                        self.p, self.mu, self.lam, self.es = p, mu, lam, es
                    def pdf(self, x):
                        return self.p * _wald_pdf(x, self.mu, self.lam) + (1 - self.p) * scipy_stats.expon(scale=self.es).pdf(x)
                    def sf(self, x):
                        return self.p * _WaldDist(self.mu, self.lam).sf(x) + (1 - self.p) * scipy_stats.expon(scale=self.es).sf(x)
                models.append({
                    "name": "Wald+Exp mix",
                    "dist": _MixWaldExp(p, mu, lam, exp_scale),
                    "params": {"p": p, "mu": mu, "lam": lam, "exp_scale": exp_scale},
                    "n_params": 4,
                    "loglike": loglike,
                    "aic": aic,
                    "bic": bic,
                    "bootstrap_ci": None,
                })
        except Exception:
            pass

    return models


def plot_dwell_time_fits(
    dwell_times,
    filepath,
    n_bins=80,
    n_bootstrap=200,
    show=False,
    plot_per_model=False,
    fit_mixtures=False,
):
    """
    Fit Gamma, Weibull, Inverse-Gaussian, Exponential, Lognormal to dwell times via MLE.
    Compute log-likelihood, AIC, BIC for model comparison.
    """
    dwell_times = np.asarray(dwell_times, dtype=float)
    dwell_times = dwell_times[np.isfinite(dwell_times) & (dwell_times > 0)]
    if dwell_times.size < 2:
        print("Not enough dwell times for fit comparison.")
        return

    models = _fit_dwell_time_models(dwell_times, n_bootstrap=n_bootstrap, fit_mixtures=fit_mixtures)
    if not models:
        print("No models could be fitted (scipy required).")
        return

    n = len(dwell_times)
    t_min = max(dwell_times.min(), 1e-10)
    t_max = dwell_times.max()
    t_plot = np.linspace(t_min, t_max, 300)

    # Empirical PDF: histogram with density=True
    hist, edges = np.histogram(dwell_times, bins=n_bins, density=True)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    width = edges[1] - edges[0] if len(edges) > 1 else 1.0

    # Empirical survival (for left panel: linear fit + CI, previous version)
    sorted_dt = np.sort(dwell_times)
    survival_emp = 1.0 - np.arange(1, n + 1) / (n + 1.0)
    log_survival = np.log(np.clip(survival_emp, 1e-15, 1.0))

    # Linear fit for survival: log(S) = slope*t + intercept => tau = -1/slope
    if _HAS_SCIPY and scipy_stats is not None and n >= 3:
        try:
            res = scipy_stats.linregress(sorted_dt, log_survival)
            slope, intercept, r_val, se_slope = res.slope, res.intercept, res.rvalue, res.stderr
        except Exception:
            slope, intercept, r_val, se_slope = _linear_fit_log_survival(sorted_dt, log_survival, n)
    else:
        slope, intercept, r_val, se_slope = _linear_fit_log_survival(sorted_dt, log_survival, n)
    rate_fit = -slope if slope != 0 else 1.0 / float(np.mean(dwell_times))
    tau_fit = 1.0 / rate_fit if rate_fit > 0 else np.nan
    se_tau = (float(np.abs(se_slope)) / (slope ** 2)) if slope != 0 and np.isfinite(se_slope) else np.nan
    if not np.isfinite(se_tau) or se_tau <= 0:
        se_tau = tau_fit / np.sqrt(n) if n > 0 and np.isfinite(tau_fit) else np.nan
    r_squared = (r_val ** 2) if np.isfinite(r_val) else np.nan
    t_fit = np.linspace(sorted_dt.min(), sorted_dt.max(), 200)
    survival_exp = np.exp(slope * t_fit + intercept)
    lambda_lo = lambda_hi = rate_fit
    s_lo = s_hi = survival_exp
    if np.isfinite(se_slope) and se_slope > 0 and slope != 0:
        lambda_lo = -(slope + 1.96 * se_slope)
        lambda_hi = -(slope - 1.96 * se_slope)
        if lambda_lo > 0 and lambda_hi > 0:
            s_lo = np.exp(-lambda_lo * t_fit + intercept)
            s_hi = np.exp(-lambda_hi * t_fit + intercept)

    # Match style and colors with plot_dwell_fit_with_params
    dot_color = "#5B6498"
    cmap = plt.get_cmap("tab20c")
    col_model = cmap(0)
    col_g = cmap(6)
    col_sna = cmap(10)
    col_vsna = cmap(5)

    fig, (ax_surv, ax_fp) = plt.subplots(
        1, 2, figsize=(8.0, 3.5), gridspec_kw={"width_ratios": [1.0, 1.4]}
    )

    # Left: survival with exponential fit (style-matched)
    ax_surv.scatter(
        sorted_dt,
        survival_emp,
        label="Empirical survival",
        s=14,
        facecolors="none",
        edgecolors=dot_color,
        linewidths=0.6,
    )
    ax_surv.semilogy(t_fit, survival_exp, "-", color=col_model, linewidth=1.2, label=f"Exponential (τ = {tau_fit:.3g})")
    ax_surv.set_xlabel("Dwell time Δt", fontsize=8)
    ax_surv.set_ylabel("Survival 1 − CDF", fontsize=8)
    ax_surv.set_title("Survival vs time", fontsize=9)
    text_lines = [f"τ = {tau_fit:.4g} ± {se_tau:.4g}", f"R² = {r_squared:.4f}" if np.isfinite(r_squared) else "R² = —"]
    ax_surv.text(0.05, 0.35, "\n".join(text_lines), transform=ax_surv.transAxes, fontsize=10, verticalalignment="top",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8), family="monospace")
    leg_surv = ax_surv.legend(loc="upper right", fontsize=8, frameon=True)
    if leg_surv is not None:
        leg_surv.get_frame().set_edgecolor("0.7")
        leg_surv.get_frame().set_linewidth(0.7)
    ax_surv.set_ylim(bottom=min(0.5 / (n + 1), float(survival_emp.min()) * 0.5))
    ax_surv.grid(True, which="both", ls="--", alpha=0.2, linewidth=0.4, color="0.7")
    for spine in ax_surv.spines.values():
        spine.set_linewidth(0.7)

    # Right: first-passage density (histogram) + best model fits overlaid (style-matched)
    ax_fp.bar(
        bin_centers,
        hist,
        width=width * 0.9,
        alpha=0.35,
        color=dot_color,
        label="Empirical (histogram)",
        edgecolor="none",
    )
    # Use tab20c colors for model curves in order
    model_colors = [col_model, col_g, col_sna, col_vsna]
    for i, m in enumerate(models):
        c = model_colors[i % len(model_colors)]
        try:
            pdf_vals = m["dist"].pdf(t_plot)
            pdf_vals = np.clip(pdf_vals, 1e-20, None)
            ax_fp.plot(t_plot, pdf_vals, "-", color=c, linewidth=1.2, label=m["name"])
        except Exception:
            pass
    ax_fp.set_xlabel("Dwell time t", fontsize=8)
    ax_fp.set_ylabel("Density f(t)", fontsize=8)
    ax_fp.set_title("First-passage density and fits", fontsize=9)
    leg_fp = ax_fp.legend(loc="upper right", fontsize=8, frameon=True)
    if leg_fp is not None:
        leg_fp.get_frame().set_edgecolor("0.7")
        leg_fp.get_frame().set_linewidth(0.7)
    ax_fp.set_xlim(left=0)
    ax_fp.set_ylim(bottom=0)
    ax_fp.grid(True, which="both", alpha=0.2, linestyle="--", linewidth=0.4, color="0.7")
    for spine in ax_fp.spines.values():
        spine.set_linewidth(0.7)
    
    # Right: dwell_fits_summary on the plot
    best_aic = min(m["aic"] for m in models)
    best_bic = min(m["bic"] for m in models)
    name_aic = next(m["name"] for m in models if m["aic"] == best_aic)
    name_bic = next(m["name"] for m in models if m["bic"] == best_bic)
    summary_lines = ["Model comparison (MLE)", "n = {}".format(n), "-" * 44]
    for m in models:
        summary_lines.append("  {}  LL={:.4f}  AIC={:.2f}  BIC={:.2f}".format(
            m["name"].ljust(18), m["loglike"], m["aic"], m["bic"]))
        summary_lines.append("    params: {}".format(m["params"]))
        if m.get("bootstrap_ci"):
            for k, (lo, hi) in m["bootstrap_ci"].items():
                summary_lines.append("    95% CI {}: ({:.4g}, {:.4g})".format(k, lo, hi))
    summary_lines.extend(["", "Best AIC: " + name_aic, "Best BIC: " + name_bic])
    # ax_txt.axis("off")
    # ax_txt.text(0.02, 0.98, "\n".join(summary_lines), transform=ax_txt.transAxes,
    #             fontsize=8, verticalalignment="top", fontfamily="monospace",
    #             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.9))
    # ax_txt.set_title("Dwell-fits summary", fontsize=12)
    
    fig.tight_layout()

    try:
        out_dir = PLOT_DIR
    except NameError:
        out_dir = Path(filepath).parent / "plots"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(filepath).stem
    outpath = out_dir / f"{stem}_dwell_fits.png"
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved dwell-time fits: left=survival (linear fit+CI), right=fp_density (histogram+all models): {outpath}")
    
    # Optional: separate plot per model (histogram + PDF, survival + fit)
    if plot_per_model:
        for i, m in enumerate(models):
            c = colors[i % len(colors)]
            fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.bar(bin_centers, hist, width=width * 0.9, alpha=0.6, color="#2e86ab", label="Empirical PDF", edgecolor="none")
            try:
                pdf_vals = m["dist"].pdf(t_plot)
                pdf_vals = np.clip(pdf_vals, 1e-20, None)
                ax1.plot(t_plot, pdf_vals, "-", color=c, linewidth=2, label=m["name"])
            except Exception:
                pass
            ax1.set_xlabel("Dwell time t")
            ax1.set_ylabel("Density f(t)")
            ax1.set_title(f"PDF: {m['name']} fit")
            ax1.legend()
            ax1.set_xlim(left=0)
            ax1.set_ylim(bottom=0)
            ax1.grid(True, alpha=0.3)
            ax2.semilogy(sorted_dt, survival_emp, "o", color="#2e86ab", markersize=4, label="Empirical survival")
            try:
                sf_vals = m["dist"].sf(t_plot)
                sf_vals = np.clip(sf_vals, 1e-15, 1.0)
                ax2.semilogy(t_plot, sf_vals, "-", color=c, linewidth=2, label=m["name"])
            except Exception:
                pass
            ax2.set_xlabel("Dwell time t")
            ax2.set_ylabel("Survival 1 − CDF")
            ax2.set_title(f"Survival: {m['name']} fit")
            ax2.legend()
            ax2.set_ylim(bottom=min(0.5 / (n + 1), float(survival_emp.min()) * 0.5))
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            safe_name = m["name"].replace(" ", "_").replace("-", "_")
            outpath_one = out_dir / f"{stem}_dwell_fits_{safe_name}.png"
            fig2.savefig(outpath_one, dpi=200)
            plt.close(fig2)
        print(f"Saved per-model plots: {stem}_dwell_fits_<Model>.png")

    # Summary table: log-likelihood, AIC, BIC (and optionally bootstrap CIs); also save to file
    lines_out = ["Dwell-time model comparison (MLE)", "n = {}".format(n), "-" * 60]
    print("\nDwell-time model comparison (MLE):")
    print("-" * 60)
    for m in models:
        line = f"  {m['name']:20s}  LL={m['loglike']:12.4f}  AIC={m['aic']:10.2f}  BIC={m['bic']:10.2f}  params={m['params']}"
        print(line)
        lines_out.append(line)
        if m.get("bootstrap_ci"):
            for k, (lo, hi) in m["bootstrap_ci"].items():
                s = f"      95% CI {k}: ({lo:.4g}, {hi:.4g})"
                print(s)
                lines_out.append(s)
    best_aic = min(m["aic"] for m in models)
    best_bic = min(m["bic"] for m in models)
    name_aic = next(m["name"] for m in models if m["aic"] == best_aic)
    name_bic = next(m["name"] for m in models if m["bic"] == best_bic)
    print(f"\n  Best AIC: {name_aic}")
    print(f"  Best BIC: {name_bic}")
    lines_out.extend(["", "Best AIC: " + name_aic, "Best BIC: " + name_bic])
    summary_path = out_dir / f"{stem}_dwell_fits_summary.txt"
    with open(summary_path, "w") as f:
        f.write("\n".join(lines_out))
    print(f"Saved summary: {summary_path}")
    if show:
        plt.show()


def dwell_time_analysis(y, 
                        filepath,
                        x=None,
                        box_height=None,
                        num_boxes=None,
                        ymin=None,
                        ymax=None,
                        stable_after_tol=0.0,   # used as a small threshold shift
                        max_gap=3,              # kept for API compatibility (not used)
                        min_run_len=1,          # kept for API compatibility (not used)
                        plot=True,
                        show=False,
                        show_boundaries=True,
                        plot_bins='auto'):
    """
    Simplified dwell-time:
      Record ONLY the FIRST time y crosses each boundary in an oscillation streak,
      then compute time differences between consecutive recorded crossings.

    Outputs:
      - boundaries: boundary levels (between boxes)
      - run_records: ndarray (M x 4): [boundary_idx, boundary_level, t_cross, sample_idx]
      - dwell_times: np.diff(t_cross) (length M-1)
      - run_start_times: t_cross values (length M)
      - info
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        raise ValueError("Empty y provided")

    if x is None:
        x = np.arange(n, dtype=float)
    else:
        x = np.asarray(x, dtype=float)
        if len(x) != n:
            raise ValueError("x and y must have same length")

    finite = np.isfinite(y)
    if not np.any(finite):
        raise ValueError("y contains no finite values")

    if ymin is None:
        ymin = float(np.nanmin(y))
    if ymax is None:
        ymax = float(np.nanmax(y))
    if ymax <= ymin:
        raise ValueError("Invalid ymin/ymax")

    if box_height is None:
        box_height = (ymax - ymin) / float(num_boxes)
    if box_height <= 0:
        raise ValueError("box_height must be positive")

    # ensure num_boxes matches box_height if box_height was explicitly provided
    num_boxes = max(1, int(np.ceil((ymax - ymin) / float(box_height))))
    if num_boxes < 2:
        raise ValueError("Need at least 2 boxes (increase num_boxes or decrease box_height)")

    # boundaries between boxes: ymin + (k+1)*box_height for k=0..num_boxes-2
    boundaries = np.array([ymin + (k+1) * box_height for k in range(num_boxes-1)], dtype=float)
    K = boundaries.size

    # record crossings: first time per oscillation streak for same boundary
    recorded = []  # (k, level, t_cross, i) where crossing occurs in interval (i-1,i)
    last_k = None  # dedupe repeated crossings of same boundary

    # Scan each interval once (fast)
    for i in range(1, n):
        y0, y1 = y[i-1], y[i]
        if not (np.isfinite(y0) and np.isfinite(y1)):
            continue
        if y1 == y0:
            continue

        # Determine which boundaries were crossed in this interval.
        # For dwell-time we ONLY care about first passage to the NEXT UPPER boundary,
        # so we ignore downward crossings entirely.
        if y1 > y0:
            # upward: boundaries with y0 < b <= y1
            s = np.searchsorted(boundaries, y0 + stable_after_tol, side='right')
            e = np.searchsorted(boundaries, y1 + stable_after_tol, side='right')
            if s >= e:
                continue
            ks = range(s, e)  # low -> high
        else:
            # ignore downward moves for dwell-time statistics
            continue

        for k in ks:
            level = boundaries[k]
            t_cross = _interp_cross_time(x[i-1], y0, x[i], y1, level)
            if t_cross is None:
                t_cross = float(x[i])

            # For dwell times we want FIRST passage to the next *new* upper boundary.
            # Ignore:
            #   - repeated crossings of the same boundary (oscillations)
            #   - crossings of any boundary that is at or below the last one already reached
            if last_k is None or k > last_k:
                recorded.append((int(k), float(level), float(t_cross), int(i)))
                last_k = k
            # else: k <= last_k  -> coming back up to a lower or same boundary; ignore

    if recorded:
        run_records = np.array(recorded, dtype=float)  # [k, level, t_cross, sample_i]
        run_start_times = run_records[:, 2]
        dwell_times = np.diff(run_start_times)
    else:
        run_records = np.empty((0, 4), dtype=float)
        run_start_times = np.empty((0,), dtype=float)
        dwell_times = np.empty((0,), dtype=float)

    info = {
        "ymin": ymin,
        "ymax": ymax,
        "box_height": float(box_height),
        "num_boxes": int(num_boxes),
        "n_crossings": int(run_records.shape[0]),
        "stable_after_tol": float(stable_after_tol),
    }

    result = {
        "boundaries": boundaries,
        "run_records": run_records,      # columns: boundary_idx, boundary_level, t_cross, sample_i
        "dwell_times": dwell_times,      # consecutive differences
        "run_start_times": run_start_times,
        "info": info,
    }

    # -------- plotting & saving (kept) --------
    if plot:
        # Side-by-side layout: left = crossings, right = dwell-time histogram
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.5), gridspec_kw={"width_ratios": [1.3, 1.0]})
        ax = axes[0]
        # Slightly thicker trajectory line for clarity
        ax.plot(x, y, linewidth=0.5, label="Position", color="#5B6498")

        if show_boundaries:
            for b in boundaries:
                ax.axhline(b, color='gray', linewidth=0.5, alpha=0.4)

        # plot recorded crossings using tab10 (consistent with trajectory plots), thinner edges
        cmap = plt.get_cmap("tab20")
        for rec in run_records:
            k = int(rec[0])
            level = rec[1]
            t_cross = rec[2]
            color = cmap(k % 10)

            ax.scatter(
                t_cross,
                level,
                s=8,
                marker='o',
                facecolor=color,
                edgecolor='grey',
                linewidths=0.5,
                zorder=6,
            )

        ax.set_title("First boundary crossings (deduped for oscillations)", fontsize=11)
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_ylabel("Position", fontsize=10)
        ax.grid(True, which="both", alpha=0.2, linestyle="--", linewidth=0.4, color="0.7")
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_edgecolor("black")
        leg_ax = ax.legend(loc="upper left", fontsize=9, frameon=True)
        if leg_ax is not None:
            leg_ax.get_frame().set_edgecolor("0.7")
            leg_ax.get_frame().set_linewidth(0.7)

        ax2 = axes[1]
        if dwell_times.size > 0:
            # Lighter blue for histogram bins
            ax2.hist(
                dwell_times,
                bins=20,
                alpha=0.9,
                edgecolor='black',
                linewidth=0.5,
                color="#5B6498",  # light blue
            )
            ax2.set_xlabel("Δt between consecutive first crossings (s)", fontsize=10)
            ax2.set_ylabel("Counts", fontsize=10)
            ax2.set_title("Dwell-time histogram (crossing-to-crossing)", fontsize=11)
            ax2.grid(True, which="both", alpha=0.2, linestyle="--", linewidth=0.4, color="0.7")
            for spine in ax2.spines.values():
                spine.set_linewidth(0.7)
                spine.set_edgecolor("black")
        else:
            ax2.text(0.5, 0.5, "No crossings found", ha='center', va='center', fontsize=10)
            ax2.set_xlabel("Δt", fontsize=10)
            ax2.set_ylabel("Counts", fontsize=10)

        plt.tight_layout()

        # save plots to PLOT_DIR if defined, else create 'plots' near file
        try:
            out_dir = PLOT_DIR
        except NameError:
            out_dir = Path(filepath).parent / "plots"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        outpath = out_dir / f"{Path(filepath).stem}_dwelltime.png"
        plt.savefig(outpath, dpi=200)
        if show:
            plt.show()
        plt.close(fig)

        # survival plot (optional, kept)
        plot_dwell_time_survival(dwell_times, filepath, show=False)
        # if dwell_times.size > 0:
        #     sorted_dt = np.sort(dwell_times)
        #     survival = 1.0 - np.arange(1, len(sorted_dt) + 1) / (len(sorted_dt) + 1.0)
        #     fig2 = plt.figure(figsize=(6,4))
        #     plt.loglog(sorted_dt, survival, marker='o', linestyle='-')
        #     plt.xlabel("Δt")
        #     plt.ylabel("Survival (1-CDF)")
        #     plt.title("Empirical survival of Δt (log-log)")
        #     plt.grid(True)

        #     outpath2 = out_dir / f"{Path(filepath).stem}_survival.png"
        #     plt.savefig(outpath2, dpi=200)
        #     if show:
        #         plt.show()
        #     plt.close(fig2)
        plot_first_passage_density(dwell_times, filepath, show=False)

    return dwell_times

# -----------------------------------

def plot_one_file(headers, data_cols, filepath, x_slice_start=0, show=True, save=True, crop=False, denoise=True,
                  dwell_time=True, reference_data_col=None):
    """
    Plot using first column as x and the 3rd Y column (index 2) like your snippet.
    If that column doesn't exist, skip plotting.
    """
    if headers is None or data_cols is None:
        print(f"Skipping (empty or unreadable): {filepath}")
        return

    # Ensure we have at least 3 columns (x + at least 2 y columns so index 2 exists)
    if len(data_cols) < 3:
        print(f"File {filepath} doesn't have a third column to plot (has {len(data_cols)} columns). Skipping.")
        return

    print(f"Plotting file: {filepath} with headers: {headers}")
    x = data_cols[0]
    y_target = data_cols[1:]
    print(f"  Original x length: {len(x)}, y columns count: {len(y_target)}")
        # make sure the two arrays have same length (they should)
    n = min(len(x), len(y_target))
    x = np.array(x)
    y = np.array(y_target[2])
    
    if reference_data_col is not None:
        print("initializing reference subtraction")
        x_reference = reference_data_col[0]
        y_reference = reference_data_col[1:]
        y = [y - r for y, r in zip(data_cols[1:][2], y_reference[2])]


    # slice starting from x_slice_start (like your original code)
    if x_slice_start is not None:
        if x_slice_start >= len(x):
            print(f"Warning: slice start {x_slice_start} >= data length ({len(x)}) for file {filepath}. Plot will be empty.")
        x = x[x_slice_start:]
        y = y[x_slice_start:]

    if crop:
        cut =  detect_noisy_tail_index(y)
        if cut < len(y) and cut > 0:
            x = x[:cut]
            y = y[:cut]
            print(f" -> Cropped data at index {cut} due to detected noisy tail.")

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    # Use same color theme as other trajectory plots
    ax.plot(
        x,
        y,
        label=headers[2] if headers and len(headers) > 2 else "Signal",
        linewidth=0.5,
        color="#5B6498",
    )
    ax.set_xlabel(headers[0] if headers else "Time (s)", fontsize=10)
    ax.set_ylabel("Position (bp)", fontsize=10)
    ax.set_title(f"{Path(filepath).name} – trajectory", fontsize=11)
    leg = ax.legend(loc="upper right", fontsize=9, frameon=True)
    if leg is not None:
        leg.get_frame().set_edgecolor("0.7")
        leg.get_frame().set_linewidth(0.7)
    ax.grid(True, which="both", alpha=0.2, linestyle="--", linewidth=0.4, color="0.7")
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_edgecolor("black")
    fig.tight_layout()

    if denoise: 
        y_denoised, info, x_coor = denoiszation(x, y, headers, filepath)
        print(f" Denoising info: {info}")
        print('denoised y length:', len(y_denoised), 'x_coor length:', len(x_coor))
        print(y_denoised[:10], x_coor[:10])
        ax.plot(
            x_coor,
            y_denoised,
            label="Denoised",
            color='tab:orange',
            linewidth=0.5,
        )
        ax.legend(loc="upper right", fontsize=9)

        if dwell_time:
            
            res = dwell_time_analysis(y_denoised, filepath, x, num_boxes=80, plot=True)
            # print("Found crossings:", res['info']['n_crossings_total'])
            # print("Dwell times (count):", len(res['dwell_times']))
            # print("Mean dwell:", np.nanmean(res['dwell_times']) if len(res['dwell_times'])>0 else None)
            print("Performing dwell time analysis on denoised data...")
    if save:
        out_dir = PLOT_DIR if 'PLOT_DIR' in globals() else Path(filepath).parent / "plots"
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        outpath = out_dir / f"{Path(filepath).stem}.png"
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"Saved plot: {outpath}")
    if show:
        plt.show()

    else:
        plt.close(fig)
    return res
# ==================

def main():
    VIVIAN_DIR = Path("/Users/vivian/Desktop/Undergrad Study/Part C Project/code/vivian/")
    # PLOT_DIR = VIVIAN_DIR / "dwell_time_plots"
    folders = sorted((VIVIAN_DIR / "data").glob("*"))
    dwell_times_list_all = np.array([], dtype=float) 
    print(folders)
    for fp in folders:
        dwell_times_list = np.array([], dtype=float) 
        print(f"\nProcessing folder: {fp}")
        txt_files = sorted(glob.glob(str(Path(fp) / "*.txt")))
        if not txt_files:
            print(f"No .txt files found in {Path(fp).resolve()}")
            continue
        for tfp in txt_files:
            print(f"\nProcessing: {tfp}")
            try:
                headers, data_cols = parse_file_like_original(tfp)
                if headers is None:
                    print(" -> file empty or couldn't be parsed, skipping.")
                    continue
                # print a short summary like your original print
                x_len = len(data_cols[0]) if data_cols else 0
                y_cols_count = max(0, len(data_cols) - 1)
                print(f"Parsed {x_len} X values and {y_cols_count} Y columns (headers: {headers})")
                
                dwell_times = plot_one_file(headers, data_cols, tfp, x_slice_start=12000, show=False, save=True, crop=True, denoise = True, dwell_time=True)
                dwell_times_list = np.append(dwell_times_list, dwell_times)
                print(f"Accumulated dwell times count: {len(dwell_times_list)}")

            except Exception as e:
                print(f"Error processing {tfp}: {e}")

        dwell_times_list_all = np.append(dwell_times_list_all, dwell_times_list)
        print(fp)
        # plot_dwell_time_survival(dwell_times_list, "vivian/cmg/Survival_Plots", show=False)
        # plot_first_passage_density(dwell_times_list, "vivian/cmg/", show=False, n_bins=30)
        plot_dwell_time_survival(dwell_times_list, fp, show=False)
        plot_first_passage_density(dwell_times_list, fp, show=False, n_bins=30)
    plot_dwell_time_survival(dwell_times_list_all, "vivian/All_data", show=False)
    plot_first_passage_density(dwell_times_list_all, "vivian/All_data", show=False, n_bins=30)

if __name__ == "__main__":
    main()
