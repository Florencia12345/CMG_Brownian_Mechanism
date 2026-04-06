"""
Plot first-passage density and paper-model fit for a single dwell-time file,
using user-specified fitting parameters and the `plot_fit` helper from
`fit_dwelltime.py`.

Usage example (from this directory):

    python plot_dwell_fit_with_params.py \\
        dwell_time_all/dwell_times_all_250_High_Unwinding.txt \\
        --N_steps 20 \\
        --f_FNA 0.4 --f_SNA 0.3 --f_VSNA 0.2 --f_LLP 0.1 \\
        --T_FNA 5.0 --T_SNA 10.0 --T_VSNA 20.0 \\
        --save mle_fit_custom.png

This script:
  1. Reads dwell times from a text file (skipping the first row by default).
  2. Builds a `fit_out` dictionary in the same format returned by
     `fit_paper_model_MLE` in `fit_dwelltime.py`, but using parameters
     supplied on the command line.
  3. Calls `plot_fit(dwell_times, fit_out, N_steps=...)` to generate
     a survival + first-passage density figure.
"""

from pathlib import Path
import argparse
import numpy as np
import math
import matplotlib.pyplot as plt

# from fit_dwelltime import plot_fit

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
    centers = np.sqrt(edges[:-1] * edges[1:])
    sorted_dt = np.sort(dt)
    n = len(sorted_dt)
    surv = 1.0 - np.arange(1, n + 1) / (n + 1.0)

    # Subsample points for plotting (show ~10% of dwell-time dots)
    if n > 0:
        idx_all = np.arange(n)
        mask_surv = (idx_all % 10 == 0)
        if not np.any(mask_surv):
            mask_surv = np.ones_like(idx_all, dtype=bool)
        sorted_dt_plot = sorted_dt[mask_surv]
        surv_plot = surv[mask_surv]
    else:
        sorted_dt_plot = sorted_dt
        surv_plot = surv

    dx = np.diff(xs)
    trapz = (pdf_vals[:-1] + pdf_vals[1:]) * 0.5 * dx
    cdf_vals = np.concatenate([[0.0], np.cumsum(trapz)])
    total = cdf_vals[-1]
    if total > 1e-300:
        cdf_vals = cdf_vals / total
    S_model = 1.0 - cdf_vals

    # Styling: match overall project style, with custom colors requested
    dot_color = "#5B6498"   
    dot_color = "#E4E6F3"             # same as trajectory dots
    cmap = plt.get_cmap("tab20c")       # for fitted curves/components
    col_model = cmap(0)
    col_g = cmap(6)
    col_sna = cmap(10)
    col_vsna = cmap(5)

    # Layout: top row (fits), bottom row (residuals)
    fig = plt.figure(figsize=(8.0, 5.2))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1.3, 1.35],
        height_ratios=[3.0, 1.1],
        hspace=0.12, wspace=0.22
    )
    ax1 = fig.add_subplot(gs[0, 0])   # survival fit
    ax2 = fig.add_subplot(gs[0, 1])   # pdf fit
    ax1r = fig.add_subplot(gs[1, 0], sharex=ax1)  # survival residual
    ax2r = fig.add_subplot(gs[1, 1], sharex=ax2)  # pdf residual

    # Left: survival S(t), log-y
    ax1.scatter(
        sorted_dt_plot,
        surv_plot,
        label="Empirical survival",
        s=14,
        facecolors="none",
        edgecolors=dot_color,
        linewidths=0.6,
    )
    ax1.plot(xs, S_model, label="Analytical S(t)", lw=1.2, color=col_model)
    ax1.set_yscale("log")
    # ax1.set_xlabel("Passage time (s)", fontsize=8)
    ax1.set_ylabel("Survival S(t)", fontsize=8)
    ax1.set_title("Survival vs time", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", ls="--", alpha=0.2, linewidth=0.4)
    for spine in ax1.spines.values():
        spine.set_linewidth(0.7)
    ax1.tick_params(labelbottom=False)

    # Right: first-passage density (PDF), log-log
    # Subsample histogram-bin dots similarly (~10%)
    n_centers = len(centers)
    if n_centers > 0:
        idx_c = np.arange(n_centers)
        mask_c = (idx_c % 2 == 0)
        if not np.any(mask_c):
            mask_c = np.ones_like(idx_c, dtype=bool)
        centers_plot = centers[mask_c]
        counts_plot = counts[mask_c]
    else:
        centers_plot = centers
        counts_plot = counts

    ax2.scatter(
        centers_plot,
        counts_plot,
        label="Empirical first-passage density",
        s=14,
        facecolors="none",
        edgecolors=dot_color,
        linewidths=0.6,
    )
    ax2.plot(xs, pdf_vals, label="Analytical PDF", lw=1.2, color=col_model)
    ax2.plot(xs[20:220], g_comp[20:220], "--", lw=1.0, color=col_g, label="Gamma distribution")
    ax2.plot(xs[130:400], q_s1[130:400], "--", lw=1.0, color=col_sna, label="Exponential distribution")
    # ax2.plot(xs[200:400], q_s2[200:400], "--", lw=1.0, color=col_vsna, label="Q·VSNA")
    ax2.set_yscale("log")
    ax2.set_xscale("log")
    ax2.set_xlabel("Passage time (s)", fontsize=8)
    ax2.set_ylabel("First-passage density (PDF)", fontsize=8)
    ax2.set_title("First-passage density and fitting", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.grid(True, which="both", ls="--", alpha=0.2, linewidth=0.4)
    for spine in ax2.spines.values():
        spine.set_linewidth(0.7)

    # --- Residual panels ---
    # Survival residual in log-space: use ALL dwell-time points (no subsampling)
    s_model_at_emp_all = np.interp(sorted_dt, xs, S_model)
    surv_resid_all = np.log10(np.clip(surv, 1e-15, None)) - np.log10(np.clip(s_model_at_emp_all, 1e-15, None))
    # connect lines + small hollow dots
    ax1r.plot(sorted_dt, surv_resid_all, color=dot_color, linewidth=0.5, alpha=0.8)
    ax1r.scatter(
        sorted_dt,
        surv_resid_all,
        s=6,
        facecolors=dot_color,
        edgecolors=dot_color,
        linewidths=0.3,
    )
    ax1r.axhline(0.0, color=col_model, linestyle="--", linewidth=0.9)
    ax1r.set_xscale("log")
    ax1r.set_xlabel("Passage time (s)", fontsize=8)
    ax1r.set_ylabel(r"$\Delta \log_{10} S$", fontsize=8)
    ax1r.grid(True, which="both", ls="--", alpha=0.2, linewidth=0.4)
    for spine in ax1r.spines.values():
        spine.set_linewidth(0.7)

    # PDF residual (relative): use ALL histogram centers (no subsampling)
    model_pdf_at_centers_all = np.interp(centers, xs, pdf_vals)
    pdf_resid_all = (counts - model_pdf_at_centers_all) / np.clip(model_pdf_at_centers_all, 1e-15, None)
    # connect lines + small hollow dots
    ax2r.plot(centers, pdf_resid_all, color=dot_color, linewidth=0.5, alpha=0.8)
    ax2r.scatter(
        centers,
        pdf_resid_all,
        s=6,
        facecolors=dot_color,
        edgecolors=dot_color,
        linewidths=0.3,
    )
    ax2r.axhline(0.0, color=col_model, linestyle="--", linewidth=0.9)
    ax2r.set_xscale("log")
    # ax2r.set_yscale("log")
    ax2r.set_xlabel("Passage time (s)", fontsize=8)
    ax2r.set_ylabel("Rel. resid. (full)", fontsize=8)
    ax2r.grid(True, which="both", ls="--", alpha=0.2, linewidth=0.4)
    for spine in ax2r.spines.values():
        spine.set_linewidth(0.7)

    # Second y-scale for larger x (x >= 10^0): zoom to show oscillation details
    mask_large_x = centers >= 1.0
    if np.any(mask_large_x):
        x_large = centers[mask_large_x]
        r_large = pdf_resid_all[mask_large_x]
        ax2r_zoom = ax2r.twinx()
        ax2r_zoom.plot(x_large, r_large, color=col_sna, linewidth=0.55, alpha=0.9)
        ax2r_zoom.scatter(
            x_large,
            r_large,
            s=6,
            facecolors=col_sna,
            edgecolors=col_sna,
            linewidths=0.25,
            alpha=0.9,
        )
        ax2r_zoom.axhline(0.0, color=col_sna, linestyle=":", linewidth=0.8, alpha=0.9)
        ax2r_zoom.set_ylabel(r"Rel. resid. ($x \geq 10^0$, zoom)", fontsize=8, color=col_sna)
        ax2r_zoom.tick_params(axis="y", labelsize=8, colors=col_sna)
        # Robust zoom limits from central spread
        if r_large.size >= 3:
            q10, q90 = np.percentile(r_large, [10, 90])
            center = 0.5 * (q10 + q90)
            half = max(0.5 * (q90 - q10) * 1.25, 1e-3)
            ax2r_zoom.set_ylim(center - half, center + half)

    if save_path:
        fig.savefig(save_path, dpi=1500, bbox_inches='tight')
        plt.close(fig)
    if show:
        plt.show()
    return fig


def _load_dwell_times(path: Path, skiprows: int = 1) -> np.ndarray:
    """Load dwell times from a text file, skipping header if present."""
    try:
        arr = np.loadtxt(path, skiprows=skiprows)
    except Exception:
        arr = np.loadtxt(path)
    arr = np.asarray(arr, dtype=float).ravel()
    arr = arr[np.isfinite(arr) & (arr > 0)]
    return arr


def build_fit_out_from_params(
    f_FNA: float,
    f_SNA: float,
    f_VSNA: float,
    f_LLP: float,
    T_FNA: float,
    T_SNA: float,
    T_VSNA: float,
) -> dict:
    """
    Construct a `fit_out` dict compatible with `plot_fit` from user-specified parameters.
    """
    params = {
        "f_FNA": float(f_FNA),
        "f_SNA": float(f_SNA),
        "f_VSNA": float(f_VSNA),
        "f_LLP": float(f_LLP),
        "T_FNA": float(T_FNA),
        "T_SNA": float(T_SNA),
        "T_VSNA": float(T_VSNA),
    }
    fit_out = {
        "params": params,
        "logL": float("nan"),
        "rss": None,
        "nll": float("nan"),
        "bic": float("nan"),
        "success": True,
        "opt": None,
        "time_sec": 0.0,
        "convergence_message": "user-specified parameters",
        "n_bins": None,
        "n_dwell_times": None,
    }
    return fit_out


def main():
    p = argparse.ArgumentParser(
        description="Plot first-passage density and paper-model PDF for a dwell-time file using user-specified parameters."
    )
    p.add_argument("dwelltime_path", type=str, help="Path to dwell-time text file (one column of times).")
    p.add_argument("--skiprows", type=int, default=1, help="Header rows to skip when loading dwell times (default: 1).")
    p.add_argument("--N_steps", type=int, default=3, help="N_steps parameter for the paper PDF (gamma shape).")

    # Fractions
    p.add_argument("--f_FNA", type=float, required=True, help="Fraction f_FNA.")
    p.add_argument("--f_SNA", type=float, required=True, help="Fraction f_SNA.")
    p.add_argument("--f_VSNA", type=float, required=True, help="Fraction f_VSNA.")
    p.add_argument("--f_LLP", type=float, required=True, help="Fraction f_LLP.")

    # Time scales
    p.add_argument("--T_FNA", type=float, required=True, help="Timescale T_FNA.")
    p.add_argument("--T_SNA", type=float, required=True, help="Timescale T_SNA.")
    p.add_argument("--T_VSNA", type=float, required=True, help="Timescale T_VSNA.")

    p.add_argument("--nbins", type=int, default=200, help="Number of histogram bins for first-passage density in plot_fit.")
    p.add_argument("--save", type=str, default=None, help="Path to save the plot PNG. If omitted, show only.")
    p.add_argument("--no-show", action="store_true", help="Do not display the figure (only save).")

    args = p.parse_args()

    dwell_path = Path(args.dwelltime_path)
    if not dwell_path.is_file():
        raise FileNotFoundError(f"Dwell-time file not found: {dwell_path}")

    dwell_times = _load_dwell_times(dwell_path, skiprows=args.skiprows)
    if dwell_times.size == 0:
        raise ValueError(f"No valid dwell times loaded from: {dwell_path}")

    fit_out = build_fit_out_from_params(
        args.f_FNA,
        args.f_SNA,
        args.f_VSNA,
        args.f_LLP,
        args.T_FNA,
        args.T_SNA,
        args.T_VSNA,
    )

    print(f"Loaded {dwell_times.size} dwell times from {dwell_path}")
    print("Using parameters:", fit_out["params"])

    plot_fit(
        dwell_times,
        fit_out,
        N_steps=args.N_steps,
        nbins=args.nbins,
        show=not args.no_show,
        save_path=args.save,
    )


if __name__ == "__main__":
    main()

# python plot_dwell_fit_with_params.py dwell_time_all/sim_traces_nbin_2500__run_Family_D_5_noise_0.5__duration_10000_ntraj_200__dwell_times_all.txt --N_steps 20 --f_FNA 0.64971 --f_SNA 3.47E-07 --f_VSNA 3.57121e-07 --f_LLP 0.350289  --T_FNA 0.237745 --T_SNA 4.96929 --T_VSNA 9.99578 --save mle_fit_custom.png 