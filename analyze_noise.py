"""
Analyze how dwell-time distribution statistics depend on the added noise level.

Assumes dwell-time files for a given condition (e.g. High_Unwinding, nbin=250)
are saved in `dwell_time_all/` with names like:

  dwell_times_all_250_High_Unwinding.txt                 (no extra noise, level=0)
  dwell_times_all_250_High_Unwinding_noise_0.3.txt       (noise=0.3)
  dwell_times_all_250_High_Unwinding_noise_0.4.txt       (noise=0.4)
  ...

(replicates can be distinguished by extra suffixes before `.txt`;
all files sharing the same parsed noise level will be pooled as replicates.)

Pipeline (similar to analyze_nbin_stats.py):
  1. Parse noise level from filename.
  2. For each file, compute dwell-time distribution stats (mean, median, std, cv,
     percentiles, log_mean, peak_height, peak_prominence).
  3. For each noise level, aggregate stats across replicates (mean and std).
  4. For each base label (e.g. '250_High_Unwinding'), create a single figure:
       - row 1: central tendency vs noise
       - row 2: spread / tail vs noise
       - row 3: peak height (left) and peak prominence (right) vs noise
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


# Base pattern to select which condition to analyze
# (adjust if you want to look at a different base, e.g. 250_Balanced, etc.)
BASE_GLOB = "dwell_times_all_250_High_Unwinding*.txt"


def _load_dwell_times(path: Path) -> np.ndarray:
    """Load dwell times from a text file, skipping a header row if present."""
    try:
        arr = np.loadtxt(path, skiprows=1)
    except Exception:
        arr = np.loadtxt(path)
    arr = np.asarray(arr, dtype=float).ravel()
    arr = arr[np.isfinite(arr) & (arr > 0)]
    return arr


def _basic_stats(x: np.ndarray) -> dict:
    """Return basic distribution statistics and simple peak features for a 1D dwell-time array."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {}
    stats = {
        "n": int(x.size),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "cv": float(np.std(x, ddof=1) / np.mean(x)) if x.size > 1 and np.mean(x) > 0 else np.nan,
        "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)),
        "p90": float(np.percentile(x, 90)),
    }
    if np.all(x > 0):
        lx = np.log(x)
        stats["log_mean"] = float(np.mean(lx))
    else:
        stats["log_mean"] = np.nan

    # Simple measure of "peakiness": histogram on log-time grid
    if x.size > 10 and np.all(x > 0):
        t_min = float(np.min(x) * 0.9)
        t_max = float(np.max(x) * 1.1)
        if t_min <= 0:
            t_min = np.min(x[x > 0]) * 0.9
        edges = np.logspace(np.log10(t_min), np.log10(t_max), 60)
        density, _ = np.histogram(x, bins=edges, density=True)
        centers = np.sqrt(edges[:-1] * edges[1:])
        if np.any(density > 0):
            imax = int(np.argmax(density))
            peak_height = float(density[imax])
            peak_time = float(centers[imax])
            positive = density[density > 0]
            median_bg = float(np.median(positive))
            stats["peak_height"] = peak_height
            stats["peak_time"] = peak_time
            stats["peak_prominence"] = float(peak_height / median_bg) if median_bg > 0 else np.nan
        else:
            stats["peak_height"] = np.nan
            stats["peak_time"] = np.nan
            stats["peak_prominence"] = np.nan
    else:
        stats["peak_height"] = np.nan
        stats["peak_time"] = np.nan
        stats["peak_prominence"] = np.nan
    return stats


def _parse_noise_level(name: str) -> float:
    """
    Parse noise level from filename.

    Rules:
      - If pattern 'noise_<value>' appears before '.txt', use that <value> (float).
      - Otherwise, treat it as noise level 0.0 (baseline).
    """
    base = name
    if base.lower().endswith(".txt"):
        base = base[:-4]
    m = re.search(r"noise_([0-9]*\.?[0-9]+)", base, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    return 0.0


def collect_stats_by_noise(root: Path) -> dict:
    """
    Scan dwell_time_all for files matching BASE_GLOB and compute stats by noise level.
    Returns:
      stats_by_noise[noise_level] = list of stats dicts (one per file/replicate).
    """
    stats_by_noise: dict[float, list] = {}
    files = sorted(root.glob(BASE_GLOB))
    if not files:
        raise FileNotFoundError(f"No files matching '{BASE_GLOB}' in {root}")

    for path in files:
        noise = _parse_noise_level(path.name)
        x = _load_dwell_times(path)
        if x.size == 0:
            print(f"Skipping {path.name}: no valid dwell times")
            continue
        s = _basic_stats(x)
        stats_by_noise.setdefault(noise, []).append(s)

    return stats_by_noise


def aggregate_stats_by_noise(stats_by_noise: dict) -> dict:
    """
    For each noise_level, aggregate replicate stats:
      agg[noise][metric] = {'mean': ..., 'std': ...}
    """
    agg: dict[float, dict] = {}
    for noise, stats_list in stats_by_noise.items():
        if not stats_list:
            continue
        keys = stats_list[0].keys()
        agg[noise] = {}
        for k in keys:
            vals = np.array([s[k] for s in stats_list], dtype=float)
            agg[noise][k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
            }
    return agg


def plot_stats_vs_noise(agg: dict, out_dir: Path, label: str = "250_High_Unwinding"):
    """
    Make a single, professional-looking figure with statistics vs noise level.

    Layout (one PNG per label):
      - Row 1: central tendency (mean, median, log_mean)
      - Row 2: spread / tail metrics (std, cv, p75, p90)
      - Row 3: bottom-left = peak_height, bottom-right = peak_prominence

    Central and spread metrics use line + shaded band;
    peak metrics use explicit error bars.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Palette (color-blind friendly)
    metric_colors = {
        "mean": "#1f77b4",
        "median": "#ff7f0e",
        "std": "#2ca02c",
        "cv": "#d62728",
        "p75": "#9467bd",
        "p90": "#8c564b",
        "log_mean": "#17becf",
        "peak_height": "#e377c2",
        "peak_prominence": "#7f7f7f",
    }
    metric_titles = {
        "mean": "Mean dwell time",
        "median": "Median dwell time",
        "std": "Std. deviation",
        "cv": "Coeff. of variation (std/mean)",
        "p75": "75th percentile",
        "p90": "90th percentile",
        "log_mean": "Mean log(dwell time)",
        "peak_height": "Peak height (PDF at mode)",
        "peak_prominence": "Peak prominence",
    }

    central_metrics = ["mean", "median", "log_mean"]
    spread_metrics = ["std", "cv", "p75", "p90"]

    def _plot_group(ax, by_noise: dict, metrics: list, title: str, log_y: bool, use_errorband: bool = True):
        noises_sorted = sorted(by_noise.keys())
        if not noises_sorted:
            return

        for metric in metrics:
            y = []
            yerr = []
            for noise in noises_sorted:
                stat_dict = by_noise[noise]
                if metric not in stat_dict:
                    y.append(np.nan)
                    yerr.append(0.0)
                else:
                    y.append(stat_dict[metric]["mean"])
                    yerr.append(stat_dict[metric]["std"])
            y = np.array(y, dtype=float)
            yerr = np.array(yerr, dtype=float)

            color = metric_colors.get(metric, "#1f77b4")

            if log_y and np.all(y > 0):
                ax.set_yscale("log")

            if use_errorband:
                ax.plot(
                    noises_sorted,
                    y,
                    "-o",
                    color=color,
                    linewidth=1.8,
                    markersize=4,
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                    label=metric_titles.get(metric, metric),
                )
                y_upper = y + yerr
                y_lower = y - yerr
                ax.fill_between(
                    noises_sorted,
                    y_lower,
                    y_upper,
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )
            else:
                ax.errorbar(
                    noises_sorted,
                    y,
                    yerr=yerr,
                    fmt="-o",
                    color=color,
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=4,
                    capthick=1.2,
                    markersize=4,
                    markerfacecolor="white",
                    markeredgewidth=1.2,
                    label=metric_titles.get(metric, metric),
                )

        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=9)

    noises_sorted = sorted(agg.keys())
    if not noises_sorted:
        return

    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(9, 10))
    gs = gridspec.GridSpec(3, 2, height_ratios=[1.1, 1.1, 1.0], hspace=0.35)

    ax_central = fig.add_subplot(gs[0, :])
    ax_spread = fig.add_subplot(gs[1, :])
    ax_peak_left = fig.add_subplot(gs[2, 0])
    ax_peak_right = fig.add_subplot(gs[2, 1], sharex=ax_peak_left)

    _plot_group(ax_central, agg, central_metrics, "Central tendency vs noise level", log_y=False, use_errorband=True)
    _plot_group(ax_spread, agg, spread_metrics, "Spread / tail metrics vs noise level", log_y=True, use_errorband=True)

    _plot_group(ax_peak_left, agg, ["peak_height"], "Peak height vs noise level", log_y=True, use_errorband=False)
    _plot_group(ax_peak_right, agg, ["peak_prominence"], "Peak prominence vs noise level", log_y=True, use_errorband=False)

    ax_peak_left.set_xlabel("Noise level", fontsize=11)
    ax_peak_right.set_xlabel("Noise level", fontsize=11)

    fig.suptitle(f"Dwell-time statistics vs noise level\nCondition: {label}", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.94])
    out_path = out_dir / f"noise_stats_{label}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def write_noise_stats_table(agg: dict, out_dir: Path, label: str = "250_High_Unwinding"):
    """Write a tab-separated table summarizing stats vs noise level."""
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["mean", "median", "std", "cv", "p75", "p90", "log_mean", "peak_height", "peak_prominence"]
    noises_sorted = sorted(agg.keys())
    if not noises_sorted:
        return

    path = out_dir / f"noise_stats_{label}.txt"
    with path.open("w") as f:
        header_parts = ["noise"]
        for m in metrics:
            header_parts.append(f"{m}_mean")
            header_parts.append(f"{m}_std")
        f.write("\t".join(header_parts) + "\n")

        for noise in noises_sorted:
            row = [f"{noise:.6g}"]
            stat_dict = agg[noise]
            for m in metrics:
                if m in stat_dict:
                    row.append(f"{stat_dict[m]['mean']:.6g}")
                    row.append(f"{stat_dict[m]['std']:.6g}")
                else:
                    row.append("nan")
                    row.append("nan")
            f.write("\t".join(row) + "\n")
    print("Saved table:", path)


if __name__ == "__main__":
    base = Path(__file__).parent / "dwell_time_all"
    stats_by_noise = collect_stats_by_noise(base)
    agg = aggregate_stats_by_noise(stats_by_noise)
    out_dir = Path(__file__).parent / "dwell_time_all" / "noise_stats_output"
    label = "250_High_Unwinding"
    plot_stats_vs_noise(agg, out_dir, label=label)
    write_noise_stats_table(agg, out_dir, label=label)

