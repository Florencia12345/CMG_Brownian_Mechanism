"""
Analyze how dwell-time distribution statistics depend on the number of bins (nbins).

Assumes dwell-time files are in `dwell_time_all/` with names like:
    dwell_times_all_<nbins>_<transition_label>.txt
e.g.
    dwell_times_all_200_High_Unwinding_3.txt
    dwell_times_all_80_Balanced_5.txt

We:
1. Parse nbins and a base transition label from each filename.
   - base_label = everything after nbins, with a trailing _<rep> stripped (e.g. _2,_3,_5).
2. For each file, compute distribution stats of the dwell times.
3. For each (base_label, nbins), average stats over replicates.
4. Plot selected stats vs nbins for each base_label to see which nbins choices are stable.
"""

from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


def _load_dwell_times(path: Path) -> np.ndarray:
    """Load dwell times from a text file, skipping a header row if present."""
    try:
        arr = np.loadtxt(path, skiprows=1)
    except Exception:
        # Fallback: try without skipping
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
    # log-mean (geometric-like) if all positive
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


def _parse_name(name: str):
    """
    Parse nbins and base_label from filename.
    Example:
        dwell_times_all_200_High_Unwinding_3.txt -> (200, 'High_Unwinding')
        dwell_times_all_80_Balanced_5.txt        -> (80, 'Balanced')
    """
    m = re.match(r"dwell_times_all_(\d+)_([^\.]+)\.txt$", name)
    if not m:
        return None, None
    nbins = int(m.group(1))
    label = m.group(2)
    # Strip trailing _<digits> as replicate id if present
    base_label = re.sub(r"_[0-9]+$", "", label)
    return nbins, base_label


def collect_stats(root: Path) -> dict:
    """
    Scan dwell_time_all for dwell_times_all_*.txt and return:
      stats_by_label[base_label][nbins] = list of per-file stats dicts
    """
    stats_by_label: dict = {}
    files = sorted(root.glob("dwell_times_all_*.txt"))
    if not files:
        raise FileNotFoundError(f"No dwell_times_all_*.txt files found in {root}")

    for path in files:
        nbins, base_label = _parse_name(path.name)
        if nbins is None:
            print(f"Skipping {path.name}: could not parse nbins/label")
            continue
        x = _load_dwell_times(path)
        if x.size == 0:
            print(f"Skipping {path.name}: no valid dwell times")
            continue
        s = _basic_stats(x)
        stats_by_label.setdefault(base_label, {}).setdefault(nbins, []).append(s)

    return stats_by_label


def aggregate_stats(stats_by_label: dict) -> dict:
    """
    For each (label, nbins), aggregate replicate stats:
      agg[label][nbins][metric] = (mean, std_across_reps)
    """
    agg: dict = {}
    for label, by_nbin in stats_by_label.items():
        agg[label] = {}
        for nbins, stats_list in by_nbin.items():
            if not stats_list:
                continue
            keys = stats_list[0].keys()
            agg[label][nbins] = {}
            for k in keys:
                vals = np.array([s[k] for s in stats_list], dtype=float)
                agg[label][nbins][k] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
                }
    return agg


def plot_stats_vs_nbins(agg: dict, out_dir: Path):
    """
    For each label, make a single, professional-looking figure with related
    statistics plotted together as functions of nbins.

    Layout (one PNG per label):
      - Top panel: central tendency (mean, median, log_mean)
      - Middle panel: spread / tail metrics (std, cv, p75, p90)
      - Bottom panel: peak features (peak_height, peak_prominence)

    Each metric is a line with a light uncertainty band (std across replicates).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Consistent, professional color palette (color-blind friendly)
    metric_colors = {
        "mean": "#1f77b4",             # blue
        "median": "#ff7f0e",           # orange
        "std": "#2ca02c",              # green
        "cv": "#d62728",               # red
        "p75": "#9467bd",              # purple
        "p90": "#8c564b",              # brown
        "log_mean": "#17becf",         # cyan
        "peak_height": "#e377c2",      # pink
        "peak_prominence": "#7f7f7f",  # gray
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
    peak_metrics = ["peak_height", "peak_prominence"]

    def _plot_group(ax, by_nbin: dict, metrics: list, title: str, log_y: bool, use_errorband: bool = True):
        nbins_sorted = sorted(by_nbin.keys())
        if not nbins_sorted:
            return

        for metric in metrics:
            y = []
            yerr = []
            for nb in nbins_sorted:
                stat_dict = by_nbin[nb]
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
                # Line + light error band
                ax.plot(
                    nbins_sorted,
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
                    nbins_sorted,
                    y_lower,
                    y_upper,
                    color=color,
                    alpha=0.18,
                    linewidth=0,
                )
            else:
                # Explicit error bars
                ax.errorbar(
                    nbins_sorted,
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

    for label, by_nbin in agg.items():
        nbins_sorted = sorted(by_nbin.keys())
        if not nbins_sorted:
            continue

        import matplotlib.gridspec as gridspec

        # One figure per label with 3 rows:
        # row 0: central (full width), row 1: spread (full), row 2: peaks (left/right)
        fig = plt.figure(figsize=(9, 10))
        gs = gridspec.GridSpec(3, 2, height_ratios=[1.1, 1.1, 1.0], hspace=0.35)

        ax_central = fig.add_subplot(gs[0, :])
        ax_spread = fig.add_subplot(gs[1, :])
        ax_peak_left = fig.add_subplot(gs[2, 0])
        ax_peak_right = fig.add_subplot(gs[2, 1], sharex=ax_peak_left)

        # Central and spread with shaded error bands
        _plot_group(ax_central, by_nbin, central_metrics, "Central tendency vs nbins", log_y=False, use_errorband=True)
        _plot_group(ax_spread, by_nbin, spread_metrics, "Spread / tail metrics vs nbins", log_y=True, use_errorband=True)

        # Peak height (left) and peak prominence (right) with explicit error bars
        _plot_group(ax_peak_left, by_nbin, ["peak_height"], "Peak height vs nbins", log_y=True, use_errorband=False)
        _plot_group(ax_peak_right, by_nbin, ["peak_prominence"], "Peak prominence vs nbins", log_y=True, use_errorband=False)

        ax_peak_left.set_xlabel("Number of bins (nbins)", fontsize=11)
        ax_peak_right.set_xlabel("Number of bins (nbins)", fontsize=11)

        fig.suptitle(f"Dwell-time statistics vs nbins\nTransition: {label}", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.94])
        out_path = out_dir / f"nbin_stats_{label}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("Saved:", out_path)


def write_stats_table(agg: dict, out_dir: Path):
    """
    Write a tab-separated table per label summarizing stats vs nbins.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = ["mean", "median", "std", "cv", "p75", "p90", "log_mean", "peak_height", "peak_prominence"]

    for label, by_nbin in agg.items():
        nbins_sorted = sorted(by_nbin.keys())
        if not nbins_sorted:
            continue
        path = out_dir / f"nbin_stats_{label}.txt"
        with open(path, "w") as f:
            header_parts = ["nbins"]
            for m in metrics:
                header_parts.append(f"{m}_mean")
                header_parts.append(f"{m}_std")
            f.write("\t".join(header_parts) + "\n")

            for nb in nbins_sorted:
                row = [str(nb)]
                stat_dict = by_nbin[nb]
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
    stats_by_label = collect_stats(base)
    agg = aggregate_stats(stats_by_label)
    out_dir = Path(__file__).parent / "dwell_time_all" / "nbin_stats_output"
    plot_stats_vs_nbins(agg, out_dir)
    write_stats_table(agg, out_dir)

