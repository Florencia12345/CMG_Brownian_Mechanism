"""
Analyze relationships between fitted dwell-time parameters and mean velocity
for different transition-matrix families.

Reads a text file (default: dwell_time_param.txt) with columns like

  nbin_dir   family   N_steps   f_FNA   f_SNA   f_VSNA   f_LLP   T_FNA   T_SNA   T_VSNA   Mean velocity

(header names can vary slightly; we normalize them).

We:
  - Read the SECOND column ('family') and collapse names into canonical
    families:
        High_Unwinding, Low_Unwinding, High_Pausing, Balanced,
        Family_B, Family_C, Family_D.
  - For each family, compute two relationships:
        1)  x = T_FNA / N_steps           vs  y = mean_velocity
        2)  x = mean_velocity * N_steps / T_FNA  vs  y = f_FNA^(1/N_steps)
  - Plot the two panels side by side per family and save each figure as
        dwell_param_plots/dwell_param_relations_<Family>.png
"""

from pathlib import Path
import re
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

# Match visualize_2.py global style
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


def _norm(s: str) -> str:
    """Normalize header names."""
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())


def _canonical_family(name: str) -> str:
    """
    Collapse various 'family' strings (e.g. run_High_Unwinding_3_noise_0.5)
    into canonical labels.
    """
    base = Path(name).name
    base = re.sub(r"^[Rr][Uu][Nn]_", "", base)
    base = re.sub(r"_[0-9]+$", "", base)
    key = base.lower()

    if "high_unw" in key:
        return "High_Unwinding"
    if "low_unw" in key:
        return "Low_Unwinding"
    if "high_paus" in key:
        return "High_Pausing"
    if "balanced" in key:
        return "Balanced"
    if "family_b" in key or "family_b" in key or key.endswith("_b"):
        return "Family_B"
    if "family_c" in key or "family_c" in key or key.endswith("_c"):
        return "Family_C"
    if "family_d" in key or "family_d" in key or key.endswith("_d"):
        return "Family_D"
    return base


def _load_param_table(path: Path) -> Dict[str, np.ndarray]:
    """
    Load dwell_time_param.txt as a dict of numpy arrays.
    We only need: family (2nd column), N_steps, T_FNA, f_FNA, mean_velocity.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open("r") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"No data in {path}")

    header = lines[0]
    sep = "\t" if "\t" in header else None
    header_cols = header.split(sep) if sep is not None else header.split()
    norm = [_norm(c) for c in header_cols]

    col_family = 1  # explicitly second column per spec
    # Find numeric/parameter columns by normalized header
    col_n = None
    col_tfna = None
    col_ffna = None
    col_meanv = None
    col_fsna = None
    col_fvsna = None
    col_fllp = None
    col_tsna = None
    col_tvsna = None
    for i, key in enumerate(norm):
        if key in ("nsteps", "n", "nstep", "n_step"):
            col_n = i
        elif key in ("tfna", "t_fna", "tfnas", "tfnasec"):
            col_tfna = i
        elif key in ("ffna", "f_fna"):
            col_ffna = i
        elif key in ("fsna", "f_sna"):
            col_fsna = i
        elif key in ("fvsna", "f_vsna"):
            col_fvsna = i
        elif key in ("fllp", "f_llp"):
            col_fllp = i
        elif key in ("tsna", "t_sna", "tsnasec"):
            col_tsna = i
        elif key in ("tvsna", "t_vsna", "tvsnasec"):
            col_tvsna = i
        elif key in ("meanv", "meanvel", "mean_velocity", "vmean", "v_mean") or "meanvelocity" in key:
            # Always prefer the rightmost match in case header has multiple pieces;
            # this ensures we capture the *last* column, which is Mean velocity.
            col_meanv = i

    # If we still didn't detect mean_v by name, default to the last column
    if col_meanv is None:
        col_meanv = len(header_cols) - 1

    missing = []
    if col_n is None:
        missing.append("N_steps")
    if col_tfna is None:
        missing.append("T_FNA")
    if col_ffna is None:
        missing.append("f_FNA")
    if col_meanv is None:
        missing.append("mean_velocity")
    if missing:
        raise ValueError(
            f"Missing required columns {missing} in header of {path}.\n"
            f"Header (normalized): {norm}"
        )

    fam_raw_list: List[str] = []
    fam_list: List[str] = []
    n_list: List[float] = []
    tfna_list: List[float] = []
    ffna_list: List[float] = []
    fsna_list: List[float] = []
    fvsna_list: List[float] = []
    fllp_list: List[float] = []
    tsna_list: List[float] = []
    tvsna_list: List[float] = []
    meanv_list: List[float] = []

    for ln in lines[1:]:
        parts = ln.split(sep) if sep is not None else ln.split()
        if len(parts) < len(header_cols):
            parts = parts + [""] * (len(header_cols) - len(parts))
        try:
            fam_raw = parts[col_family]
            n = float(parts[col_n])
            tfna = float(parts[col_tfna])
            ffna = float(parts[col_ffna])
            fsna = float(parts[col_fsna]) if col_fsna is not None else np.nan
            fvsna = float(parts[col_fvsna]) if col_fvsna is not None else np.nan
            fllp = float(parts[col_fllp]) if col_fllp is not None else np.nan
            tsna = float(parts[col_tsna]) if col_tsna is not None else np.nan
            tvsna = float(parts[col_tvsna]) if col_tvsna is not None else np.nan
            mv = float(parts[col_meanv])
        except Exception:
            continue
        fam_raw_list.append(fam_raw)
        fam_list.append(_canonical_family(fam_raw))
        n_list.append(n)
        tfna_list.append(tfna)
        ffna_list.append(ffna)
        fsna_list.append(fsna)
        fvsna_list.append(fvsna)
        fllp_list.append(fllp)
        tsna_list.append(tsna)
        tvsna_list.append(tvsna)
        meanv_list.append(mv)
    if not fam_list:
        raise ValueError(f"No valid rows parsed from {path}")

    return {
        "family_raw": np.array(fam_raw_list, dtype=str),
        "family": np.array(fam_list, dtype=str),
        "n_steps": np.array(n_list, dtype=float),
        "t_fna": np.array(tfna_list, dtype=float),
        "f_fna": np.array(ffna_list, dtype=float),
        "f_sna": np.array(fsna_list, dtype=float),
        "f_vsna": np.array(fvsna_list, dtype=float),
        "f_llp": np.array(fllp_list, dtype=float),
        "t_sna": np.array(tsna_list, dtype=float),
        "t_vsna": np.array(tvsna_list, dtype=float),
        "mean_v": np.array(meanv_list, dtype=float),
    }


def _setup_axes(ax):
    ax.grid(True, which="both", alpha=0.2, linestyle="--", linewidth=0.4, color="0.7")
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_edgecolor("black")


def _pad_axes(ax, frac: float = 0.1):
    """
    Extend x and y limits slightly beyond data range to give visual padding,
    approximating 'one grid' in each direction.
    """
    try:
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
    except Exception:
        return
    if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
        dx = (xmax - xmin) * frac
        ax.set_xlim(xmin - dx, xmax + dx)
    if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
        dy = (ymax - ymin) * frac
        ax.set_ylim(ymin - dy, ymax + dy)


def _add_linear_fit_1(ax, x, y, color, label_prefix=""):
    """
    Fit y = a*x + b on finite points, draw the line, and annotate equation + R^2.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return
    x_fit = x[mask]
    y_fit = y[mask]
    # Ordinary least squares
    a, b = np.polyfit(x_fit, y_fit, 1)
    y_pred = a * x_fit + b
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Plot fitted line over slightly extended x-range
    x_min, x_max = x_fit.min(), x_fit.max()
    dx = (x_max - x_min) * 0.25
    x_line = np.linspace(x_min - dx, x_max + dx, 200)
    y_line = a * x_line + b
    ax.plot(x_line, y_line, color=color, linewidth=1.0, linestyle="--", alpha=0.9)

    # Text box with equation and R^2
    eq_label = f"{label_prefix}$v_{{\\mathrm{{mean}}}}$ = {a:.3g} $\\frac{{T_{{\\mathrm{{FNA}}}}}}{{N_{{\\mathrm{{steps}}}}}}$ + {b:.3g}\nR² = {r2:.3f}"
    ax.text(
        0.03,
        0.88,
        eq_label,
        transform=ax.transAxes,
        fontsize=6.5,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="0.7"),
    )


def _add_linear_fit_2(ax, x, y, color, label_prefix=""):
    """
    Fit y = a*x + b on finite points, draw the line, and annotate equation + R^2.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return
    x_fit = x[mask]
    y_fit = y[mask]
    # Ordinary least squares
    a, b = np.polyfit(x_fit, y_fit, 1)
    y_pred = a * x_fit + b
    ss_res = np.sum((y_fit - y_pred) ** 2)
    ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Plot fitted line over slightly extended x-range
    x_min, x_max = x_fit.min(), x_fit.max()
    dx = (x_max - x_min) * 0.25
    x_line = np.linspace(x_min - dx, x_max + dx, 200)
    y_line = a * x_line + b
    ax.plot(x_line, y_line, color=color, linewidth=1.0, linestyle="--", alpha=0.9)

    # Text box with equation and R^2
    eq_label = f"{label_prefix}$f_{{\\mathrm{{FNA}}}}^{{1/N_{{\\mathrm{{steps}}}}}}$ = {a:.3g} $\\frac{{v_{{\\mathrm{{mean}}}} \\cdot N_{{\\mathrm{{steps}}}}}}{{T_{{\\mathrm{{FNA}}}}}}$ + {b:.3g}\nR² = {r2:.3f}"
    ax.text(
        0.03,
        0.88,
        eq_label,
        transform=ax.transAxes,
        fontsize=6.5,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="0.7"),
    )


def plot_family_relationships(data: Dict[str, np.ndarray], out_dir: Path):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fam = data["family"]
    unique_fam = np.unique(fam)
    cmap = plt.get_cmap("tab20c")

    for fam_name in unique_fam:
        mask = fam == fam_name
        n = data["n_steps"][mask]
        tfna = data["t_fna"][mask]
        ffna = data["f_fna"][mask]
        mv = data["mean_v"][mask]

        if n.size == 0:
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            x1 = np.where(n > 0, tfna / n, np.nan)
            y1 = mv
            x2 = np.where(tfna > 0, mv * n / tfna, np.nan)
            ffna_clip = np.clip(ffna, 1e-12, None)
            y2 = np.where(n > 0, ffna_clip ** (1.0 / np.where(n > 0, n, 1.0)), np.nan)

        mask1 = np.isfinite(x1) & np.isfinite(y1)
        mask2 = np.isfinite(x2) & np.isfinite(y2)
        if not (mask1.any() or mask2.any()):
            continue

        fig, (ax1, ax2) = plt.subplots(
            1, 2, figsize=(8.0, 3.5), gridspec_kw={"width_ratios": [1.0, 1.0]}
        )
        color = cmap(0)
        color2 = cmap(6)


        if mask1.any():
            ax1.scatter(
                x1[mask1],
                y1[mask1],
                s=20,
                facecolors="none",
                edgecolors=color,
                linewidths=0.8,
            )
            ax1.set_xlabel(r"$T_{\mathrm{FNA}} / N_{\mathrm{steps}}$", fontsize=10)
            ax1.set_ylabel(r"Mean velocity (steps/s)", fontsize=10)
            ax1.set_title("Timescale per step vs mean velocity", fontsize=11)
            _setup_axes(ax1)
            _add_linear_fit_1(ax1, x1[mask1], y1[mask1], color2, label_prefix="")
            _pad_axes(ax1)
            print(x1[mask1], y1[mask1])

        if mask2.any():
            ax2.scatter(
                x2[mask2],
                y2[mask2],
                s=20,
                facecolors="none",
                edgecolors=color,
                linewidths=0.8,
            )
            ax2.set_xlabel(
                r"$v_{\mathrm{mean}} \cdot N_{\mathrm{steps}} / T_{\mathrm{FNA}}$",
                fontsize=10,
            )
            ax2.set_ylabel(r"$f_{\mathrm{FNA}}^{1/N_{\mathrm{steps}}}$", fontsize=10)
            ax2.set_title("Effective rate vs per-step binding prob.", fontsize=11)
            _setup_axes(ax2)
            _add_linear_fit_2(ax2, x2[mask2], y2[mask2], color2, label_prefix="")
            _pad_axes(ax2)

        fig.suptitle(f"Family: {fam_name}", fontsize=12)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_path = out_dir / f"dwell_param_relations_{fam_name}.png"
        fig.savefig(out_path, dpi=1000)
        plt.close(fig)
        print("Saved:", out_path)


def plot_overview_relationships(data: Dict[str, np.ndarray], out_path: Path):
    """
    Build a stacked-row overview figure (single column), as in the screenshot:
      Rows: one per family
      Plot: v_mean vs (T_FNA / N_steps), with linear fit and R^2 box
    """
    fam = data["family"]
    unique_fam = np.unique(fam)
    n_fam = len(unique_fam)
    if n_fam == 0:
        return

    cmap = plt.get_cmap("tab20c")
    fig, axes = plt.subplots(
        n_fam,
        1,
        figsize=(6.5, 2.0 * n_fam),
        squeeze=False,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0] * n_fam, "hspace": 0.5},
    )

    for row, fam_name in enumerate(unique_fam):
        mask = fam == fam_name
        n = data["n_steps"][mask]
        tfna = data["t_fna"][mask]
        mv = data["mean_v"][mask]
        if n.size == 0:
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(n > 0, tfna / n, np.nan)
            y = mv

        mask_xy = np.isfinite(x) & np.isfinite(y)
        if not mask_xy.any():
            continue

        ax = axes[row, 0]
        color = cmap(0)
        color_fit = cmap(6)
        ax.scatter(
            x[mask_xy],
            y[mask_xy],
            s=18,
            facecolors="none",
            edgecolors=color,
            linewidths=0.8,
        )
        if row == n_fam - 1:
            ax.set_xlabel(r"$T_{\mathrm{FNA}} / N_{\mathrm{steps}}$", fontsize=9)
        ax.set_ylabel(r"$v_{\mathrm{mean}}$", fontsize=9)
        if row == 0:
            ax.set_title(r"$v_{\mathrm{mean}}$ vs $T_{\mathrm{FNA}}/N_{\mathrm{steps}}$", fontsize=9)
        _setup_axes(ax)
        _add_linear_fit_1(ax, x[mask_xy], y[mask_xy], color_fit, label_prefix="")
        _pad_axes(ax, frac=0.15)

        # Family label on left margin
        ax.text(
            -0.14,
            0.5,
            fam_name,
            transform=ax.transAxes,
            fontsize=9,
            va="center",
            ha="right",
        )

    fig.tight_layout(rect=[0.16, 0.05, 0.98, 0.98])
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved overview:", out_path)


def _fit_line(x: np.ndarray, y: np.ndarray):
    """Return slope, intercept, R^2 for y=a*x+b on finite points."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return None
    xf = x[mask]
    yf = y[mask]
    a, b = np.polyfit(xf, yf, 1)
    y_pred = a * xf + b
    ss_res = np.sum((yf - y_pred) ** 2)
    ss_tot = np.sum((yf - np.mean(yf)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return a, b, r2


def plot_overlaid_family_fits(data: Dict[str, np.ndarray], out_dir: Path):
    """
    Create one combined figure with two overlaid-family panels:
      Left:  v_mean vs T_FNA / N_steps
      Right: f_FNA^(1/N_steps) vs v_mean * N_steps / T_FNA
    Each family has its own scatter + linear fit.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fam = data["family"]
    families = np.unique(fam)
    if families.size == 0:
        return

    preferred_order = [
        "Balanced",
        "High_Unwinding",
        "High_Pausing",
        "Low_Unwinding",
        "Family_B",
        "Family_C",
        "Family_D",
    ]
    fam_set = set(families.tolist())
    ordered = [f for f in preferred_order if f in fam_set] + sorted([f for f in families.tolist() if f not in preferred_order])

    cmap = plt.get_cmap("tab20c")
    fam_colors = {f: cmap(i*2) for i, f in enumerate(ordered)}

    # -------- Combined Figure: eq1 + eq2 --------
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(8.4, 4.2), gridspec_kw={"width_ratios": [1.0, 1.0]}
    )
    fit_params_eq1 = []
    for family in ordered:
        m = fam == family
        n = data["n_steps"][m]
        tfna = data["t_fna"][m]
        mv = data["mean_v"][m]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(n > 0, tfna / n, np.nan)
            y = mv
        mask_xy = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask_xy):
            continue
        c = fam_colors[family]
        ax1.scatter(x[mask_xy], y[mask_xy], s=20, facecolors="none", edgecolors=c, linewidths=0.8)
        fit = _fit_line(x[mask_xy], y[mask_xy])
        if fit is not None:
            a, b, r2 = fit
            fit_params_eq1.append((family, a, b, r2, c))

    ax1.set_xlabel(r"$T_{\mathrm{FNA}} / N_{\mathrm{steps}}$", fontsize=10)
    ax1.set_ylabel(r"$v_{\mathrm{mean}}$ (steps/s)", fontsize=10)
    ax1.set_title(r"Overlay by family: $v_{\mathrm{mean}}$ vs $T_{\mathrm{FNA}}/N_{\mathrm{steps}}$", fontsize=11)
    _setup_axes(ax1)
    _pad_axes(ax1, frac=0.12)
    # Draw each family fit across the full visible x-range (entire frame)
    xlim1 = ax1.get_xlim()
    xx1 = np.linspace(xlim1[0], xlim1[1], 200)
    for family, a, b, r2, c in fit_params_eq1:
        yy1 = a * xx1 + b
        ax1.plot(xx1, yy1, color=c, linestyle="--", linewidth=1.0, label=f"{family} (R²={r2:.3f})")
    leg1 = ax1.legend(loc="best", fontsize=8, frameon=True)
    if leg1 is not None:
        leg1.get_frame().set_edgecolor("0.7")
        leg1.get_frame().set_linewidth(0.7)

    fit_params_eq2 = []
    for family in ordered:
        m = fam == family
        n = data["n_steps"][m]
        tfna = data["t_fna"][m]
        ffna = data["f_fna"][m]
        mv = data["mean_v"][m]
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.where(tfna > 0, mv * n / tfna, np.nan)
            ffna_clip = np.clip(ffna, 1e-12, None)
            y = np.where(n > 0, ffna_clip ** (1.0 / np.where(n > 0, n, 1.0)), np.nan)
        mask_xy = np.isfinite(x) & np.isfinite(y)
        if not np.any(mask_xy):
            continue
        c = fam_colors[family]
        ax2.scatter(x[mask_xy], y[mask_xy], s=20, facecolors="none", edgecolors=c, linewidths=0.8)
        fit = _fit_line(x[mask_xy], y[mask_xy])
        if fit is not None:
            a, b, r2 = fit
            fit_params_eq2.append((family, a, b, r2, c))

    ax2.set_xlabel(r"$v_{\mathrm{mean}} \cdot N_{\mathrm{steps}} / T_{\mathrm{FNA}}$", fontsize=10)
    ax2.set_ylabel(r"$f_{\mathrm{FNA}}^{1/N_{\mathrm{steps}}}$", fontsize=10)
    ax2.set_title(r"Overlay by family: $f_{\mathrm{FNA}}^{1/N}$ vs $v_{\mathrm{mean}}N/T_{\mathrm{FNA}}$", fontsize=11)
    _setup_axes(ax2)
    _pad_axes(ax2, frac=0.12)
    # Draw each family fit across the full visible x-range (entire frame)
    xlim2 = ax2.get_xlim()
    xx2 = np.linspace(xlim2[0], xlim2[1], 200)
    for family, a, b, r2, c in fit_params_eq2:
        yy2 = a * xx2 + b
        ax2.plot(xx2, yy2, color=c, linestyle="--", linewidth=1.0, label=f"{family} (R²={r2:.3f})")
    leg2 = ax2.legend(loc="best", fontsize=8, frameon=True)
    if leg2 is not None:
        leg2.get_frame().set_edgecolor("0.7")
        leg2.get_frame().set_linewidth(0.7)

    fig.tight_layout()
    out_combined = out_dir / "dwell_param_overlay_fit_eq1_eq2.png"
    fig.savefig(out_combined, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_combined)


def plot_parameter_histograms_by_family(data: Dict[str, np.ndarray], out_path: Path):
    """
    Multi-panel 2D column (bar) distribution of fitted parameters across families.
    For each parameter panel:
      - x-axis is grouped by family
      - each family gets exactly 5 bar slots (one per replicate/value)
      - bar height is the exact fitted value (no histogram binning)
    """
    families = np.unique(data["family"])
    if families.size == 0:
        return

    # Requested family ordering
    preferred_order = [
        "Balanced",
        "High_Unwinding",
        "High_Pausing",
        "Low_Unwinding",
        "Family_B",
        "Family_C",
        "Family_D",
    ]
    fam_set = set(families.tolist())
    ordered_families = [f for f in preferred_order if f in fam_set]
    # Append any unexpected family labels at the end (stable sorted)
    ordered_families.extend(sorted([f for f in families.tolist() if f not in set(ordered_families)]))
    families = np.array(ordered_families, dtype=object)

    # Preferred parameter order for display
    ordered_params = [
        "f_fna",
        "f_sna",
        "f_vsna",
        "f_llp",
        "t_fna",
        "t_sna",
        "t_vsna",
        "mean_v",
    ]
    param_labels = {
        "n_steps": "N_steps",
        "f_fna": "f_FNA",
        "f_sna": "f_SNA",
        "f_vsna": "f_VSNA",
        "f_llp": "f_LLP",
        "t_fna": "T_FNA",
        "t_sna": "T_SNA",
        "t_vsna": "T_VSNA",
        "mean_v": "Mean velocity",
    }

    params = []
    for p in ordered_params:
        if p in data and np.isfinite(np.asarray(data[p], dtype=float)).any():
            params.append(p)
    if not params:
        return

    n_params = len(params)
    ncols = 3
    nrows = int(np.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(8.5, 2.6 * nrows), squeeze=False)
    # Custom gradient: green -> blue -> "#5B6498"
    from matplotlib.colors import LinearSegmentedColormap
    blue_cmap = LinearSegmentedColormap.from_list(
        "green_blue_5B6498",
        # ["#73aa78", "#65b2d8", "#5B6498"],
        ["#c7692a", "#65b2d8", "#5B6498"],
    )
    # Discrete 7-step palette (one color per family)
    discrete_steps = np.linspace(0.0, 1.0, 7)
    discrete_colors = [blue_cmap(v) for v in discrete_steps]
    fam_colors = {fam: discrete_colors[i % 7] for i, fam in enumerate(families)}

    # Fixed grouped layout: 5 slots per family + 1 slot gap between families
    group_size = 5
    group_gap = 1

    for idx, p in enumerate(params):
        r = idx // ncols
        c = idx % ncols
        ax = axes[r, c]
        # Build bars in grouped x-axis layout
        xticks = []
        xticklabels = []
        x_cursor = 1
        max_x = 0
        for fam in families:
            fam_mask = data["family"] == fam
            vals = np.asarray(data[p][fam_mask], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                # still reserve a block for consistent family spacing
                vals = np.array([], dtype=float)

            # Use up to 5 values; pad missing with NaN so each family has 5 slots
            vals = vals[:group_size]
            if vals.size < group_size:
                vals = np.concatenate([vals, np.full(group_size - vals.size, np.nan)])

            x_block = np.arange(x_cursor, x_cursor + group_size)
            finite_mask = np.isfinite(vals)
            if np.any(finite_mask):
                ax.bar(
                    x_block[finite_mask],
                    vals[finite_mask],
                    width=0.65,
                    color=fam_colors[fam],
                    edgecolor=fam_colors[fam],
                    linewidth=0.6,
                    alpha=0.95,
                )

            # Family label at block center
            xticks.append(x_cursor + (group_size - 1) / 2.0)
            xticklabels.append(fam)
            max_x = x_cursor + group_size - 1
            x_cursor += group_size + group_gap

        ax.set_title(param_labels.get(p, p), fontsize=10)
        ax.set_xlim(0.5, max_x + 0.5 if max_x > 0 else 1.5)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("Value", fontsize=9)
        _setup_axes(ax)
        ax.tick_params(axis="both", labelsize=8)

    # Hide any empty trailing panels
    for idx in range(n_params, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        axes[r, c].set_visible(False)

    fig.suptitle("Fitted parameter values by family (5 bars per family)", fontsize=11)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    fig.savefig(out_path, dpi=1000, bbox_inches="tight")
    plt.close(fig)
    print("Saved grouped bar overview:", out_path)


def main():
    import argparse

    p = argparse.ArgumentParser(
        description="Plot dwell-time fit parameter relationships vs mean velocity, per family."
    )
    p.add_argument(
        "param_file",
        nargs="?",
        default="dwell_time_param.txt",
        help="Path to dwell_time_param.txt (default: dwell_time_param.txt in current directory).",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: <param_file_dir>/dwell_param_plots).",
    )
    args = p.parse_args()

    param_path = Path(args.param_file)
    data = _load_param_table(param_path)

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = param_path.parent / "dwell_param_plots"

    plot_family_relationships(data, out_dir=out_dir)
    # Plus one combined overview figure with all families and fits
    overview_path = out_dir / "dwell_param_relations_overview.png"
    plot_overview_relationships(data, out_path=overview_path)
    # Overlaid family fits: two figures, one for each equation
    plot_overlaid_family_fits(data, out_dir=out_dir)
    # Plus histogram comparison across families for fitted parameters
    hist_path = out_dir / "dwell_param_histograms_by_family.png"
    plot_parameter_histograms_by_family(data, out_path=hist_path)


if __name__ == "__main__":
    main()

