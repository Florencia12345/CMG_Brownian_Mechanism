"""
Summarize trajectory statistics per family from simulated runs.

Example:
    python summarize_family_trajectory_stats.py sim_traces_nbin_250

This script:
  1) Scans run folders under the given base directory (run_*).
  2) Reads trajectory files matching cmg_tra_*.txt.
  3) Computes per-trajectory metrics:
       - mean velocity = (x_end - x_start) / (t_end - t_start)
       - state fractions for:
           unwinding (state 0),
           pausing (state 1),
           backtracking/shortening (state 2)
  4) Aggregates per family (mean ± std across simulations/trajectories).
  5) Writes a tab-separated summary text file.

Output columns:
  family
  mean_velocity
  velocity_uncertainty
  unwinding_fraction
  unwinding_uncertainty
  pausing_fraction
  pausing_uncertainty
  backtracking_fraction
  backtracking_uncertainty
"""

from pathlib import Path
import argparse
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from visualize_2 import parse_file_like_original


def canonical_family_from_run(run_name: str) -> str:
    """Map run folder names to canonical family labels."""
    name = Path(run_name).name
    name = re.sub(r"^[Rr][Uu][Nn]_", "", name)
    # Remove common suffixes
    name = re.sub(r"_noise_[0-9.]+$", "", name)
    name = re.sub(r"_[Tt]est$", "", name)
    name = re.sub(r"_[0-9]+$", "", name)
    key = name.lower()

    if "high_unw" in key:
        return "High_Unwinding"
    if "low_unw" in key:
        return "Low_Unwinding"
    if "high_paus" in key or "high_pause" in key:
        return "High_Pausing"
    if "balanced" in key:
        return "Balanced"
    if "familyb" in key or "family_b" in key:
        return "Family_B"
    if "familyc" in key or "family_c" in key:
        return "Family_C"
    if "familyd" in key or "family_d" in key:
        return "Family_D"
    return name


def load_trajectory(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load one trajectory file and return (time, position, state) arrays."""
    headers, data_cols = parse_file_like_original(str(path))
    if headers is None or data_cols is None or len(data_cols) < 3:
        return None

    n = min(len(c) for c in data_cols)
    if n < 2:
        return None

    t = np.asarray(data_cols[0][:n], dtype=float)
    x = np.asarray(data_cols[1][:n], dtype=float)
    s = np.asarray(data_cols[2][:n], dtype=float)

    mask = np.isfinite(t) & np.isfinite(x) & np.isfinite(s)
    t = t[mask]
    x = x[mask]
    s = s[mask]
    if t.size < 2:
        return None
    return t, x, s


def trajectory_metrics(t: np.ndarray, x: np.ndarray, s: np.ndarray) -> Optional[Dict[str, float]]:
    """Compute per-trajectory mean velocity and state fractions."""
    dt_total = float(t[-1] - t[0])
    if not np.isfinite(dt_total) or dt_total <= 0:
        return None

    v_mean = float((x[-1] - x[0]) / dt_total)

    # States are expected to be 0,1,2 from CMG simulation code
    s_int = np.rint(s).astype(int)
    n = s_int.size
    if n == 0:
        return None

    frac_unw = float(np.mean(s_int == 0))
    frac_pau = float(np.mean(s_int == 1))
    frac_bak = float(np.mean(s_int == 2))  # shortening/backtracking

    return {
        "mean_velocity": v_mean,
        "unwinding_fraction": frac_unw,
        "pausing_fraction": frac_pau,
        "backtracking_fraction": frac_bak,
    }


def summarize_by_family(base_dir: Path, pattern: str = "cmg_tra_*.txt") -> Dict[str, Dict[str, float]]:
    """Aggregate per-trajectory metrics into family-level mean ± std."""
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")

    metrics_by_family: Dict[str, List[Dict[str, float]]] = {}

    run_dirs = sorted([p for p in base_dir.glob("run_*") if p.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run_* folders found under {base_dir}")

    for run_dir in run_dirs:
        family = canonical_family_from_run(run_dir.name)
        files = sorted(run_dir.glob(pattern))
        if not files:
            continue
        for fp in files:
            traj = load_trajectory(fp)
            if traj is None:
                continue
            t, x, s = traj
            m = trajectory_metrics(t, x, s)
            if m is None:
                continue
            metrics_by_family.setdefault(family, []).append(m)

    summary: Dict[str, Dict[str, float]] = {}
    for family, rows in metrics_by_family.items():
        if not rows:
            continue
        v = np.array([r["mean_velocity"] for r in rows], dtype=float)
        u = np.array([r["unwinding_fraction"] for r in rows], dtype=float)
        p = np.array([r["pausing_fraction"] for r in rows], dtype=float)
        b = np.array([r["backtracking_fraction"] for r in rows], dtype=float)

        summary[family] = {
            "n_trajectories": int(len(rows)),
            "mean_velocity": float(np.mean(v)),
            "velocity_uncertainty": float(np.std(v, ddof=1)) if v.size > 1 else 0.0,
            "unwinding_fraction": float(np.mean(u)),
            "unwinding_uncertainty": float(np.std(u, ddof=1)) if u.size > 1 else 0.0,
            "pausing_fraction": float(np.mean(p)),
            "pausing_uncertainty": float(np.std(p, ddof=1)) if p.size > 1 else 0.0,
            "backtracking_fraction": float(np.mean(b)),
            "backtracking_uncertainty": float(np.std(b, ddof=1)) if b.size > 1 else 0.0,
        }
    return summary


def write_summary_table(summary: Dict[str, Dict[str, float]], out_path: Path):
    """Write family summary as TSV text file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preferred_order = [
        "Balanced",
        "High_Unwinding",
        "High_Pausing",
        "Low_Unwinding",
        "Family_B",
        "Family_C",
        "Family_D",
    ]
    fams = list(summary.keys())
    ordered = [f for f in preferred_order if f in fams] + sorted([f for f in fams if f not in preferred_order])

    header = [
        "family",
        "n_trajectories",
        "mean_velocity",
        "velocity_uncertainty",
        "unwinding_fraction",
        "unwinding_uncertainty",
        "pausing_fraction",
        "pausing_uncertainty",
        "backtracking_fraction",
        "backtracking_uncertainty",
    ]

    lines = ["\t".join(header)]
    for family in ordered:
        s = summary[family]
        row = [
            family,
            str(s["n_trajectories"]),
            f"{s['mean_velocity']:.6g}",
            f"{s['velocity_uncertainty']:.6g}",
            f"{s['unwinding_fraction']:.6g}",
            f"{s['unwinding_uncertainty']:.6g}",
            f"{s['pausing_fraction']:.6g}",
            f"{s['pausing_uncertainty']:.6g}",
            f"{s['backtracking_fraction']:.6g}",
            f"{s['backtracking_uncertainty']:.6g}",
        ]
        lines.append("\t".join(row))

    out_path.write_text("\n".join(lines) + "\n")
    print(f"Saved family stats table: {out_path}")


def main():
    p = argparse.ArgumentParser(
        description="Compute family-level trajectory stats (velocity + state fractions) with uncertainties."
    )
    p.add_argument(
        "base_dir",
        nargs="?",
        default="sim_traces_nbin_250",
        help="Directory containing run_* folders (default: sim_traces_nbin_250).",
    )
    p.add_argument(
        "--pattern",
        default="cmg_tra_*.txt",
        help="Trajectory filename pattern inside each run folder.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output txt/tsv path (default: <base_dir>/family_trajectory_stats.txt).",
    )
    args = p.parse_args()

    base_dir = Path(args.base_dir)
    out_path = Path(args.out) if args.out else (base_dir / "family_trajectory_stats.txt")

    summary = summarize_by_family(base_dir, pattern=args.pattern)
    if not summary:
        raise RuntimeError(f"No valid trajectories were processed under {base_dir}")

    write_summary_table(summary, out_path)


if __name__ == "__main__":
    main()

