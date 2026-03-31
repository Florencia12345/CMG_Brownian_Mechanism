"""
Recompute dwell times by directly calling `visualize_2.dwell_time_analysis`
with a fixed `box_height` window size.

Input:
  ./sim_traces_nbin_250/run_*/cmg_tra_*.txt

Output:
  ./dwell_time_all_fixed_window/<run_name>/dwell_times_all.txt
"""

from pathlib import Path
import argparse
import re
import numpy as np

from visualize_2 import parse_file_like_original
from visualize_2 import dwell_time_analysis as _visualize2_dwell_time_analysis


def canonical_family_from_run(run_name: str) -> str:
    """
    Map run folder names (e.g. run_High_Unwinding_2_noise_0.5) to a family key.
    """
    name = run_name.strip()
    if name.lower().startswith("run_"):
        name = name[4:]
    # remove trailing noise suffix
    name = re.sub(r"_noise_[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$", "", name)
    # remove trailing replicate index
    name = re.sub(r"_\d+$", "", name)
    return name


def load_trajectory(path: Path, time_col: int = 0, pos_col: int = 1):
    """Load one trajectory file and return finite, time-sorted (t, x)."""
    headers, data_cols = parse_file_like_original(str(path))
    if headers is None or data_cols is None:
        return None
    if len(data_cols) <= max(time_col, pos_col):
        return None

    n = min(len(c) for c in data_cols)
    if n < 2:
        return None
    t = np.asarray(data_cols[time_col][:n], dtype=float)
    x = np.asarray(data_cols[pos_col][:n], dtype=float)

    mask = np.isfinite(t) & np.isfinite(x)
    t = t[mask]
    x = x[mask]
    if t.size < 2:
        return None

    # Ensure monotonic-time ordering and remove duplicate timestamps
    order = np.argsort(t)
    t = t[order]
    x = x[order]
    uniq_t, uniq_idx = np.unique(t, return_index=True)
    t = uniq_t
    x = x[uniq_idx]
    if t.size < 2 or t[-1] <= t[0]:
        return None
    return t, x

def process_run_folder(run_dir: Path, box_height: float):
    """Run visualize_2.dwell_time_analysis on each trajectory and concatenate dwell times."""
    traj_files = sorted(run_dir.glob("cmg_tra_*.txt"))
    all_dwell = []
    for fp in traj_files:
        traj = load_trajectory(fp)
        if traj is None:
            continue
        t, x = traj
        dwell_times = _visualize2_dwell_time_analysis(
            y=x,
            filepath=str(fp),
            x=t,
            box_height=box_height,
            plot=False,
            show=False,
        )
        dwell_times = np.asarray(dwell_times, dtype=float)
        if dwell_times.size:
            all_dwell.append(dwell_times)
    if not all_dwell:
        return np.array([], dtype=float)
    return np.concatenate(all_dwell)


def save_dwell_file(path: Path, dwell_times: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("dwell_time\n")
        for v in dwell_times:
            f.write(f"{float(v):.10g}\n")


def main():
    p = argparse.ArgumentParser(
        description="Recompute dwell-time files using visualize_2.dwell_time_analysis with fixed box_height."
    )
    p.add_argument(
        "sim_dir",
        nargs="?",
        default="./sim_traces_nbin_250",
        help="Simulation directory containing run_* folders (default: ./sim_traces_nbin_250).",
    )
    p.add_argument(
        "--out-dir",
        default="./dwell_time_all_fixed_window",
        help="Output root directory for recomputed dwell times.",
    )
    p.add_argument(
        "--box-height",
        type=float,
        default=2.5,
        help="Fixed dwell-time window size used by visualize_2.dwell_time_analysis.",
    )
    args = p.parse_args()

    sim_dir = Path(args.sim_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted([d for d in sim_dir.glob("run_*") if d.is_dir()])
    if not run_dirs:
        raise FileNotFoundError(f"No run_* folders found in {sim_dir}")

    print(f"Input sim directory: {sim_dir}")
    print(f"Output directory:    {out_root}")
    print(f"box_height={args.box_height}")

    family_dwell = {}

    for run_dir in run_dirs:
        dwell_all = process_run_folder(run_dir, box_height=args.box_height)
        out_path = out_root / run_dir.name / "dwell_times_all.txt"
        save_dwell_file(out_path, dwell_all)
        print(f"[{run_dir.name}] n_dwell={dwell_all.size} -> {out_path}")

        fam = canonical_family_from_run(run_dir.name)
        family_dwell.setdefault(fam, []).append(dwell_all)

    # Also save one merged dwelltime file per family across all runs in that family.
    for fam, pieces in sorted(family_dwell.items()):
        valid = [np.asarray(p, dtype=float) for p in pieces if np.asarray(p).size > 0]
        if valid:
            fam_all = np.concatenate(valid)
        else:
            fam_all = np.array([], dtype=float)
        fam_path = out_root / fam / "dwelltime.txt"
        save_dwell_file(fam_path, fam_all)
        print(f"[family: {fam}] n_dwell={fam_all.size} -> {fam_path}")


if __name__ == "__main__":
    main()

