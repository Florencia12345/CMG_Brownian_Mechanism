"""
python plot_trajectories_together.py ./sim_traces_nbin_40/run_Balanced --pattern "cmg_tra_4*" 
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from visualize_2 import parse_file_like_original


def load_trajectory_from_parsed(headers, data_cols):
    """
    Convert (headers, data_cols) from parse_file_like_original into a trajectory array.
    Column 0 = time, column 1 = position (or first signal). Extra columns (e.g. state) kept if present.
    Returns: np.ndarray of shape (n_steps, n_cols), or None if invalid.
    """
    if headers is None or data_cols is None or len(data_cols) == 0:
        return None
    n = min(len(c) for c in data_cols)
    if n == 0:
        return Nonebp
    cols = [np.asarray(c[:n], dtype=float) for c in data_cols]
    return np.column_stack(cols)


def load_trajectories_from_folder(folder, pattern="traj_*.txt"):
    """
    Find all files matching pattern in folder, parse each with parse_file_like_original,
    and return a list of (filename, trajectory_array). Skips files that fail to parse or have no data.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    files = sorted(folder.glob(pattern))
    out = []
    for path in files:
        headers, data_cols = parse_file_like_original(str(path))
        traj = load_trajectory_from_parsed(headers, data_cols)
        if traj is not None and len(traj) > 0:
            out.append((path.name, traj))
        else:
            print(f"Skipping {path.name}: empty or could not parse.")
    return out


def plot_trajectories_together(
    folder=None,
    pattern="traj_*.txt",
    time_col=0,
    position_col=1,
    state_col=2,
    show_state=False,
    alpha=0.85,
    figsize=(6.5, 4.0),
    title=None,
    xlabel="Time (s)",
    ylabel="Position (steps)",
    show=True,
    save_path=None,
):
    """
    Load all traj files in folder and plot them on one figure.
    Each trajectory gets a distinct color; legend shows filename.

    folder: path to directory containing traj_xx.txt (default: script directory)
    pattern: glob for trajectory files (default "traj_*.txt")
    time_col, position_col: column indices for time and position (0-based)
    state_col: column index for state (used only if show_state=True)
    show_state: if True and column exists, color segments by state (optional; else simple line plot)
    """
    if folder is None:
        folder = Path(__file__).parent
    folder = Path(folder)
    trajectories = load_trajectories_from_folder(folder, pattern=pattern)
    if not trajectories:
        print(f"No trajectories loaded from {folder} with pattern '{pattern}'.")
        return None

    fig, ax = plt.subplots(figsize=figsize)
    n_traj = len(trajectories)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_traj, 1)))
    if n_traj > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, n_traj))

    mean_velocities = []
    for i, (fname, traj) in enumerate(trajectories):
        t = traj[:, time_col]
        y = traj[:, position_col]
        mask = np.isfinite(t) & np.isfinite(y)
        t, y = t[mask], y[mask]
        if len(t) < 2:
            continue
        c = colors[i % len(colors)]
        ax.plot(t, y, "-", color=c, alpha=alpha, linewidth=0.8)

        # Per-trajectory mean velocity (total displacement / total time)
        dt_tot = t[-1] - t[0]
        if dt_tot > 0:
            v_mean = (y[-1] - y[0]) / dt_tot
            mean_velocities.append(v_mean)

    # Compute collective average velocity across trajectories and show in legend
    if mean_velocities:
        global_mean_v = float(np.mean(mean_velocities))
        from matplotlib.lines import Line2D

        handle = Line2D([0], [0], color="black", lw=1.5)
        ax.legend(
            [handle],
            [f"Mean velocity (all trajectories) = {global_mean_v:.3g} steps/s"],
            loc="best",
            fontsize=9,
        )

    # Default title if not provided: include folder name and pattern
    if title is None:
        title = f"Trajectories in '{folder.name}' (pattern: '{pattern}')"

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    # Determine default save path if none provided:
    # save to "Collections_of_Trajectories/trajectories_<folder-name>.png"
    if save_path is None:
        import os
        script_dir = Path(__file__).parent
        out_dir = script_dir / "Collections_of_Trajectories"
        out_dir.mkdir(parents=True, exist_ok=True)
        # Use the folder name as identifier (e.g. run_Balanced -> trajectories_run_Balanced.png)
        name_slug = folder.name.replace(" ", "_")
        save_path = out_dir / f"trajectories_{name_slug}.png"

    if save_path:
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Plot traj_xx.txt trajectories from a folder on one plot.")
    p.add_argument("folder", nargs="?", default=None, help="Folder containing traj_*.txt files (default: script dir)")
    p.add_argument("--pattern", default="traj_*.txt", help="Glob pattern for trajectory files")
    p.add_argument("--no-show", action="store_true", help="Do not call plt.show()")
    p.add_argument("--save", type=str, default=None, help="Save figure to this path")
    p.add_argument("--show-state", action="store_true", help="Color by state column if present")
    args = p.parse_args()

    plot_trajectories_together(
        folder=args.folder,
        pattern=args.pattern,
        show_state=args.show_state,
        show=not args.no_show,
        save_path=args.save,
    )
