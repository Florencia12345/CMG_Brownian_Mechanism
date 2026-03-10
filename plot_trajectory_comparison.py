"""
Compare trajectories from different transition-matrix runs (e.g. High_Pausing, High_Pause_2, _3, _4, _5).
Plots velocity (main), position, and state for each run on the same diagram.
Uses parse_file_like_original from visualize_2.py to read trajectory files.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from visualize_2 import parse_file_like_original


# Default run folders to compare (subfolders under base_dir)
DEFAULT_RUN_FOLDERS = [
    '/Users/vivian/Desktop/Undergrad Study/Part C Project/code/sim_traces_nbin_200/run_Low_Unwinding_5',
    '/Users/vivian/Desktop/Undergrad Study/Part C Project/code/sim_traces_nbin_200/run_Low_Unwinding_4',
    '/Users/vivian/Desktop/Undergrad Study/Part C Project/code/sim_traces_nbin_200/run_Low_Unwinding_3',
    '/Users/vivian/Desktop/Undergrad Study/Part C Project/code/sim_traces_nbin_200/run_Low_Unwinding_2',
    '/Users/vivian/Desktop/Undergrad Study/Part C Project/code/sim_traces_nbin_200/run_Low_Unwinding'
]


def load_trajectory(path):
    """
    Load one trajectory file via parse_file_like_original.
    Returns (time, position, state) arrays.
    """
    headers, data_cols = parse_file_like_original(str(path))
    if headers is None or data_cols is None or len(data_cols) < 2:
        return None
    n = min(len(c) for c in data_cols)
    if n == 0:
        return None
    time = np.array(data_cols[0][:n], dtype=float)
    position = np.array(data_cols[1][:n], dtype=float)
    state = np.array(data_cols[2][:n], dtype=float) if len(data_cols) > 2 else np.zeros(n)
    mask = np.isfinite(time) & np.isfinite(position)
    time = time[mask]
    position = position[mask]
    state = state[mask] if state.size == n else np.zeros_like(time)
    if len(time) < 2:
        return None
    return time, position, state


def compute_velocity(time, position):
    """Velocity = d(position)/d(time), same length as time (last value repeated)."""
    dt = np.diff(time)
    if len(dt) == 0:
        return np.zeros_like(position)
    dt = np.where(dt > 0, dt, np.nan)
    if not np.any(np.isfinite(dt)):
        return np.zeros_like(position)
    dt_safe = np.where(np.isfinite(dt), dt, np.nanmedian(dt))
    v = np.diff(position) / dt_safe
    return np.concatenate([v, [v[-1]]])


def load_run_summaries(base_dir, run_folders, pattern="cmg_tra_*.txt"):
    """
    For each run folder, load *all* trajectory files matching pattern.
    Returns list of (run_name, rep_time, rep_position, all_states, mean_velocity_over_all_trajectories).
    The first successfully loaded trajectory is used as the representative position trace.
    """
    base = Path(base_dir)
    out = []
    for run_name in run_folders:
        folder = base / run_name
        if not folder.is_dir():
            print(f"Skip {run_name}: not a directory")
            continue
        files = sorted(folder.glob(pattern))
        if not files:
            print(f"Skip {run_name}: no files matching {pattern}")
            continue
        rep_time = None
        rep_pos = None
        all_states_list = []
        all_vel_list = []
        for path in files:
            result = load_trajectory(path)
            if result is None:
                print(f"Skip {path}: could not load")
                continue
            time, position, state = result
            vel = compute_velocity(time, position)
            if rep_time is None:
                rep_time, rep_pos = time, position
            all_states_list.append(state.astype(int))
            all_vel_list.append(vel)
        if rep_time is None or not all_states_list or not all_vel_list:
            print(f"Skip {run_name}: no valid trajectories loaded")
            continue
        all_states = np.concatenate(all_states_list)
        all_vel = np.concatenate(all_vel_list)
        mean_v = float(np.mean(all_vel)) if all_vel.size > 0 else 0.0
        out.append((run_name, rep_time, rep_pos, all_states, mean_v))
    return out


def plot_trajectory_comparison(
    base_dir=None,
    run_folders=None,
    pattern="cmg_tra_*.txt",
    figsize=(14, 8),
    show_mean_velocity_in_legend=True,
    save_path=None,
    show=True,
):
    """
    Plot comparison of trajectories from different transition matrices.
    Figure 1: position vs time (legend shows mean velocity for each run).
    Figure 2: separate state-occupancy histogram, with each state filled in a different color.
    """
    base_dir = Path(base_dir) if base_dir else Path(__file__).parent
    run_folders = run_folders or DEFAULT_RUN_FOLDERS

    data = load_run_summaries(base_dir, run_folders, pattern=pattern)
    if not data:
        print("No trajectory data loaded. Check base_dir and run_folders.")
        return None

    # Figure 1: position vs time
    fig_pos, ax_pos = plt.subplots(1, 1, figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(data)))
    if len(data) > 10:
        colors = plt.cm.tab20(np.linspace(0, 1, len(data)))

    for i, (run_name, time, position, state_all, mean_v) in enumerate(data):
        c = colors[i % len(colors)]
        # Legend now attached to position plot, with mean velocity in label
        label = f"{run_name}  (v_mean={mean_v:.3f})" if show_mean_velocity_in_legend else run_name

        ax_pos.plot(time, position, "-", color=c, label=label, linewidth=1.2, alpha=0.9)

    ax_pos.set_ylabel("Position", fontsize=11)
    ax_pos.set_title("Position (legend shows mean velocity)", fontsize=12)
    ax_pos.legend(loc="upper right", fontsize=8)
    ax_pos.grid(True, alpha=0.4)
    fig_pos.tight_layout()

    # Figure 2: state occupancy histogram (stacked bars, one bar per run, each state different color)
    fig_state, ax_state = plt.subplots(1, 1, figsize=(figsize[0], figsize[1] * 0.6))
    # Determine unique states across all runs
    all_states = np.concatenate([s for (_, _, _, s, _) in data])
    unique_states = np.unique(all_states.astype(int))
    unique_states = np.sort(unique_states)
    n_runs = len(data)
    x = np.arange(n_runs)

    # High-contrast, clearly distinct colors per state (easy to tell apart when stacked)
    base_state_colors = [
        "#2166ac",   # strong blue
        "#67a9cf",   # light blue
        "#f4a582",   # orange/salmon
        "#ca0020",   # red
        "#762a83",   # purple
    ]
    state_colors = {state: base_state_colors[i % len(base_state_colors)] for i, state in enumerate(unique_states)}

    bottoms = np.zeros(n_runs)
    for state_val in unique_states:
        counts = []
        for (_, _, _, state, _) in data:
            counts.append(np.sum(state.astype(int) == state_val))
        counts = np.array(counts)
        ax_state.bar(
            x,
            counts,
            bottom=bottoms,
            color=state_colors[state_val],
            edgecolor="black",
            linewidth=0.5,
            label=f"state {state_val}",
        )
        bottoms += counts

    ax_state.set_xticks(x)
    ax_state.set_xticklabels([name for (name, _, _, _, _) in data], rotation=45, ha="right", fontsize=9)
    ax_state.set_xlabel("Run", fontsize=11)
    ax_state.set_ylabel("Count", fontsize=11)
    ax_state.set_title("State occupancy histogram (stacked by state)", fontsize=12)
    ax_state.legend(title="State", fontsize=8)
    ax_state.grid(True, axis="y", alpha=0.4)
    fig_state.tight_layout()

    if save_path:
        fig_pos.savefig(save_path, dpi=150, bbox_inches="tight")
        fig_state.savefig(str(Path(save_path).with_name(Path(save_path).stem + "_states.png")), dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return (fig_pos, fig_state)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Compare trajectories from different transition matrices (velocity, position, state).")
    p.add_argument("base_dir", nargs="?", default=None, help="Base directory containing run_* folders (default: script dir)")
    p.add_argument("--runs", nargs="+", default=None, help="Run folder names, e.g. run_High_Pausing run_High_Pause_2 run_High_Pause_3")
    p.add_argument("--pattern", default="cmg_tra_*.txt", help="Glob for trajectory files inside each run folder")
    p.add_argument("--save", type=str, default=None, help="Save figure path")
    p.add_argument("--no-show", action="store_true", help="Do not show plot")
    args = p.parse_args()

    run_folders = args.runs or DEFAULT_RUN_FOLDERS
    plot_trajectory_comparison(
        base_dir=args.base_dir,
        run_folders=run_folders,
        pattern=args.pattern,
        save_path=args.save,
        show=not args.no_show,
    )
