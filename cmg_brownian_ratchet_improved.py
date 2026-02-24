"""
Discrete-time Markov chain simulation of CMG helicase as a Brownian ratchet motor.
Updated Feb. 24 - Cleanups 

This code simulates DNA unwinding using a discrete-time Markov chain model. 
The helicase acts as a Brownian ratchet motor, 
where thermal fluctuations are rectified by gausian noise and transition matrix due to
ATP binding and hydrolysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List, Union, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from collections import deque

# Use visualize_2 for dwell-time analysis and plotting
try:
    from visualize_2 import (
        dwell_time_analysis as _visualize2_dwell_time_analysis,
        plot_dwell_time_survival as _visualize2_plot_survival,
        plot_first_passage_density as _visualize2_plot_fp_density,
    )
    _HAS_VISUALIZE2_DWELL = True
    _HAS_VISUALIZE2_PLOTS = True
except Exception:
    _HAS_VISUALIZE2_DWELL = False
    _HAS_VISUALIZE2_PLOTS = False
try:
    from visualize_2 import plot_dwell_time_fits as _visualize2_plot_dwell_fits
except Exception:
    _visualize2_plot_dwell_fits = None


def _validate_transition_matrix(T: np.ndarray, n: int = 3) -> np.ndarray:
    """
    Ensure T is (n,n) and row-stochastic; return float array.
    Applied for normalisation.
    """
    T = np.asarray(T, dtype=float)
    if T.shape != (n, n):
        raise ValueError(f"transition_matrix must be {n}x{n}, got {T.shape}")
    row_sums = T.sum(axis=1)
    for i in range(n):
        if row_sums[i] <= 0:
            T[i, :] = 1.0 / n
        else:
            T[i, :] /= row_sums[i]
    return T


@dataclass
class RatchetParameters:
    """Parameters for the Markov state simulation. Input is the 3x3 transition matrix."""
    # P(X_{t+1}=j | X_t=i) = transition_matrix[i,j]; rows must sum to 1
    transition_matrix: np.ndarray = None
    # Simulation mechanics
    step_size: float = 1.0
    dt: float = 1.0
    max_position: int = 10000
    # Gaussian noise on recorded trajectory (applied at record time; 0 = no noise)
    position_noise_std: float = 0.0   # std of N(0, std) added to position (base pairs)
    time_noise_std: float = 0.0      # std of N(0, std) added to time (seconds)

    def __post_init__(self):
        if self.transition_matrix is None:
            # Default: moderate persistence (diagonal 0.8) and equal off-diagonal
            self.transition_matrix = np.array([
                [0.8, 0.1, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.1, 0.8],
            ], dtype=float)
        self.transition_matrix = _validate_transition_matrix(self.transition_matrix, 3)


class CMGBrownianRatchet:
    """
    Discrete-time Markov chain simulation of CMG helicase as a Brownian ratchet.
    The model includes:
    - Motor states: unwinding (forward), pausing, shortening (backward)
    - Asymmetric transition probabilities creating ratchet effect
    """
    
    # State constants
    STATE_UNWINDING = 0
    STATE_PAUSING = 1
    STATE_SHORTENING = 2
    
    STATE_NAMES = ['unwinding', 'pausing', 'shortening']
    # Unwinding=green, Pausing=orange, Shortening=red (match HMM/HSMM analysis)
    STATE_COLORS = ('green', 'orange', 'red')
    
    def __init__(self, params: Optional[RatchetParameters] = None, random_state: Optional[int] = None):
        """
        Initialize the CMG helicase Brownian ratchet simulation.
        """
        self.params = params or RatchetParameters()
        # T[i,j] = P(X_{t+1} = j | X_t = i); simulate by sampling from T[state, :]
        self.transition_matrix = np.asarray(self.params.transition_matrix, dtype=float)
        self.position = 0
        self.state = self.STATE_UNWINDING
        self.time = 0.0
        self.trajectory = []  # (time, position, state, 0) — 4th column kept for compatibility
    
    def step(self) -> Dict[str, float]:
        """
        One step: sample next state from P(X_{t+1}=j | X_t=i) = transition_matrix[i,j].
        """
        transition_info = {
            'forward': False,
            'backward': False,
            'state': self.state,
            'state_name': self.STATE_NAMES[self.state],
            'state_changed': False
        }
        old_state = self.state
        # Sample next state from P(X_{t+1}=j | X_t=i) = transition_matrix[i,j]
        self.state = int(np.random.choice(3, p=self.transition_matrix[self.state, :]))
        transition_info['state'] = self.state
        transition_info['state_name'] = self.STATE_NAMES[self.state]
        if self.state != old_state:
            transition_info['state_changed'] = True
        
        # Position update is deterministic from state.
        # Unwinding → extend; Pausing → same; Shortening → decrease.
        if self.state == self.STATE_UNWINDING:
            if self.position < self.params.max_position:
                self.position += self.params.step_size
                transition_info['forward'] = True
        elif self.state == self.STATE_PAUSING:
            pass  # No position change
        elif self.state == self.STATE_SHORTENING:
            if self.position > 0:
                self.position -= self.params.step_size
                transition_info['backward'] = True
        
        self.time += self.params.dt
        # Trajectory is appended in simulate() when record_every matches, to avoid duplicates
        
        return transition_info
    
    def simulate(self, duration: float, record_interval: Optional[float] = None) -> np.ndarray:
        """
        Run the simulation for a specified duration.
        
        Args:
            duration: Total simulation time (seconds)
            record_interval: Time interval for recording states (None = every step)
        
        Returns:
            Array of (time, position, state, ATP_bound) tuples
        """
        num_steps = int(duration / self.params.dt)
        record_every = 1 if record_interval is None else int(record_interval / self.params.dt)
        
        self.trajectory = []
        
        for i in range(num_steps):
            self.step()
            if i % record_every == 0:
                t = self.time
                p = float(self.position)
                if self.params.time_noise_std > 0:
                    t = t + np.random.normal(0, self.params.time_noise_std)
                    t = max(0.0, t)
                if self.params.position_noise_std > 0:
                    p = p + np.random.normal(0, self.params.position_noise_std)
                    p = max(0.0, p)
                self.trajectory.append((t, p, self.state, 0))
        
        return np.array(self.trajectory)
    
    def reset(self):
        """Reset the simulation to initial state."""
        self.position = 0
        self.state = self.STATE_UNWINDING
        self.time = 0.0
        self.trajectory = []
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Calculate statistics from the trajectory.
        Returns:
            Dictionary with mean velocity, diffusion coefficient, etc.
        """
        if len(self.trajectory) < 2:
            return {}
        
        traj = np.array(self.trajectory)
        times = traj[:, 0]
        positions = traj[:, 1]
        
        # Mean velocity
        if len(times) > 1:
            total_time = times[-1] - times[0]
            total_displacement = positions[-1] - positions[0]
            mean_velocity = total_displacement / total_time if total_time > 0 else 0.0
        else:
            mean_velocity = 0.0
        
        # Mean squared displacement
        if len(positions) > 1:
            msd = np.mean((positions - positions[0])**2)
        else:
            msd = 0.0
        
        # Diffusion coefficient (from MSD = 2Dt)
        if len(times) > 1:
            D = msd / (2 * (times[-1] - times[0])) if (times[-1] - times[0]) > 0 else 0.0
        else:
            D = 0.0
        
        # State fractions
        if len(traj) > 0 and traj.shape[1] > 2:
            states = traj[:, 2]
            unwinding_fraction = np.mean(states == self.STATE_UNWINDING)
            pausing_fraction = np.mean(states == self.STATE_PAUSING)
            shortening_fraction = np.mean(states == self.STATE_SHORTENING)
        else:
            unwinding_fraction = pausing_fraction = shortening_fraction = 0.0
        
        return {
            'mean_velocity': mean_velocity,
            'mean_squared_displacement': msd,
            'diffusion_coefficient': D,
            # 'ATP_bound_fraction': ATP_bound_fraction,
            'unwinding_fraction': unwinding_fraction,
            'pausing_fraction': pausing_fraction,
            'shortening_fraction': shortening_fraction,
            'final_position': positions[-1] if len(positions) > 0 else 0.0,
            'total_time': times[-1] if len(times) > 0 else 0.0
        }


def run_ensemble_simulation(
    n_trajectories: int = 1,
    duration: float = 10.0,
    params: Optional[RatchetParameters] = None,
    random_state: Optional[int] = None,
) -> Tuple[list, Dict[str, float]]:
    """
    Run multiple independent simulations and return ensemble statistics.
    Input arguments of this function:
        n_trajectories: Number of independent trajectories
        duration: Simulation duration for each trajectory
        random_state: Optional seed
    """
    all_trajectories = []
    all_stats = []
    
    for i in range(n_trajectories):
        seed = (random_state + i) if random_state is not None else None
        if seed is not None:
            np.random.seed(seed)
        sim = CMGBrownianRatchet(params, random_state=seed)
        traj = sim.simulate(duration)
        stats = sim.get_statistics()
        
        all_trajectories.append(traj)
        all_stats.append(stats)

    print(all_trajectories, "trajectories simulated.")
    
    # Calculate ensemble averages
    ensemble_stats = {
        'mean_velocity': np.mean([s['mean_velocity'] for s in all_stats]),
        'std_velocity': np.std([s['mean_velocity'] for s in all_stats]),
        'mean_final_position': np.mean([s['final_position'] for s in all_stats]),
        'std_final_position': np.std([s['final_position'] for s in all_stats]),
        # 'mean_ATP_fraction': np.mean([s['ATP_bound_fraction'] for s in all_stats]),
        'mean_unwinding_fraction': np.mean([s['unwinding_fraction'] for s in all_stats]),
        'mean_pausing_fraction': np.mean([s['pausing_fraction'] for s in all_stats]),
        'mean_shortening_fraction': np.mean([s['shortening_fraction'] for s in all_stats]),
    }
    
    return all_trajectories, ensemble_stats


def save_trajectories(
    trajectories,
    output_dir: str,
    prefix: str = "cmg_traj",
) -> None:
    """
    Save each trajectory to a separate text file in `output_dir`.
    """
    from pathlib import Path

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    for i, traj in enumerate(trajectories):
        arr = np.asarray(traj, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue

        fname = outdir / f"{prefix}_{i+1}.txt"
        # If fewer than 4 columns, pad with zeros so header stays consistent
        if arr.shape[1] < 4:
            pad_cols = 4 - arr.shape[1]
            arr = np.hstack([arr, np.zeros((arr.shape[0], pad_cols))])

        np.savetxt(
            fname,
            arr,
            header="time\tposition\tstate\tATP_bound",
            comments="",
        )
        print(f"Saved trajectory {i+1} to: {fname}")


def add_gaussian_noise_to_trajectories(
    trajectories: Union[List[np.ndarray], np.ndarray],
    position_std: float = 0.0,
    time_std: float = 0.0,
    random_state: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Apply Gaussian noise to trajectory arrays (time and/or position columns).
    """
    rng = np.random.default_rng(random_state)
    if isinstance(trajectories, np.ndarray) and trajectories.ndim == 2:
        traj_list = [trajectories]
    elif isinstance(trajectories, np.ndarray):
        traj_list = list(trajectories)
    else:
        traj_list = [np.asarray(t, dtype=float) for t in trajectories]
    out = []
    for arr in traj_list:
        arr = np.asarray(arr, dtype=float).copy()
        if arr.ndim != 2 or arr.shape[1] < 2:
            out.append(arr)
            continue
        if time_std > 0:
            arr[:, 0] += rng.normal(0, time_std, size=arr.shape[0])
            arr[:, 0] = np.maximum(arr[:, 0], 0.0)
        if position_std > 0:
            arr[:, 1] += rng.normal(0, position_std, size=arr.shape[0])
            arr[:, 1] = np.maximum(arr[:, 1], 0.0)
        out.append(arr)
    return out


def save_simulation_run(
    trajectories: Union[List[np.ndarray], np.ndarray],
    params: Optional[RatchetParameters],
    output_dir: Union[str, Path],
    prefix: str = "cmg_traj",
) -> None:

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_trajectories(trajectories, str(outdir), prefix=prefix)
    if params is not None:
        param_path = outdir / "params.json"
        d = asdict(params)
        d["transition_matrix"] = np.asarray(d["transition_matrix"]).tolist()
        with open(param_path, "w") as f:
            json.dump(d, f, indent=2)
        print(f"Saved parameters to: {param_path}")

def compute_empirical_transition_matrix(
    trajectories: Union[List[np.ndarray], np.ndarray],
    n_states: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the empirical state-to-state transition matrix from simulated
    trajectories.

    Return: trans_counts and empricial transition matrix
    """
    if isinstance(trajectories, np.ndarray) and trajectories.dtype == object:
        traj_list = [np.asarray(t, dtype=float) for t in trajectories]
    elif isinstance(trajectories, np.ndarray):
        traj_list = [trajectories]
    else:
        traj_list = [np.asarray(t, dtype=float) for t in trajectories]

    # Concatenate state sequences (same idea as states_concat in HMM analysis)
    state_sequences = []
    for traj in traj_list:
        if traj.ndim != 2 or traj.shape[1] < 3:
            continue
        states = traj[:, 2].astype(int)
        state_sequences.append(states)
    if not state_sequences:
        trans_counts = np.zeros((n_states, n_states))
        trans_matrix = np.ones((n_states, n_states)) / n_states
        return trans_counts, trans_matrix

    states_concat = np.concatenate(state_sequences)

    # Count transitions: from_state -> to_state (same as HMM empirical)
    trans_counts = np.zeros((n_states, n_states))
    for t in range(len(states_concat) - 1):
        from_state = int(states_concat[t])
        to_state = int(states_concat[t + 1])
        if 0 <= from_state < n_states and 0 <= to_state < n_states:
            trans_counts[from_state, to_state] += 1

    # Row-normalize to get transition probabilities
    trans_matrix = np.zeros_like(trans_counts, dtype=float)
    for i in range(n_states):
        row_sum = trans_counts[i, :].sum()
        if row_sum > 0:
            trans_matrix[i, :] = trans_counts[i, :] / row_sum
        else:
            trans_matrix[i, :] = 1.0 / n_states

    return trans_counts, trans_matrix


def get_theoretical_transition_matrix(params: Optional[RatchetParameters] = None) -> np.ndarray:
    """Return the 3x3 transition matrix from params"""
    p = params or RatchetParameters()
    return np.asarray(p.transition_matrix, dtype=float)


def plot_transition_matrix(
    transmat: np.ndarray,
    out_path: Union[str, Path],
    title: str = "Transition Matrix",
    state_names: Optional[List[str]] = None,
) -> None:
    """
    Plot and save a state transition matrix as a heatmap.
    """
    n_states = transmat.shape[0]
    if state_names is None:
        state_names = CMGBrownianRatchet.STATE_NAMES[:n_states]
    state_names = list(state_names)[:n_states]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(transmat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Transition probability", fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(n_states))
    ax.set_yticks(np.arange(n_states))
    ax.set_xticklabels([f"{state_names[j]}" for j in range(n_states)], fontsize=10)
    ax.set_yticklabels([f"{state_names[i]}" for i in range(n_states)], fontsize=10)
    ax.set_xlabel("To state", fontsize=12, fontweight="bold")
    ax.set_ylabel("From state", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

    thresh = transmat.max() / 2.0
    for i in range(n_states):
        for j in range(n_states):
            text_color = "white" if transmat[i, j] > thresh else "black"
            ax.text(j, i, f"{transmat[i, j]:.3f}", ha="center", va="center",
                    color=text_color, fontsize=11, fontweight="bold")

    ax.set_xticks(np.arange(n_states + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_states + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)
    ax.tick_params(which="minor", size=0)

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved transition matrix plot: {out_path}")


# actually this function after the latest update in code is not really useful.  
def _concatenate_trajectories_signal(trajectories: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Build (time, position) from a list of trajectories, with time offset so time is monotonic."""
    times_list = []
    positions_list = []
    t_end = 0.0
    for traj in trajectories:
        arr = np.asarray(traj, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            continue
        t = arr[:, 0]
        y = arr[:, 1]
        times_list.append(t + t_end)
        positions_list.append(y)
        if len(t) > 0:
            t_end = float(times_list[-1][-1])
    if not times_list:
        return np.array([]), np.array([])
    return np.concatenate(times_list), np.concatenate(positions_list)


# note for myself here, the dwell time from states is different than the normal dwelltime analysis with thresholds
def extract_dwell_times_from_states(states: np.ndarray) -> np.ndarray:
    """
    Extract dwell times (run lengths in time steps) from a state sequence.
    """
    states = np.asarray(states, dtype=int).ravel()
    if len(states) == 0:
        return np.array([], dtype=float)
    dwells = []
    current = states[0]
    run_len = 1
    for s in states[1:]:
        if s == current:
            run_len += 1
        else:
            dwells.append(run_len)
            current = s
            run_len = 1
    dwells.append(run_len)
    return np.array(dwells, dtype=float)


# keep this function here, but haven't written a plt functin for t
def _empirical_hazard(dwell_times: np.ndarray, n_bins: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute empirical hazard h(t) = f(t)/S(t).
    Exponential distribution has constant hazard h(t) = λ.
    """
    dwell_times = np.asarray(dwell_times, dtype=float)
    dwell_times = dwell_times[np.isfinite(dwell_times) & (dwell_times > 0)]
    n = len(dwell_times)
    if n < 2:
        return np.array([]), np.array([]), np.array([])
    t_min = max(dwell_times.min(), 1e-10)
    t_max = dwell_times.max()
    bins = np.linspace(t_min, t_max, n_bins + 1)
    hist, edges = np.histogram(dwell_times, bins=bins)
    width = (edges[1] - edges[0]) if len(edges) > 1 else 1.0
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    # Density f(t) = (count in bin) / (n * width)
    f = hist / (n * width)
    # Survival S(t) = P(T > t) = proportion of observations > t; at bin center use S(center)
    S = np.array([np.sum(dwell_times > c) / n for c in bin_centers], dtype=float)
    # Avoid division by zero: hazard only where S > 0
    valid = S > 0
    h = np.zeros_like(f)
    h[valid] = f[valid] / S[valid]
    return bin_centers, h, S


def run_dwell_time_analysis(
    trajectories: List[np.ndarray],
    output_dir: Union[str, Path],
    dt: float = 1.0,
) -> np.ndarray:
    """
    Run dwell-time analysis (boundary-crossing from visualize_2 when available),
    and plot survival + first-passage density. Saves figures into output_dir.
    Returns dwell_times : 1D array of dwell times
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    time_arr, position_arr = _concatenate_trajectories_signal(trajectories)
    if position_arr.size == 0:
        print("No trajectory data for dwell-time analysis.")
        return np.array([], dtype=float)

    used_visualize2 = False
    if _HAS_VISUALIZE2_DWELL:
        (output_dir / "plots").mkdir(parents=True, exist_ok=True)
        filepath = str(output_dir / "dwell_times" )
        print('filepath', filepath)
        dwell_times = _visualize2_dwell_time_analysis(
            position_arr,
            filepath=filepath,
            x=time_arr,
            plot=True,
            show=False,
        )
        dwell_times = np.asarray(dwell_times, dtype=float)

    return dwell_times


def plot_trajectory(trajectory: np.ndarray, ax: Optional[plt.Axes] = None, 
                   label: Optional[str] = None, show_state: bool = True):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    times = trajectory[:, 0]
    positions = trajectory[:, 1]
    
    if show_state and trajectory.shape[1] > 2:
        states = trajectory[:, 2].astype(int)
        state_names = CMGBrownianRatchet.STATE_NAMES
        state_colors = CMGBrownianRatchet.STATE_COLORS
        # Color segment (i -> i+1) by state at i+1: that state caused the move to positions[i+1]
        for i in range(len(times) - 1):
            s = min(int(states[i + 1]), len(state_colors) - 1)
            ax.plot(times[i:i+2], positions[i:i+2], color=state_colors[s], alpha=0.7, linewidth=1)
        for s, name in enumerate(state_names):
            ax.plot([], [], color=state_colors[s], label=name.capitalize(), linewidth=2)
    else:
        ax.plot(times, positions, label=label, linewidth=2)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Position (base pairs)', fontsize=12)
    ax.set_title('CMG Helicase Brownian Ratchet Trajectory', fontsize=14)
    ax.grid(True, alpha=0.3)
    if label or show_state:
        ax.legend()
    
    return ax


def plot_ensemble(trajectories, ax: Optional[plt.Axes] = None, label: Optional[str] = None,
                 show_individual: bool = True, color: Optional[str] = None):
    """
    Plot multiple trajectories with mean and standard deviation.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    
    # Convert to list if it's an object array
    if isinstance(trajectories, np.ndarray) and trajectories.dtype == object:
        traj_list = [np.asarray(traj) for traj in trajectories]
    else:
        traj_list = [np.asarray(traj) for traj in trajectories]
    
    if len(traj_list) == 0:
        return ax
    
    # Find common time points (use first trajectory as reference)
    ref_traj = np.asarray(traj_list[0])
    ref_times = ref_traj[:, 0]
    
    # Interpolate all trajectories to common time points
    all_positions = []
    for traj in traj_list:
        traj = np.asarray(traj)
        times = traj[:, 0]
        positions = traj[:, 1]
        interp_positions = np.interp(ref_times, times, positions)
        all_positions.append(interp_positions)
    
    all_positions = np.array(all_positions)
    mean_pos = np.mean(all_positions, axis=0)
    std_pos = np.std(all_positions, axis=0)
    
    # Plot individual trajectories (transparent) - only if requested
    if show_individual:
        for traj in traj_list:
            traj = np.asarray(traj)
            times = traj[:, 0]
            positions = traj[:, 1]
            ax.plot(times, positions, alpha=0.15, color='gray', linewidth=0.3)
    
    # Plot mean and std
    plot_label = label if label else 'Mean'
    line = ax.plot(ref_times, mean_pos, linewidth=2.5, label=plot_label)
    if color is not None:
        line[0].set_color(color)
    
    # Only show fill_between if not comparing multiple sets (to reduce clutter)
    if show_individual:
        ax.fill_between(ref_times, mean_pos - std_pos, mean_pos + std_pos, 
                        alpha=0.2, label='±1 std' if label is None else None)
    
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Position (base pairs)', fontsize=12)
    if ax.get_title() == '':
        ax.set_title('CMG Helicase Ensemble Trajectories', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax


def main():
    """Run multiple CMG Brownian ratchet simulations with different parameters."""
    
    # Simulation parameters
    duration = 2000.0  # Longer simulation time (seconds)
    n_trajectories = 1  # More trajectories for better statistics
    record_interval = 0.1  # Record every 0.1 seconds
    
    # 3x3 transition matrices: T[i,j] = P(X_{t+1}=j | X_t=i); rows sum to 1
    # States: 0=unwinding, 1=pausing, 2=shortening
    param_sets = [
        {
            "name": "High Unwinding",
            "params": RatchetParameters(
                transition_matrix=np.array([
                    [0.9, 0.05, 0.05],   # from unwinding
                    [0.6, 0.3, 0.1],    # from pausing
                    [0.5, 0.2, 0.3],    # from shortening
                ]),
                dt=1.0,
                max_position=10000,
                position_noise_std=0.5,   # e.g. ±0.5 bp Gaussian noise
                time_noise_std=0.0
            )
        },
        {
            "name": "Low Unwinding",
            "params": RatchetParameters(
                transition_matrix=np.array([
                    [1, 0.3, 0.3],
                    [1, 0.3, 0.3],
                    [1, 0.3, 0.3],
                ]),
                dt=1.0,
                max_position=10000,
                position_noise_std=0.5,   # e.g. ±0.5 bp Gaussian noise
                time_noise_std=0.0
            )
        },
        {
            "name": "High Pausing",
            "params": RatchetParameters(
                transition_matrix=np.array([
                    [0.3, 0.3, 0.3],
                    [0.8, 0.8, 0.3],
                    [0.8, 0.3, 0.3],
                ]),
                dt=1,
                max_position=10000,
                position_noise_std=0.8,   # e.g. ±0.5 bp Gaussian noise
                time_noise_std=0.0
            )
        },
        {
            "name": "Balanced",
            "params": RatchetParameters(
                transition_matrix=np.array([
                    [0.5, 0.05, 0.05],   # from unwinding
                    [0.6, 0.3, 0.1],    # from pausing
                    [0.5, 0.2, 0.3],    # from shortening
                ]),
                dt=1.0,
                max_position=10000,
                position_noise_std=0.5,   # e.g. ±0.5 bp Gaussian noise
                time_noise_std=0.0
            )
        },
    ]
    # Run simulations for each parameter set
    all_results = []
    
    traj = 0
    for param_set in param_sets:
        traj = traj + 1
        print(f"\n{'='*60}")
        print(f"Running simulation: {param_set['name']}")
        print("  3x3 transition matrix (rows = from state, cols = to state):")
        print(param_set["params"].transition_matrix)
        print(f"{'='*60}")
        
        trajectories, ensemble_stats = run_ensemble_simulation(
            n_trajectories=n_trajectories,
            duration=duration,
            params=param_set['params']
        )
        

        out_dir = Path("sim_traces") / f"run_{param_set['name'].replace(' ', '_')}"
        save_simulation_run(
            trajectories,
            param_set['params'],
            output_dir=out_dir,
            prefix=f"cmg_tra_{traj}",
        )

        # Empirical vs theoretical transition matrix (same calculation as HMM analysis)
        trans_counts, trans_emp = compute_empirical_transition_matrix(trajectories, n_states=3)
        trans_theory = get_theoretical_transition_matrix(param_set['params'])
        print(f"  Empirical transition matrix (from simulated states):")
        print(trans_emp)
        print(f"  Theoretical transition matrix:")
        print(trans_theory)
        print(f"  Total transitions: {int(trans_counts.sum())}")

        # Plot and save both transition matrices
        plot_transition_matrix(
            trans_emp,
            out_path=out_dir / "transition_matrix_empirical.png",
            title=f"{param_set['name']} — Empirical Transition Matrix",
        )
        plot_transition_matrix(
            trans_theory,
            out_path=out_dir / "transition_matrix_theoretical.png",
            title=f"{param_set['name']} — Theoretical Transition Matrix",
        )


        # Dwell-time analysis: for each trajectory extract dwell times (state-based), then combine and plot All_survival, All_dwell_times_fp_density (like visualize_2 end treatment)
        dt_sec = float(np.median(np.diff(trajectories[0][:, 0]))) if len(trajectories) > 0 and len(trajectories[0]) > 1 else 1.0
        
        dwell_times_list_all = np.array([], dtype=float) 

        for i in range(len(trajectories)):
            output_dir = out_dir / f"traj_{i+1}"
            dwell_times =run_dwell_time_analysis([np.array(trajectories[i])], output_dir=output_dir, dt=dt_sec)
            dwell_times_list_all = np.concatenate([dwell_times_list_all, dwell_times])
            print(len(dwell_times), 'dwell_times')

        dwell_times_per_traj = []
        
        dwell_times_all = np.concatenate(dwell_times_per_traj) if dwell_times_per_traj else np.array([], dtype=float)
        dwell_times_all = dwell_times_all[np.isfinite(dwell_times_all) & (dwell_times_all > 0)]

        dwell_times_all = np.asarray(dwell_times_list_all, dtype=float)

        # save the dwelltimes in the local directory 
        np.savetxt(
            out_dir / "dwell_times_all.txt",
            dwell_times_all,
            header="dwell times",
            comments="",
        )   
        print(len(dwell_times_all))

        _visualize2_plot_survival(dwell_times_list_all, out_dir, show=False)
        _visualize2_plot_fp_density(dwell_times_list_all, out_dir, show=False, n_bins=100)
        # MLE fits (Gamma, Weibull, InvGauss, Exp, Lognorm), AIC/BIC, bootstrap CIs, PDF + survival overlays
        if _visualize2_plot_dwell_fits is not None:
            try:
                _visualize2_plot_dwell_fits(
                    dwell_times_list_all,
                    str(out_dir),
                    n_bins=80,
                    n_bootstrap=200,
                    show=False,
                )
            except Exception as e:
                print(f"  Dwell-time fits skipped: {e}")
        
        all_results.append({
            'name': param_set['name'],
            'params': param_set['params'],
            'trajectories': trajectories,
            'stats': ensemble_stats
        })

        
        print(f"\nEnsemble statistics for {param_set['name']}:")
        for key, value in ensemble_stats.items():
            print(f"  {key}: {value:.4f}")
        
    
    # Plot comparison of all parameter sets
    print("\nGenerating comparison plots...")
    
    # Main comparison plot: all ensembles together
    fig, ax = plt.subplots(figsize=(14, 8))
    # First three match CMG states (unwinding, pausing, shortening); fourth for comparison
    colors = [*CMGBrownianRatchet.STATE_COLORS, 'blue']
    
    for i, result in enumerate(all_results):
        plot_ensemble(result['trajectories'], ax=ax, 
                     label=f"{result['name']} (v={result['stats']['mean_velocity']:.2f} bp/s)",
                     show_individual=False,  # Don't show individual trajectories in comparison
                     color=colors[i % len(colors)])
    
    ax.set_title(f'CMG Helicase: Comparison of Different State Transition Probabilities\n'
                 f'({n_trajectories} trajectories, {duration}s duration)', fontsize=14)
    plt.tight_layout()
    plt.savefig('cmg_ratchet_parameter_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved parameter comparison plot to 'cmg_ratchet_parameter_comparison.png'")
    
    # Plot state fractions for each parameter set
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(all_results):
        ax = axes[i]
        stats = result['stats']
        
        # Create bar plot of state fractions
        states = ['Unwinding', 'Pausing', 'Shortening']
        fractions = [
            stats['mean_unwinding_fraction'],
            stats['mean_pausing_fraction'],
            stats['mean_shortening_fraction']
        ]
        colors_bar = list(CMGBrownianRatchet.STATE_COLORS)
        
        bars = ax.bar(states, fractions, color=colors_bar, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Time Fraction', fontsize=11)
        ax.set_title(f"{result['name']}\n"
                     f"Velocity: {stats['mean_velocity']:.2f} bp/s\n"
                     f"Final pos: {stats['mean_final_position']:.1f} bp", 
                     fontsize=11)
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, frac in zip(bars, fractions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{frac:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('cmg_ratchet_state_fractions.png', dpi=150, bbox_inches='tight')
    print("Saved state fractions plot to 'cmg_ratchet_state_fractions.png'")
    
    # Plot a single example trajectory from each parameter set
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(all_results):
        ax = axes[i]
        # Plot first trajectory from ensemble
        example_traj = result['trajectories'][0]
        plot_trajectory(example_traj, ax=ax, show_state=True)
        ax.set_title(f"{result['name']} - Trajectory", fontsize=12)
    
    plt.tight_layout()
    plt.savefig('cmg_ratchet_example_trajectories.png', dpi=150, bbox_inches='tight')
    print("Saved example trajectories plot to 'cmg_ratchet_example_trajectories.png'")
    
    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Parameter Set':<20} {'Velocity (bp/s)':<18} {'Unwinding':<12} "
          f"{'Pausing':<12} {'Shortening':<12} {'Final Pos (bp)':<15}")
    print("-"*80)
    for result in all_results:
        stats = result['stats']
        print(f"{result['name']:<20} {stats['mean_velocity']:<18.2f} "
              f"{stats['mean_unwinding_fraction']:<12.2f} "
              f"{stats['mean_pausing_fraction']:<12.2f} "
              f"{stats['mean_shortening_fraction']:<12.2f} "
              f"{stats['mean_final_position']:<15.1f}")
    print("="*80)
    
    # plt.show()

if __name__ == "__main__":
    main()
