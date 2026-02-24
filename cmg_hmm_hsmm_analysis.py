"""
Analysis script for CMG helicase traces using HMM / HSMM-style models.

Usage (from this directory):
    python cmg_hmm_hsmm_analysis.py \\
        --pattern "vivian/data/20251110/*.txt" \\
        --n_states 3 \\
        --feature velocity
- Load all matching files,
- Fit a pooled 3-state HMM on velocity,
- Save summary plots and print basic statistics.
"""

from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import matplotlib.pyplot as plt

try:
    from hmmlearn import hmm
    HAS_HMMLEARN = True
except Exception:  # pragma: no cover - user environment dependent
    HAS_HMMLEARN = False

try:
    # Optional: GaussianMixture for better HMM initialization
    from sklearn.mixture import GaussianMixture  # type: ignore
    HAS_GMM = True
except Exception:  # pragma: no cover
    HAS_GMM = False

# try:
#     # ssm is a nice HMM/HSMM library by Linderman et al.
#     import ssm  # type: ignore
#     HAS_SSM = True
# except Exception:  # pragma: no cover
#     HAS_SSM = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY_STATS = True
except Exception:
    HAS_SCIPY_STATS = False
    scipy_stats = None


try:
    from visualize_2 import (
        parse_file_like_original, 
        detect_noisy_tail_index,
        denoiszation,
        _HAS_SCIPY,
        plot_dwell_time_survival,
        plot_first_passage_density,
    )  # type: ignore
    HAS_PARSE_HELPER = True
    HAS_TAIL_DETECTION = True
    HAS_DENOISING = True
    HAS_DWELL_PLOTS = True
except Exception:  # pragma: no cover
    HAS_PARSE_HELPER = False
    HAS_TAIL_DETECTION = False
    HAS_DENOISING = False
    HAS_DWELL_PLOTS = False
    _HAS_SCIPY = False


CMG_STATE_COLORS = ("green", "orange", "red")  # unwinding, pausing, backtracking

@dataclass
class Trace:
    """Container for a single CMG trace."""
    filename: Path
    time: np.ndarray        # shape (T,)
    signal: np.ndarray      # shape (T,)
    velocity: np.ndarray    # shape (T,) with padding
    acceleration: np.ndarray  # shape (T,) with padding

def load_trace(path: Path, 
               crop_tail: bool = True,
               denoise: bool = True,
               x_slice_start: Optional[int] = None,
               tail_detection_params: Optional[Dict] = None,
               denoise_params: Optional[Dict] = None) -> Trace:
    """
    Load a single trace and compute velocity.
    """
    # Step 1: Parse file exactly like visualize_2.py plot_one_file
    if HAS_PARSE_HELPER:
        print("Using visualize_2.py parser for loading trace.")
        headers, data_cols = parse_file_like_original(str(path))
        if headers is None or data_cols is None:
            raise ValueError(f"{path} could not be parsed or is empty")
        if len(data_cols) < 3:
            raise ValueError(f"{path} doesn't have a third column (has {len(data_cols)} columns)")
        
        # Step 2: Extract columns exactly like visualize_2.py plot_one_file
        # x = data_cols[0]
        # y_target = data_cols[1:]
        # y = np.array(y_target[2])
        x = data_cols[0]
        y_target = data_cols[1:]
        # Make sure the two arrays have same length (they should)
        n = min(len(x), len(y_target[0]) if len(y_target) > 0 else 0)
        t = np.array(x[:n], dtype=float)

        # Signal = first data column (position/extension); column 0 = time
        y = np.array(y_target[0][:n], dtype=float)

    # Step 3: Optional x_slice_start 
    if x_slice_start is not None:
        if x_slice_start >= len(t):
            print(f"Warning: slice start {x_slice_start} >= data length ({len(t)}) for {path}")
        else:
            t = t[x_slice_start:]
            y = y[x_slice_start:]

    # Step 4: Crop tail 
    if crop_tail and HAS_TAIL_DETECTION:
        if tail_detection_params is None:
            # Use default parameters from visualize_2.py
            tail_detection_params = {
                'deriv_window': 20,
                'deriv_thr_mul': 8.0,
                'deriv_min_consec': 25,
                'median_window': 10,
                'median_drop_frac': 0.7,
                'median_min_consec': 15,
                'margin': 2
            }
        
        cut_idx = detect_noisy_tail_index(y, **tail_detection_params)
        if cut_idx < len(y) and cut_idx > 0:
            t = t[:cut_idx]
            y = y[:cut_idx]

    # Step 5: Apply denoising
    if denoise and HAS_DENOISING:
        if denoise_params is None:
            # Use default parameters from visualize_2.py
            '''
            denoise_params = {
                'interpolate_nans': True,
                'median_kernel': 11,
                'savgol_window': 500,
                'savgol_polyorder': 2,
                'aggressive_guard': True,
                'aggressive_thresh_mult': 19.0,
            }
            '''
            denoise_params = {
                'interpolate_nans': True,
                'median_kernel': 3,
                'savgol_window': 10,
                'savgol_polyorder': 2,
                'aggressive_guard': True,
                'aggressive_thresh_mult': 2.0,
            }
        
        # denoiszation expects (x, y, headers, filepath, ...)
        # Note: denoiszation returns (y_denoised, info, x) - 3 values
        # Use actual headers if available, otherwise dummy
        if HAS_PARSE_HELPER and headers:
            use_headers = headers
        else:
            use_headers = ['time', 'col1', 'col2', 'signal']
        
        # y_denoised, denoise_info,,t_denoised = denoiszation(
        y_denoised, denoise_info,t_denoised = denoiszation(
            t, y, use_headers, str(path),
            save=False,  # Don't save plots
            show=False,  # Don't show plots
            **denoise_params
        )
        y = y_denoised
        t = t_denoised  # Use time array returned by denoiszation

    # Compute velocity (simple finite difference)
    dt = np.diff(t)
    dt[dt == 0] = np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0
    v_diff = np.diff(y) / dt

    # Pad velocity to length T by repeating last value
    if len(v_diff) == 0:
        v = np.zeros_like(y)
    else:
        v = np.concatenate([v_diff, v_diff[-1:]])

    # Compute acceleration as finite difference of velocity
    if len(v) > 1:
        a_diff = np.diff(v) / dt  # dt has length T-1, same as np.diff(v)
        a = np.concatenate([a_diff, a_diff[-1:]])
    else:
        a = np.zeros_like(v)

    return Trace(filename=path, time=t, signal=y, velocity=v, acceleration=a)


def median_filter(x: np.ndarray, k: int) -> np.ndarray:
    """Simple 1D median filter with odd window size k."""
    if k <= 1:
        return x.copy()
    if k % 2 == 0:
        k += 1
    pad = k // 2
    x_pad = np.pad(x, pad, mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = np.median(x_pad[i:i + k])
    return out


def preprocess_trace(trace: Trace,
                     median_k: int = 1,
                     feature: str = "velocity") -> np.ndarray:
    """
    Build a feature array for HMM fitting.
    Y : ndarray, shape (T, D)
        Feature matrix for this trace.
    """
    if feature == "velocity":
        x = trace.velocity
    elif feature == "signal":
        x = trace.signal
    elif feature == "acceleration":
        x = trace.acceleration
    else:
        raise ValueError(f"Unknown feature '{feature}'. Expected 'velocity', 'signal', or 'acceleration'.")

    x_filt = median_filter(x, median_k)
    Y = x_filt.reshape(-1, 1)
    return Y


# ---------------------------------------------------------------------------
# HMM fitting (using hmmlearn)
def fit_hmm_gaussian(traces: List[Trace],
                     n_states: int = 3,
                     feature: str = "velocity",
                     median_k: int = 1,
                     n_iter: int = 500,
                     tol: float = 1e-3,
                     random_state: int = 0,
                     n_restarts: int = 10
                     ) -> Tuple[Optional[hmm.GaussianHMM], Dict[str, np.ndarray]]:
    """
    Fit a Gaussian HMM to one or more traces, with optional multi-restarts
    and GMM-based initialization to improve convergence.

    n_states : int
        Number of hidden states.
    median_k : int
        Median filter window.

    Returns: model : GaussianHMM or None
        Fitted model, or None if hmmlearn is not available.
    """

    print(f"\n{'='*70}")
    print("HMM FITTING PROCESS")
    print(f"{'='*70}")
    
    # Build concatenated feature matrix and lengths
    print(f"\n[Step 1/4] Preprocessing traces...")
    print(f"  - Feature: {feature}")
    print(f"  - Median filter window: {median_k}")
    Ys: List[np.ndarray] = []
    lengths: List[int] = []
    for i, tr in enumerate(traces):
        Y = preprocess_trace(tr, median_k=median_k, feature=feature)
        Ys.append(Y)
        lengths.append(len(Y))
        if i == 0:
            print(f"  - Trace {i+1}: {len(Y)} time points")

    Y_all = np.vstack(Ys)
    total_points = len(Y_all)
    print(f"  - Total data points across all traces: {total_points}")


    # Multi-restart HMM fitting with GMM initialization
    print(f"\n[Step 2/4] Fitting HMM with {n_restarts} restarts...")
    print(f"  - Number of states: {n_states}")
    print(f"  - Maximum iterations per restart: {n_iter}")
    print(f"  - Convergence tolerance: {tol}")
    print(f"  - Using {'GMM init' if HAS_GMM else 'random init only'}")

    best_model: Optional[hmm.GaussianHMM] = None
    best_ll = -np.inf

    for seed in range(n_restarts):
        print(f"\n  Restart {seed+1}/{n_restarts}")
        # Initialize HMM
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=n_iter,
            algorithm='viterbi',
            tol=tol,
            random_state=seed,
            verbose=False,
        )

        # Optional: GMM-based initialization
        if HAS_GMM:
            try:
                gmm = GaussianMixture(
                    n_components=n_states,
                    covariance_type="diag",
                    random_state=seed,
                ).fit(Y_all)
                model.means_init = gmm.means_
                model.covars_init = gmm.covariances_
                print("    - Initialized with GMM means/covariances")
            except Exception as e:
                print(f"    - GMM init failed ({e}), falling back to random init")

        try:
            model.fit(Y_all, lengths)
            ll = model.score(Y_all, lengths)
            print(f"    - Log-likelihood: {ll:.4f}")
            print(f"    - Converged: {model.monitor_.converged}, "
                  f"iterations: {model.monitor_.iter}")

            if ll > best_ll:
                best_ll = ll
                best_model = model
                print("    ✓ New best model found")
        except Exception as e:
            print(f"    ✗ HMM fit error on restart {seed+1}: {e}")
            continue

    if best_model is None:
        print("✗ All HMM restarts failed. Returning None.")
        return None, {}

    print(f"\n[Step 3/4] Using best model with log-likelihood = {best_ll:.4f}")


    # Decode state sequences with best model
    print(f"\n[Step 4/4] Decoding state sequences with best model...")
    states_list: List[np.ndarray] = []
    post_list: List[np.ndarray] = []
    offset = 0
    for i, L in enumerate(lengths):
        Y_tr = Y_all[offset:offset + L]
        z = best_model.predict(Y_tr)
        gamma = best_model.predict_proba(Y_tr)
        states_list.append(z)
        post_list.append(gamma)
        if i == 0:
            print(f"  - Trace {i+1}: Decoded {len(z)} states")
            unique, counts = np.unique(z, return_counts=True)
            print(f"    State distribution: {dict(zip(unique, counts))}")
        offset += L
    print(f"  ✓ All {len(lengths)} traces decoded")
    print(f"{'='*70}\n")

    results: Dict[str, np.ndarray] = {
        "Y_all": Y_all,
        "lengths": np.asarray(lengths, dtype=int),
        "states_concat": np.concatenate(states_list),
    }
    # Store as object arrays for convenience
    results["states_per_trace"] = np.array(states_list, dtype=object)
    results["post_per_trace"] = np.array(post_list, dtype=object)

    return best_model, results


def count_params_hmm(n_states: int, d: int, cov_type: str = "diag") -> int:
    """
    Count the number of free parameters in a Gaussian HMM.
    Returns:
    k : int
        Number of free parameters.
    """
    # initial probabilities (n-1 free, last is 1 - sum)
    k = (n_states - 1)
    # transition matrix: each of n rows has (n-1) free params
    k += n_states * (n_states - 1)
    # emission means
    k += n_states * d
    # emission covariances
    if cov_type == "diag":
        k += n_states * d
    elif cov_type == "full":
        k += n_states * d * (d + 1) // 2
    else:
        raise ValueError("cov_type must be 'diag' or 'full'")
    return k


def compute_empirical_transition_matrix(
    state_sequences: Any,
    n_states: int,
    exclude_boundary_transitions: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical state-to-state transition matrix from decoded state sequences.
    Returns: 
    trans_counts: Raw count of transitions from state i to state j.
    trans_matrix
    """
    trans_counts = np.zeros((n_states, n_states), dtype=float)

    if isinstance(state_sequences, np.ndarray) and state_sequences.ndim == 1:
        states = state_sequences.astype(int)
        for t in range(len(states) - 1):
            from_state = int(states[t])
            to_state = int(states[t + 1])
            if 0 <= from_state < n_states and 0 <= to_state < n_states:
                trans_counts[from_state, to_state] += 1
    elif isinstance(state_sequences, (list, tuple)) and len(state_sequences) > 0:
        for seq in state_sequences:
            if seq is None:
                continue
            states = np.asarray(seq, dtype=float)
            if states.size < 2:
                continue
            states = states.ravel().astype(int)
            for t in range(len(states) - 1):
                from_state = int(states[t])
                to_state = int(states[t + 1])
                if 0 <= from_state < n_states and 0 <= to_state < n_states:
                    trans_counts[from_state, to_state] += 1
    else:
        states = np.asarray(state_sequences, dtype=float)
        if states.size >= 2:
            states = states.ravel().astype(int)
            for t in range(len(states) - 1):
                from_state = int(states[t])
                to_state = int(states[t + 1])
                if 0 <= from_state < n_states and 0 <= to_state < n_states:
                    trans_counts[from_state, to_state] += 1

    trans_matrix = np.zeros_like(trans_counts)
    for i in range(n_states):
        row_sum = trans_counts[i, :].sum()
        if row_sum > 0:
            trans_matrix[i, :] = trans_counts[i, :] / row_sum
        else:
            trans_matrix[i, :] = 1.0 / n_states

    return trans_counts, trans_matrix


def _fit_best_hmm_on_features(
    Y_all: np.ndarray,
    lengths: List[int],
    n_states: int,
    n_iter: int = 500,
    tol: float = 1e-3,
    n_restarts: int = 3,
    seed_offset: int = 0,
) -> Tuple[Optional["hmm.GaussianHMM"], float]:
    """
    Internal helper: fit a GaussianHMM on precomputed feature matrix with
    multiple random (or GMM-based) restarts and return the best model + logL.
    """
    if not HAS_HMMLEARN:
        return None, -np.inf

    best_model: Optional[hmm.GaussianHMM] = None
    best_ll = -np.inf

    for r in range(n_restarts):
        seed = seed_offset + r
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            algorithm='viterbi',
            n_iter=n_iter,
            tol=tol,
            random_state=seed,
            verbose=False,
        )

        # Optional GMM-based initialization for faster / more stable EM
        if HAS_GMM:
            try:
                gmm = GaussianMixture(
                    n_components=n_states,
                    covariance_type="diag",
                    random_state=seed,
                ).fit(Y_all)
                model.means_init = gmm.means_
                model.covars_init = gmm.covariances_
            except Exception:
                # If GMM init fails, we silently fall back to random init
                pass

        try:
            model.fit(Y_all, lengths)
            ll = float(model.score(Y_all, lengths))
        except Exception:
            continue

        if ll > best_ll:
            best_ll = ll
            best_model = model

    return best_model, best_ll


def scan_hmm_n_states(
    traces: List[Trace],
    n_states_list: List[int],
    feature: str = "velocity",
    median_k: int = 1,
    n_iter: int = 500,
    tol: float = 1e-3,
    n_restarts: int = 3,
    cv_folds: int = 5,
    random_state: int = 0,
    outdir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """
    Scan over different numbers of hidden states, compute AIC/BIC and
    cross-validated predictive performance (negative log-likelihood per point).

    This follows the high-level idea in your pseudo-code:
    - loop over candidate n_states
    - for each, fit a model
    - evaluate a performance measure perf() (here: CV negative log-likelihood)
    - choose the n_states that minimizes the error.
    """
    if not HAS_HMMLEARN:
        print("hmmlearn is not available; skipping n_states scan.")
        return []

    if len(traces) == 0:
        print("No traces provided; skipping n_states scan.")
        return []

    print(f"\n{'='*70}")
    print("HMM MODEL SELECTION: SCAN OVER NUMBER OF STATES")
    print(f"{'='*70}")
    print(f"Candidate numbers of states: {n_states_list}")

    # Precompute feature matrices per trace once
    Ys: List[np.ndarray] = []
    lengths_per_trace: List[int] = []
    for i, tr in enumerate(traces):
        Y = preprocess_trace(tr, median_k=median_k, feature=feature)
        Ys.append(Y)
        lengths_per_trace.append(len(Y))
        if i == 0:
            print(f"  - Example trace length: {len(Y)} time points")

    Y_all = np.vstack(Ys)
    N_total = Y_all.shape[0]
    d = Y_all.shape[1]

    # Build K-fold partition over traces (not individual time points)
    n_traces = len(traces)
    do_cv = n_traces >= 2
    if not do_cv:
        print("Not enough traces for cross-validation; will skip CV and only report AIC/BIC.")
    else:
        k = min(cv_folds, n_traces)
        rng = np.random.RandomState(random_state)
        indices = np.arange(n_traces)
        rng.shuffle(indices)
        folds = [indices[i::k] for i in range(k)]
        print(f"Using {k}-fold cross-validation at the trace level.")

    results: List[Dict[str, Any]] = []

    for n_states in n_states_list:
        print(f"\n[Scan] Fitting HMM with n_states = {n_states}")

        # Full-data fit for AIC/BIC
        model_full, ll_full = _fit_best_hmm_on_features(
            Y_all,
            lengths_per_trace,
            n_states=n_states,
            n_iter=n_iter,
            tol=tol,
            n_restarts=n_restarts,
            seed_offset=1000 * n_states,
        )
        if model_full is None:
            print("  ✗ Could not fit model (all restarts failed). Skipping this n_states.")
            continue

        k_params = count_params_hmm(n_states, d, cov_type="diag")
        AIC = 2 * k_params - 2 * ll_full
        BIC = np.log(N_total) * k_params - 2 * ll_full
        print(f"  - logL (full data): {ll_full:.4f}")
        print(f"  - AIC              : {AIC:.4f}")
        print(f"  - BIC              : {BIC:.4f}")

        # Cross-validated negative log-likelihood per observation
        cv_error = None
        if do_cv:
            per_fold_errors: List[float] = []
            for fold_id, val_idx in enumerate(folds):
                train_idx = np.setdiff1d(indices, val_idx, assume_unique=True)
                if train_idx.size == 0 or val_idx.size == 0:
                    continue

                Y_train = np.vstack([Ys[j] for j in train_idx])
                lengths_train = [lengths_per_trace[j] for j in train_idx]
                Y_val = np.vstack([Ys[j] for j in val_idx])
                lengths_val = [lengths_per_trace[j] for j in val_idx]

                model_cv, ll_train = _fit_best_hmm_on_features(
                    Y_train,
                    lengths_train,
                    n_states=n_states,
                    n_iter=n_iter,
                    tol=tol,
                    n_restarts=max(1, n_restarts // 2),
                    seed_offset=2000 * n_states + 100 * fold_id,
                )
                if model_cv is None:
                    continue

                try:
                    ll_val = float(model_cv.score(Y_val, lengths_val))
                    N_val = float(sum(lengths_val))
                    per_point_neg_ll = -ll_val / max(N_val, 1.0)
                    per_fold_errors.append(per_point_neg_ll)
                except Exception:
                    continue

            if per_fold_errors:
                cv_error = float(np.mean(per_fold_errors))
                print(f"  - CV negative log-likelihood per point: {cv_error:.6f}")
            else:
                print("  - CV failed for all folds (skipping CV metric).")

        entry: Dict[str, Any] = {
            "n_states": n_states,
            "logL": float(ll_full),
            "AIC": float(AIC),
            "BIC": float(BIC),
            "cv_neg_loglike_per_point": cv_error,
        }
        results.append(entry)

    # Print a compact summary table
    if results:
        print(f"\n{'-'*70}")
        print("SUMMARY OVER N_STATES")
        print(f"{'-'*70}")
        header = f"{'n':>3} | {'logL':>12} | {'AIC':>12} | {'BIC':>12} | {'CV -logL/pt':>12}"
        print(header)
        print("-" * len(header))
        for r in sorted(results, key=lambda x: x["n_states"]):
            cv_str = f"{r['cv_neg_loglike_per_point']:.6f}" if r["cv_neg_loglike_per_point"] is not None else "   n/a    "
            print(f"{r['n_states']:3d} | {r['logL']:12.4f} | {r['AIC']:12.4f} | {r['BIC']:12.4f} | {cv_str:>12}")

        # Identify best by BIC and by CV error
        best_bic = min(results, key=lambda x: x["BIC"])
        print(f"\nBest by BIC: n_states = {best_bic['n_states']} (BIC = {best_bic['BIC']:.4f})")
        cv_valid = [r for r in results if r["cv_neg_loglike_per_point"] is not None]
        if cv_valid:
            best_cv = min(cv_valid, key=lambda x: x["cv_neg_loglike_per_point"])
            print(f"Best by CV (lowest -logL/pt): n_states = {best_cv['n_states']} "
                  f"(CV -logL/pt = {best_cv['cv_neg_loglike_per_point']:.6f})")

        # Optionally save summary to a text file in outdir
        if outdir is not None:
            summary_path = Path(outdir) / "hmm_model_selection_summary.txt"
            with open(summary_path, "w") as f:
                f.write("n_states\tlogL\tAIC\tBIC\tcv_neg_loglike_per_point\n")
                for r in sorted(results, key=lambda x: x["n_states"]):
                    if r["cv_neg_loglike_per_point"] is None:
                        cv_val_str = "nan"
                    else:
                        cv_val_str = f"{r['cv_neg_loglike_per_point']:.6f}"
                    f.write(
                        f"{r['n_states']}\t{r['logL']:.6f}\t{r['AIC']:.6f}\t"
                        f"{r['BIC']:.6f}\t{cv_val_str}\n"
                    )
            print(f"\nSaved model selection summary to: {summary_path}")

    return results


def interpret_states(model: "hmm.GaussianHMM", 
                    use_velocity: bool = True,
                    n_states: int = 3) -> Dict[int, str]:
    """
    Interpret HMM states based on their mean values.
    Better definition for CMG helicase: Unwinding, Pausing, Backtracking.
    """
    # NOTE:
    # The user requested to drop semantic naming like "Unwinding", "Pausing",
    # and "Backtracking", and instead use neutral labels: "State 0", "State 1", ...
    # This function now simply returns generic labels for each state index.
    n = model.n_components
    return {i: f"State {i}" for i in range(n)}


def get_state_colors(state_labels: Dict[int, str]) -> Dict[int, str]:
    # First three states: same as ratchet (green, orange, red)
    hidden_state_colors = ["purple", "blue", "cyan", "magenta", "yellow", "brown", "pink", "gray", "olive", "teal"]
    color_map = {}
    for state_idx in sorted(state_labels.keys()):
        if state_idx < len(CMG_STATE_COLORS):
            color_map[state_idx] = CMG_STATE_COLORS[state_idx]
        else:
            color_map[state_idx] = hidden_state_colors[(state_idx - len(CMG_STATE_COLORS)) % len(hidden_state_colors)]
    return color_map


def extract_dwell_times(states: np.ndarray, verbose: bool = False) -> Dict[int, List[int]]:

    dwells: Dict[int, List[int]] = {}
    if len(states) == 0:
        return dwells

    if verbose:
        print(f"  - Extracting dwell times from {len(states)} time points...")
    
    current = states[0]
    run_len = 1
    transitions = 0
    
    for s in states[1:]:
        if s == current:
            run_len += 1
        else:
            dwells.setdefault(int(current), []).append(run_len)
            transitions += 1
            current = s
            run_len = 1
    dwells.setdefault(int(current), []).append(run_len)
    
    if verbose:
        print(f"  - Found {transitions + 1} state segments")
        for state_idx, dwell_list in sorted(dwells.items()):
            print(f"    State {state_idx}: {len(dwell_list)} dwell events")
    
    return dwells


def get_median_dwell_length(states: np.ndarray, k: int) -> float:
    """
    Helper: compute the median dwell length for state k in a state sequence.
    Uses contiguous run lengths (same definition as extract_dwell_times).
    """
    dw = extract_dwell_times(states, verbose=False)
    lst = dw.get(int(k), [])
    if not lst:
        return 0.0
    return float(np.median(lst))
# -----------------------------------------------------
# Plotting 

def plot_fitted_hmm(
    model: "hmm.GaussianHMM",
    Y_all: np.ndarray,
    lengths: np.ndarray,
    traces: List[Trace],
    out_path: Path,
    state_labels: Optional[Dict[int, str]] = None,
    feature_name: str = "velocity",
) -> None:
    """
    Plot the fitted HMM: observations and decoded hidden states from model.predict(X).
    Two panels: (1) observation vs time colored by hidden state, (2) hidden state vs time.
    """
    lengths = np.atleast_1d(lengths).astype(int)
    # hidden_states = model.predict(X) over the full sequence (with lengths for boundaries)
    hidden_states = model.predict(Y_all, lengths)

    # Build time axis aligned with Y_all (same length as each trace's feature)
    time_chunks = []
    for i, tr in enumerate(traces):
        L = lengths[i] if i < len(lengths) else len(tr.time)
        t = tr.time[:L]
        time_chunks.append(t)
    time_all = np.concatenate(time_chunks)
    if len(time_all) != len(hidden_states):
        time_all = np.arange(len(hidden_states), dtype=float)

    X = Y_all.ravel() if Y_all.ndim > 1 else Y_all
    n_states = model.n_components
    if state_labels:
        color_map = get_state_colors(state_labels)
    else:
        color_map = {
            i: (CMG_STATE_COLORS[i] if i < len(CMG_STATE_COLORS) else "gray")
            for i in range(n_states)
        }

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_obs, ax_st = axes[0], axes[1]

    # Panel 1: observation vs time, colored by hidden state
    for i in range(len(time_all) - 1):
        s = int(hidden_states[i])
        c = color_map.get(s, "gray")
        ax_obs.plot(time_all[i : i + 2], X[i : i + 2], color=c, linewidth=1)
    ax_obs.set_ylabel(feature_name.capitalize(), fontsize=11)
    ax_obs.set_title("Fitted HMM: observation colored by hidden state", fontsize=12)
    ax_obs.grid(True, alpha=0.3)
    # Legend
    used = sorted(np.unique(hidden_states))
    handles = [
        plt.Line2D([0], [0], color=color_map.get(s, "gray"), linewidth=2, label=state_labels.get(s, f"State {s}") if state_labels else f"State {s}")
        for s in used
    ]
    ax_obs.legend(handles=handles, loc="upper right", fontsize=9)

    # Panel 2: hidden state vs time (step plot)
    ax_st.step(time_all, hidden_states, where="mid", color="#333", linewidth=0.8, alpha=0.9)
    ax_st.set_ylabel("Hidden state", fontsize=11)
    ax_st.set_xlabel("Time (s)", fontsize=11)
    ax_st.set_yticks(range(n_states))
    if state_labels:
        ax_st.set_yticklabels([state_labels.get(i, str(i)) for i in range(n_states)], fontsize=9)
    ax_st.set_ylim(-0.3, n_states - 0.7)
    ax_st.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved fitted HMM plot: {out_path.name}")


def plot_fitted_hmm_on_noisy_data(
    model: "hmm.GaussianHMM",
    Y_all: np.ndarray,
    lengths: np.ndarray,
    traces: List[Trace],
    out_path: Path,
    state_labels: Optional[Dict[int, str]] = None,
    feature_name: str = "velocity",
) -> None:
    """
    Plot another figure: (1) original noisy data only, (2) same data with
    fitted HMM predicted states overlaid on top. For direct comparison.
    """
    lengths = np.atleast_1d(lengths).astype(int)
    hidden_states = model.predict(Y_all, lengths)

    time_chunks = []
    for i, tr in enumerate(traces):
        L = lengths[i] if i < len(lengths) else len(tr.time)
        time_chunks.append(tr.time[:L])
    time_all = np.concatenate(time_chunks)
    if len(time_all) != len(hidden_states):
        time_all = np.arange(len(hidden_states), dtype=float)

    X = Y_all.ravel() if Y_all.ndim > 1 else Y_all
    n_states = model.n_components
    if state_labels:
        color_map = get_state_colors(state_labels)
    else:
        color_map = {
            i: (CMG_STATE_COLORS[i] if i < len(CMG_STATE_COLORS) else "gray")
            for i in range(n_states)
        }

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_raw, ax_fit = axes[0], axes[1]

    # Panel 1: Original (noisy) data only – single curve
    ax_raw.plot(time_all, X, color="#4a90a4", linewidth=0.8, alpha=0.9, label="Original (noisy) data")
    ax_raw.set_ylabel(feature_name.capitalize(), fontsize=11)
    ax_raw.set_title("Original simulated noisy data", fontsize=12)
    ax_raw.legend(loc="upper right", fontsize=10)
    ax_raw.grid(True, alpha=0.3)

    # Panel 2: Same data with fitted HMM predicted states on top (color by state)
    for i in range(len(time_all) - 1):
        s = int(hidden_states[i])
        c = color_map.get(s, "gray")
        ax_fit.plot(time_all[i : i + 2], X[i : i + 2], color=c, linewidth=1.2)
    ax_fit.set_ylabel(feature_name.capitalize(), fontsize=11)
    ax_fit.set_xlabel("Time (s)", fontsize=11)
    ax_fit.set_title("Fitted HMM (predicted states) on top of noisy data", fontsize=12)
    used = sorted(np.unique(hidden_states))
    handles = [
        plt.Line2D([0], [0], color=color_map.get(s, "gray"), linewidth=2,
                  label=state_labels.get(s, f"State {s}") if state_labels else f"State {s}")
        for s in used
    ]
    ax_fit.legend(handles=handles, loc="upper right", fontsize=9)
    ax_fit.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved HMM-on-noisy-data plot: {out_path.name}")


def plot_posterior_confidence(
    post_per_trace: np.ndarray,
    lengths: np.ndarray,
    traces: List[Trace],
    out_path: Path,
    state_labels: Optional[Dict[int, str]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Demonstrate posterior probabilities (model.predict_proba): compute
    per-timepoint confidence (max state probability), print summary stats,
    and plot histogram + confidence vs time for first trace.
    Returns (max_proba_concat, summary_dict).
    """
    lengths = np.atleast_1d(lengths).astype(int)
    max_proba_list = []
    for i, gamma in enumerate(post_per_trace):
        if gamma is None or not hasattr(gamma, "shape"):
            continue
        gamma = np.asarray(gamma, dtype=float)
        if gamma.ndim != 2:
            continue
        # Confidence at each timepoint = max over states
        max_proba_list.append(gamma.max(axis=1))
    if not max_proba_list:
        print("  No posterior arrays available for confidence plot.")
        return np.array([]), {}

    max_proba_concat = np.concatenate(max_proba_list)
    n = len(max_proba_concat)
    mean_conf = float(np.mean(max_proba_concat))
    median_conf = float(np.median(max_proba_concat))
    frac_90 = float(np.mean(max_proba_concat >= 0.90))
    frac_95 = float(np.mean(max_proba_concat >= 0.95))
    frac_99 = float(np.mean(max_proba_concat >= 0.99))
    summary = {
        "mean_confidence": mean_conf,
        "median_confidence": median_conf,
        "frac_above_0.90": frac_90,
        "frac_above_0.95": frac_95,
        "frac_above_0.99": frac_99,
        "n_timepoints": n,
    }

    print("\n[Posterior probabilities (model.predict_proba)]")
    print("  Confidence at each timepoint = max over state probabilities.")
    print(f"  Mean confidence:   {mean_conf:.4f}")
    print(f"  Median confidence: {median_conf:.4f}")
    print(f"  Fraction of timepoints with confidence ≥ 0.90: {frac_90:.2%}")
    print(f"  Fraction of timepoints with confidence ≥ 0.95: {frac_95:.2%}")
    print(f"  Fraction of timepoints with confidence ≥ 0.99: {frac_99:.2%}")
    print(f"  → High confidence for most timepoints: {'Yes' if mean_conf >= 0.7 else 'Moderate/Low'}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=False)
    ax_hist, ax_time = axes[0], axes[1]

    # Panel 1: Histogram of max posterior (confidence)
    ax_hist.hist(max_proba_concat, bins=50, color="#2e86ab", alpha=0.7, edgecolor="white")
    ax_hist.axvline(mean_conf, color="#e94f37", linestyle="--", linewidth=2, label=f"Mean = {mean_conf:.3f}")
    ax_hist.axvline(0.9, color="gray", linestyle=":", linewidth=1, label="0.90")
    ax_hist.set_xlabel("Posterior confidence (max state probability)", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=11)
    ax_hist.set_title("Posterior confidence: model.predict_proba (max over states)", fontsize=12)
    ax_hist.legend(loc="upper left", fontsize=9)
    ax_hist.grid(True, alpha=0.3)

    # Panel 2: Confidence vs time for first trace
    if len(max_proba_list) > 0 and len(traces) > 0:
        c0 = max_proba_list[0]
        L0 = lengths[0] if len(lengths) > 0 else len(c0)
        t0 = traces[0].time[:L0] if len(traces[0].time) >= L0 else np.arange(L0, dtype=float)
        if len(t0) != len(c0):
            t0 = np.arange(len(c0), dtype=float)
        ax_time.fill_between(t0, 0, c0, color="#2e86ab", alpha=0.5)
        ax_time.plot(t0, c0, color="#1a5a6e", linewidth=0.8)
        ax_time.axhline(0.9, color="gray", linestyle=":", linewidth=1, label="0.90")
        ax_time.set_xlabel("Time (s)", fontsize=11)
        ax_time.set_ylabel("Posterior confidence", fontsize=11)
        ax_time.set_title("Confidence vs time (first trace)", fontsize=12)
        ax_time.set_ylim(0, 1.02)
        ax_time.legend(loc="lower right", fontsize=9)
        ax_time.grid(True, alpha=0.3)
    else:
        ax_time.text(0.5, 0.5, "No trace data for time series.", ha="center", va="center", transform=ax_time.transAxes)
        ax_time.set_xlabel("Time (s)", fontsize=11)
        ax_time.set_ylabel("Posterior confidence", fontsize=11)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Saved posterior confidence plot: {out_path.name}")
    return max_proba_concat, summary


def plot_trace_with_states(trace: Trace,
                           states: np.ndarray,
                           out_path: Path,
                           title: str = "",
                           state_labels: Optional[Dict[int, str]] = None) -> None:
    """
    Plot signal vs time with inferred HMM states color-coded.
    Uses consistent colors based on state interpretation.
    
    Parameters
    ----------
    trace : Trace
    states : np.ndarray
        State sequence for this trace
    out_path : Path
    title : str
    state_labels : dict, optional
        Maps state index to interpretation string (e.g., {0: "Unwinding", 1: "Pausing", 2: "Backtracking"})
    """
    t = trace.time
    y = trace.signal
    if len(states) != len(t):
        # If states derived from velocity, they may be length T; we assume that here.
        if len(states) == len(t) - 1:
            states = np.concatenate([states, states[-1:]])

    # Same color code as cmg_brownian_ratchet: 0=green, 1=orange, 2=red, then purple, blue, ...
    if state_labels:
        color_map = get_state_colors(state_labels)
    else:
        extra = ["purple", "blue", "cyan", "magenta"]
        color_map = {i: (CMG_STATE_COLORS[i] if i < len(CMG_STATE_COLORS) else extra[(i - len(CMG_STATE_COLORS)) % len(extra)])
                    for i in range(max(states) + 1) if i in np.unique(states)}

    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(len(t) - 1):
        state_idx = int(states[i])
        c = color_map.get(state_idx, "black")
        ax.plot(t[i:i + 2], y[i:i + 2], color=c, linewidth=1)

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Signal / extension", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    else:
        ax.set_title(f"Trace {trace.filename.name} with HMM states", fontsize=13)
    ax.grid(True, alpha=0.3)

    # Legend: only show states that actually appear in this trace
    unique_states = sorted(np.unique(states))
    legend_handles = []
    for state_idx in unique_states:
        state_idx = int(state_idx)
        c = color_map.get(state_idx, "black")
        if state_labels and state_idx in state_labels:
            label = f"State {state_idx}: {state_labels[state_idx]}"
        else:
            label = f"State {state_idx}"
        legend_handles.append(plt.Line2D([0], [0], color=c, linewidth=2, label=label))
    
    ax.legend(handles=legend_handles, loc='best', fontsize=10)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_transition_matrix(transmat: np.ndarray,
                           out_path: Path,
                           state_labels: Optional[Dict[int, str]] = None) -> None:
    """
    Visualize the HMM transition matrix as a heatmap.
    """
    n_states = transmat.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create labels for states
    if state_labels:
        row_labels = [f"State {i}\n({state_labels.get(i, 'Unknown')})" for i in range(n_states)]
        col_labels = [f"State {i}\n({state_labels.get(i, 'Unknown')})" for i in range(n_states)]
    else:
        row_labels = [f"State {i}" for i in range(n_states)]
        col_labels = [f"State {i}" for i in range(n_states)]
    
    # Create heatmap
    im = ax.imshow(transmat, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, label='Transition Probability', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(n_states))
    ax.set_yticks(np.arange(n_states))
    ax.set_xticklabels(col_labels, fontsize=10, rotation=0, ha='center')
    ax.set_yticklabels(row_labels, fontsize=10)
    
    # Add text annotations with probabilities
    thresh = transmat.max() / 2.0
    for i in range(n_states):
        for j in range(n_states):
            text_color = "white" if transmat[i, j] > thresh else "black"
            ax.text(j, i, f'{transmat[i, j]:.3f}',
                   ha="center", va="center", color=text_color, fontsize=11, fontweight='bold')
    
    # Labels
    ax.set_xlabel('To State', fontsize=12, fontweight='bold')
    ax.set_ylabel('From State', fontsize=12, fontweight='bold')
    ax.set_title('HMM Transition Matrix', fontsize=14, fontweight='bold', pad=20)
    
    # Add grid
    ax.set_xticks(np.arange(n_states + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_states + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=1.5)
    ax.tick_params(which="minor", size=0)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_dwell_hist(dwells: Dict[int, List[int]],
                    out_path: Path,
                    dt: float = 1.0,
                    state_labels: Optional[Dict[int, str]] = None) -> None:
    """
    Plot dwell-time histograms per state.
    Uses consistent colors based on state interpretation.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if not dwells or all(len(durs) == 0 for durs in dwells.values()):
        ax.text(0.5, 0.5, "No dwell times found.\nCheck if HMM states were extracted correctly.",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_xlabel("Dwell time")
        ax.set_ylabel("Count")
        ax.set_title("HMM dwell-time histograms (empty)")
    else:
        # Get consistent color mapping based on state interpretations
        if state_labels:
            color_map = get_state_colors(state_labels)
        else:
            # Same as ratchet for 0,1,2; then purple, blue, ...
            extra = ['purple', 'blue', 'cyan', 'magenta', 'yellow']
            color_map = {k: (CMG_STATE_COLORS[k] if k < len(CMG_STATE_COLORS) else extra[(k - len(CMG_STATE_COLORS)) % len(extra)])
                        for k in dwells.keys()}
        
        for k, durs in sorted(dwells.items()):
            if len(durs) == 0:
                continue
            # Flatten to 1D and convert to seconds (avoid 2D/object arrays causing huge bins)
            durs_t = np.ravel(np.asarray(durs, dtype=float)) * dt
            n_vals = int(np.size(durs_t))
            if n_vals == 0:
                continue
            # Cap bins to a safe range so we never pass a huge number to np.histogram
            n_bins = min(100, max(5, n_vals // 5))
            color = color_map.get(k, "gray")

            if state_labels and k in state_labels:
                label = f"State {k}: {state_labels[k]} (n={n_vals})"
            else:
                label = f"State {k} (n={n_vals})"

            ax.hist(durs_t, bins=n_bins,
                   alpha=0.6, label=label,
                   color=color, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel("Dwell time (seconds)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title("HMM dwell-time histograms (pooled across all traces)", fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='best', fontsize=9)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _linear_fit_log_survival_hmm(x: np.ndarray, y: np.ndarray, n: int) -> Tuple[float, float, float, float]:
    """Linear fit log(survival) vs t; returns slope, intercept, r, se_slope."""
    if n < 2:
        return 0.0, 0.0, np.nan, np.nan
    coef = np.polyfit(x, y, 1)
    slope, intercept = float(coef[0]), float(coef[1])
    y_pred = slope * x + intercept
    res = y - y_pred
    ss_res = float(np.sum(res ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r = np.sqrt(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    mse = ss_res / (n - 2) if n > 2 else 0.0
    s_xx = float(np.sum((x - np.mean(x)) ** 2))
    se_slope = np.sqrt(mse / s_xx) if s_xx > 0 else np.nan
    return slope, intercept, r, se_slope


def plot_hmm_dwell_time_survival(
    dwell_times: np.ndarray,
    out_path: Path,
    show: bool = False,
) -> None:
    """
    Dwell-time survival: log(S) vs t is linear with slope = -1/τ, so the fitted
    slope gives dwell time τ = -1/slope. Plot shows τ ± SE(τ) and R².
    """
    dwell_times = np.asarray(dwell_times, dtype=float)
    dwell_times = dwell_times[np.isfinite(dwell_times) & (dwell_times > 0)]
    if dwell_times.size == 0:
        print("No dwell times to plot survival.")
        return

    sorted_dt = np.sort(dwell_times)
    n = len(sorted_dt)
    survival = 1.0 - np.arange(1, n + 1) / (n + 1.0)
    log_survival = np.log(np.clip(survival, 1e-15, 1.0))

    if HAS_SCIPY_STATS and scipy_stats is not None and n >= 3:
        try:
            res = scipy_stats.linregress(sorted_dt, log_survival)
            slope, intercept = res.slope, res.intercept
            r_val, se_slope = res.rvalue, res.stderr
        except Exception:
            slope, intercept, r_val, se_slope = _linear_fit_log_survival_hmm(sorted_dt, log_survival, n)
    else:
        slope, intercept, r_val, se_slope = _linear_fit_log_survival_hmm(sorted_dt, log_survival, n)

    rate_fit = -slope if slope != 0 else 1.0 / float(np.mean(dwell_times))
    tau_fit = 1.0 / rate_fit if rate_fit > 0 else np.nan
    se_tau = (float(np.abs(se_slope)) / (slope ** 2)) if slope != 0 and np.isfinite(se_slope) else np.nan
    if not np.isfinite(se_tau) or se_tau <= 0:
        se_tau = tau_fit / np.sqrt(n) if n > 0 and np.isfinite(tau_fit) else np.nan
    r_squared = (r_val ** 2) if np.isfinite(r_val) else np.nan

    t_fit = np.linspace(sorted_dt.min(), sorted_dt.max(), 200)
    survival_exp = np.exp(slope * t_fit + intercept)

    mean_dt = float(np.mean(dwell_times))
    median_dt = float(np.median(dwell_times))
    std_dt = float(np.std(dwell_times))
    ks_stat, ks_pval = np.nan, np.nan
    if HAS_SCIPY_STATS and scipy_stats is not None and n >= 5:
        try:
            ks_stat, ks_pval = scipy_stats.kstest(dwell_times, scipy_stats.expon(scale=mean_dt).cdf)
        except Exception:
            pass

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(sorted_dt, survival, "o-", color="#2e86ab", markersize=4, linewidth=1.5, label="Empirical survival")
    ax.semilogy(t_fit, survival_exp, "--", color="#e94f37", linewidth=2, label=f"Exponential fit (τ = {tau_fit:.3g} s)")
    ax.set_xlabel("Dwell time (s)", fontsize=12)
    ax.set_ylabel("Survival 1 − CDF", fontsize=12)
    ax.set_title("HMM dwell-time survival (log y)", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)

    text_lines = [
        f"τ (dwell time) = {tau_fit:.4g} ± {se_tau:.4g} s",
        f"R² = {r_squared:.4f}" if np.isfinite(r_squared) else "R² = —",
    ]
    if np.isfinite(ks_pval):
        text_lines.append(f"KS p = {ks_pval:.3f}")
    ax.text(0.05, 0.35, "\n".join(text_lines), transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8), family="monospace")

    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    ax.set_ylim(bottom=min(0.5 / (n + 1), float(survival.min()) * 0.5))
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  ✓ Saved: {out_path.name}")
    print("  τ (dwell time from slope of log S vs t): τ = {:.4g} ± {:.4g} s".format(tau_fit, se_tau))
    print("  Fit quality: R² = {:.4f}".format(r_squared) if np.isfinite(r_squared) else "  Fit quality: R² = —")
    print("  Statistics: n = {}, mean = {:.4g} s, median = {:.4g} s, std = {:.4g} s".format(n, mean_dt, median_dt, std_dt))
    if np.isfinite(ks_pval):
        print("  KS test (exponential): statistic = {:.4f}, p-value = {:.4f}".format(ks_stat, ks_pval))
    if show:
        plt.show()


def plot_hmm_first_passage_density(
    dwell_times: np.ndarray,
    out_path: Path,
    show: bool = False,
    n_bins: int = 200,
) -> None:
    """
    First-passage time probability density (log-log) with exponential PDF overlay
    and summary statistics.
    """
    dwell_times = np.asarray(dwell_times, dtype=float)
    dwell_times = dwell_times[np.isfinite(dwell_times) & (dwell_times > 0)]
    if dwell_times.size == 0:
        print("No dwell times to plot first-passage density.")
        return

    t_min = max(dwell_times.min(), 1e-10)
    t_max = dwell_times.max()
    if not np.isfinite(t_max) or t_max <= 0:
        print("Invalid dwell times for first-passage density plot.")
        return

    mean_dt = float(np.mean(dwell_times))
    rate_mle = 1.0 / mean_dt if mean_dt > 0 else 0.0

    bins = np.logspace(np.log10(t_min), np.log10(t_max), n_bins + 1)
    hist, edges = np.histogram(dwell_times, bins=bins, density=True)
    bin_centers = np.sqrt(edges[:-1] * edges[1:])
    mask = hist > 0
    hist = hist[mask]
    bin_centers = bin_centers[mask]

    if hist.size == 0:
        print("All histogram bins are zero; skipping first-passage density plot.")
        return

    pdf_exp = rate_mle * np.exp(-rate_mle * bin_centers)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(bin_centers, hist, "o-", color="#2e86ab", markersize=3.5, linewidth=1.5, label="Empirical density")
    ax.loglog(bin_centers, pdf_exp, "--", color="#e94f37", linewidth=2, label=f"Exponential PDF (λ = {rate_mle:.4g} s⁻¹)")
    ax.set_xlabel("Passage time Δt (s)", fontsize=12)
    ax.set_ylabel("First-passage density p(Δt)", fontsize=12)
    ax.set_title("HMM first-passage time density", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, which="major", alpha=0.4)
    ax.grid(True, which="minor", alpha=0.2, linestyle=":")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  ✓ Saved: {out_path.name}")
    print("  Exponential fit: rate λ = 1/mean = {:.4g} s⁻¹".format(rate_mle))
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Fit HMM / HSMM models to CMG helicase traces."
    )
    parser.add_argument(
        "--pattern",
        type=str,
        required=True,
        help="Glob pattern for trace files (e.g. 'vivian/data/20251110/*.txt').",
    )
    parser.add_argument(
        "--n_states",
        type=int,
        default=3,
        help="Number of hidden states in HMM/HSMM.",
    )
    parser.add_argument(
        "--feature",
        type=str,
        default="velocity",
        choices=["velocity", "signal", "acceleration"],
        help="Which feature to feed into the model.",
    )
    parser.add_argument(
        "--median_k",
        type=int,
        default=4,
        help="Median filter window size for feature preprocessing.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="hmm_analysis_outputs",
        help="Directory to save plots and summaries.",
    )
    parser.add_argument(
        "--hsmm",
        action="store_true",
        help="If set, also attempt HSMM fit (requires `ssm`).",
    )
    parser.add_argument(
        "--no_crop_tail",
        action="store_true",
        help="If set, skip tail cropping (default is to crop noisy tails like visualize_2.py).",
    )
    parser.add_argument(
        "--no_denoise",
        action="store_true",
        help="If set, skip denoising (default is to denoise using visualize_2.py denoiszation).",
    )

    args = parser.parse_args()

    base = Path(".")
    paths = sorted(base.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No files found matching pattern: {args.pattern}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {len(paths)} traces...")
    crop_tail = not args.no_crop_tail
    denoise = not args.no_denoise
    if denoise and HAS_DENOISING:
        print("Applying denoising (same as visualize_2.py)...")
    if crop_tail and HAS_TAIL_DETECTION:
        print("Applying tail detection and cropping (same as visualize_2.py)...")
    traces = [load_trace(p, crop_tail=crop_tail, denoise=denoise) for p in paths]

    # n_states_grid = [2, 3, 4, 5, 6, 7]
    # scan_hmm_n_states(
    #     traces=traces,
    #     n_states_list=n_states_grid,
    #     feature=args.feature,
    #     median_k=args.median_k,
    #     n_iter=500,
    #     tol=1e-3,
    #     n_restarts=3,
    #     cv_folds=5,
    #     random_state=0,
    #     outdir=outdir,
    # )

    # HMM fit
    model, hmm_res = fit_hmm_gaussian(
        traces=traces,
        n_states=args.n_states,
        feature=args.feature,
        median_k=args.median_k,
    )

    if model is None:
        print("Skipping HMM analysis because hmmlearn is not available.")
    else:
        print(f"\n{'='*70}")
        print("HMM PARAMETERS")
        print(f"{'='*70}")
        # ------------------------------------------------------------------
        # Model selection metrics: logL, AIC, BIC
        X_all = hmm_res["Y_all"]
        N = X_all.shape[0]
        d = X_all.shape[1]
        logL = float(model.score(X_all, hmm_res["lengths"]))
        k_params = count_params_hmm(model.n_components, d, cov_type="diag")
        AIC = 2 * k_params - 2 * logL
        BIC = np.log(N) * k_params - 2 * logL
        print("\n[Model selection summary]")
        print(f"  - Total log-likelihood (logL): {logL:.4f}")
        print(f"  - Number of observations (N) : {N}")
        print(f"  - Number of parameters (k)   : {k_params}")
        print(f"  - AIC                        : {AIC:.4f}")
        print(f"  - BIC                        : {BIC:.4f}")
        print("\n[Transition Matrix Calculation]")
        print("  The transition matrix was learned during HMM fitting using the EM algorithm.")
        print("  Each element T[i,j] represents the probability of transitioning")
        print("  from state i to state j in one time step.\n")
        print("Transition matrix:")
        print(model.transmat_)
        
        # Calculate and show empirical transition matrix (same logic as cmg_brownian_ratchet_improved)
        print("\n[Empirical Transition Matrix (from decoded states)]")
        print("  Calculating transition counts from decoded state sequences (within-trace only)...")
        n_states = model.n_components
        states_per_trace = hmm_res["states_per_trace"]
        trans_counts, trans_empirical = compute_empirical_transition_matrix(
            state_sequences=list(states_per_trace),
            n_states=n_states,
            exclude_boundary_transitions=True,
        )
        states_concat = hmm_res["states_concat"]
        print("  Empirical transition matrix (from observed transitions): T[i,j] = P(next=j | current=i)")
        print(trans_empirical)
        print(f"  Total transitions observed: {int(trans_counts.sum())}")
        
        means = model.means_.ravel()
        print("\n[Emission Parameters]")
        print("Means (per state):", means)
        print("Variances (per state):", model.covars_.ravel())
        
        print(f"\n{'='*70}")
        print("STATE INTERPRETATION")
        print(f"{'='*70}")
        print("\n[Interpreting states (generic labels)...]")
        # Interpret states with generic labels ("State 0", "State 1", ...)
        state_labels = interpret_states(model, use_velocity=False, n_states=args.n_states)
        
        print("\n" + "="*70)
        print("STATE INTERPRETATIONS:")
        print("="*70)
        # Get color mapping for display
        color_map = get_state_colors(state_labels)
        for state_idx in sorted(state_labels.keys()):
            mean_val = means[state_idx]
            interpretation = state_labels[state_idx]
            color = color_map.get(state_idx, "unknown")
            print(f"  State {state_idx}: {interpretation:20s} (mean = {mean_val:8.4f}, color = {color})")
        print("="*70 + "\n")

        # ------------------------------------------------------------------
        # Posterior probabilities (model.predict_proba) – high confidence for most timepoints
        if "post_per_trace" in hmm_res and hmm_res["post_per_trace"] is not None:
            plot_posterior_confidence(
                post_per_trace=hmm_res["post_per_trace"],
                lengths=hmm_res["lengths"],
                traces=traces,
                out_path=outdir / "hmm_posterior_confidence.png",
                state_labels=state_labels,
            )
        else:
            print("\n[Posterior probabilities (model.predict_proba)]")
            print("  Not available (post_per_trace not in results).")

        # ------------------------------------------------------------------
        # Per-state feature + dwell summaries (why the HMM split states)
        print("\n[Per-state feature and dwell summary]")
        print("  For each state: count of samples, mean/std of feature,")
        print("  and median dwell length (contiguous run length).")
        X = hmm_res["Y_all"].ravel()
        states = states_concat
        for k in range(model.n_components):
            idx = (states == k)
            count = int(idx.sum())
            if count > 0:
                vals = X[idx]
                mean_val = float(vals.mean())
                std_val = float(vals.std())
            else:
                mean_val = 0.0
                std_val = 0.0
            median_dwell = get_median_dwell_length(states, k)
            label = state_labels.get(k, f"State {k}")
            print(f"  State {k} ({label}):")
            print(f"    - sample count       : {count}")
            print(f"    - feature mean       : {mean_val:.6g}")
            print(f"    - feature std        : {std_val:.6g}")
            print(f"    - median dwell length: {median_dwell:.2f} steps")
        
        print(f"\n[Plotting transition matrix heatmap...]")
        plot_transition_matrix(
            model.transmat_,
            out_path=outdir / "hmm_transition_matrix.png",
            state_labels=state_labels
        )
        print(f"  ✓ Saved: hmm_transition_matrix.png")

        print(f"\n{'='*70}")
        print("GENERATING PLOTS")
        print(f"{'='*70}")
        # Fitted HMM: plot observations + hidden_states = model.predict(X)
        print("\n[Plotting fitted HMM (hidden_states = model.predict(X))...]")
        plot_fitted_hmm(
            model=model,
            Y_all=hmm_res["Y_all"],
            lengths=hmm_res["lengths"],
            traces=traces,
            out_path=outdir / "hmm_fitted_model.png",
            state_labels=state_labels,
            feature_name=args.feature,
        )
        # Second graph: original noisy data, then same data with HMM fit on top
        print("\n[Plotting original noisy data and HMM fit overlay...]")
        plot_fitted_hmm_on_noisy_data(
            model=model,
            Y_all=hmm_res["Y_all"],
            lengths=hmm_res["lengths"],
            traces=traces,
            out_path=outdir / "hmm_fitted_on_noisy_data.png",
            state_labels=state_labels,
            feature_name=args.feature,
        )
        print("\n[Plotting individual traces with state assignments...]")
        # Per-trace visuals and dwell times
        states_per_trace = hmm_res["states_per_trace"]

        for i, (tr, z) in enumerate(zip(traces, states_per_trace)):
            out_png = outdir / f"{tr.filename.stem}_hmm_states.png"
            plot_trace_with_states(
                tr,
                z,
                out_path=out_png,
                title=f"HMM states ({args.feature}) - {tr.filename.name}",
                state_labels=state_labels,
            )
            if i == 0:
                print(f"  - Trace {i+1}: {tr.filename.name}")
        print(f"  ✓ Saved {len(traces)} trace plots")

        print(f"\n{'='*70}")
        print("DWELL TIME EXTRACTION")
        print(f"{'='*70}")
        print("\n[Extracting dwell times from state sequences...]")
        print("  Dwell time = duration spent in a state before transitioning")
        print(f"  Processing {len(states_per_trace)} traces...\n")
        
        # Dwell-time histograms (pooled across traces)
        all_dwells: Dict[int, List[int]] = {}
        for i, z in enumerate(states_per_trace):
            verbose = (i == 0)  # Show details for first trace
            if verbose:
                print(f"  Trace {i+1}:")
            dw = extract_dwell_times(z, verbose=verbose)
            for k, lst in dw.items():
                all_dwells.setdefault(k, []).extend(lst)
            if verbose:
                print(f"    Total dwell events: {sum(len(lst) for lst in dw.values())}")
        
        print(f"\n[Pooling dwell times across all traces...]")
        print(f"  Total dwell events per state (across all traces):")
        for k, lst in sorted(all_dwells.items()):
            label = state_labels.get(k, f"State {k}")
            mean_dwell = np.mean(lst) if len(lst) > 0 else 0
            median_dwell = np.median(lst) if len(lst) > 0 else 0
            std_dwell = np.std(lst) if len(lst) > 1 else 0
            print(f"    {label}:")
            print(f"      - Count: {len(lst)} dwell events")
            print(f"      - Mean duration: {mean_dwell:.2f} time steps")
            print(f"      - Median duration: {median_dwell:.2f} time steps")
            print(f"      - Std duration: {std_dwell:.2f} time steps")

        # Use median dt from first trace for approximate time units
        print(f"\n[Converting dwell times to seconds...]")
        if len(traces[0].time) > 1:
            dt_med = float(np.median(np.diff(traces[0].time)))
        else:
            dt_med = 1.0
        print(f"  Median time step (dt): {dt_med:.6f} seconds")
        print(f"  Converting dwell times from time steps to seconds...")
        
        print(f"\n[Plotting dwell time histogram...]")
        plot_dwell_hist(
            all_dwells,
            out_path=outdir / "hmm_dwell_hist.png",
            dt=dt_med,
            state_labels=state_labels,
        )
        print(f"  ✓ Saved: hmm_dwell_hist.png")

        # --------------------------------------------------------------
        # Dwell-time survival + first-passage density (HMM-specific)
        print("\n[Plotting dwell-time survival and first-passage density...]")
        pooled_steps = []
        for lst in all_dwells.values():
            pooled_steps.extend(lst)
        pooled_steps = np.asarray(pooled_steps, dtype=float)

        if pooled_steps.size > 0:
            dwell_seconds = pooled_steps * dt_med
            surv_path = outdir / "hmm_dwell_survival.png"
            fp_path = outdir / "hmm_first_passage_density.png"
            plot_hmm_dwell_time_survival(dwell_seconds, out_path=surv_path, show=False)
            plot_hmm_first_passage_density(
                dwell_seconds,
                out_path=fp_path,
                show=False,
                n_bins=200,
            )
            print("  ✓ Saved: survival + first-passage density plots in HMM output directory")
        else:
            print("  (No dwell events to plot survival / first-passage density.)")

        print(f"{'='*70}\n")
    print(f"\nDone. Outputs saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
