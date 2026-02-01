import numpy as np
import pandas as pd
from scipy import linalg
from modules.data_prep import correlation_to_A_matrix
from modules.data_prep import compute_correlation_matrix

def compute_eigen_centrality(A):
    """
    Compute eigen-centrality from matrix A.
    Returns normalized eigenvector of largest eigenvalue.
    p_i >= 0 and sum(p_i) = 1
    """
    eigenvalues, eigenvectors = linalg.eigh(A)
    # Get eigenvector for largest eigenvalue
    idx_max = np.argmax(eigenvalues)
    v = eigenvectors[:, idx_max]

    # Ensure non-negative
    v = np.abs(v)

    # Normalize so sum = 1
    p = v / np.sum(v)
    return p

def compute_entropy(p):
    """
    Compute Shannon entropy: H = -sum(p_i * ln(p_i))
    """
    p_nonzero = p[p > 1e-12]
    return -np.sum(p_nonzero * np.log(p_nonzero))

def decompose_correlation_matrix(C):
    """
    Decompose C into market mode C_M and group-random mode C_GR.
    
    C = C_M + C_GR
    C_M = λ₁|e₁⟩⟨e₁|  (rank-1 matrix from largest eigenvalue)
    C_GR = Σᵢ₌₂ᴺ λᵢ|eᵢ⟩⟨eᵢ|  (all other eigenvalues)
    """
    C = np.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0)
    C = (C + C.T) / 2
    np.fill_diagonal(C, 1.0)
    eigenvalues, eigenvectors = linalg.eigh(C)
    
    # Sort by eigenvalue (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    N = len(eigenvalues)
    
    # Market mode: C_M = λ₁ * |e₁⟩⟨e₁|
    e1 = eigenvectors[:, 0].reshape(-1, 1)
    C_M = eigenvalues[0] * (e1 @ e1.T)
    
    # Group-Random mode: C_GR = sum over i=2 to N
    C_GR = np.zeros_like(C)
    for i in range(1, N):
        ei = eigenvectors[:, i].reshape(-1, 1)
        C_GR += eigenvalues[i] * (ei @ ei.T)
    
    return C_M, C_GR, eigenvalues, eigenvectors

def compute_three_entropies(C):
    """
    Compute all three entropies as in the paper:
    - H: from full correlation matrix C
    - H_M: from market mode C_M
    - H_GR: from group-random mode C_GR
    """
    # Decompose C
    C_M, C_GR, eigenvalues, eigenvectors = decompose_correlation_matrix(C)
    
    # Convert to A matrices (squared entries)
    A = correlation_to_A_matrix(C)
    A_M = correlation_to_A_matrix(C_M)
    A_GR = correlation_to_A_matrix(C_GR)
    
    # Compute eigen-centralities
    p = compute_eigen_centrality(A)
    p_M = compute_eigen_centrality(A_M)
    p_GR = compute_eigen_centrality(A_GR)
    
    # Compute entropies
    H = compute_entropy(p)
    H_M = compute_entropy(p_M)
    H_GR = compute_entropy(p_GR)
    
    return H, H_M, H_GR, eigenvalues


# ANALYSIS SECTION

def rolling_window_analysis(returns_df, epoch_size=40, shift=20):
    """
    Computes H, H_M, H_GR for each epoch.
    """
    returns = returns_df.values
    dates = returns_df.index
    N = returns.shape[1]
    H_max = np.log(N)  # Wishart limit
    
    results = []
    start = 0
    
    while start + epoch_size <= len(returns):
        epoch_returns = returns[start:start + epoch_size]
        epoch_end_date = dates[start + epoch_size - 1]
        
        # Correlation matrix
        C = compute_correlation_matrix(epoch_returns)
        
        # Three entropies
        H, H_M, H_GR, eigenvalues = compute_three_entropies(C)
        
        # Mean market correlation μ
        upper_tri = np.triu_indices(N, k=1)
        mu = np.mean(C[upper_tri])
        
        # Phase space coordinates
        H_minus_HM = H - H_M
        H_minus_HGR = H - H_GR
        HM_minus_HGR = H_M - H_GR
        
        # Largest eigenvalue
        lambda_max = eigenvalues[0]
        
        results.append({
            'date': epoch_end_date,
            'H': H,
            'H_M': H_M,
            'H_GR': H_GR,
            'H_max': H_max,
            'H_minus_HM': H_minus_HM,
            'H_minus_HGR': H_minus_HGR,
            'HM_minus_HGR': HM_minus_HGR,
            'abs_H_minus_HM': np.abs(H_minus_HM),
            'abs_H_minus_HGR': np.abs(H_minus_HGR),
            'mu': mu,  # mean correlation
            'lambda_max': lambda_max,
        })
        
        start += shift
        
    return pd.DataFrame(results)


def identify_regimes(df, events):
    """Label epochs based on known market events."""
    df = df.copy()
    df['regime'] = 'normal'
    df['event_name'] = ''
    
    for start, end, name in events['crashes']:
        mask = (df['date'] >= start) & (df['date'] <= end)
        df.loc[mask, 'regime'] = 'crash'
        df.loc[mask, 'event_name'] = name
    
    for start, end, name in events['bubbles']:
        mask = (df['date'] >= start) & (df['date'] <= end)
        df.loc[mask, 'regime'] = 'bubble'
        df.loc[mask, 'event_name'] = name
    
    threshold_HM = df['abs_H_minus_HM'].quantile(0.90)
    threshold_HGR = df['abs_H_minus_HGR'].quantile(0.90)
    
    type1_mask = (
        (df['abs_H_minus_HM'] > threshold_HM) &
        (df['abs_H_minus_HGR'] > threshold_HGR) &
        (df['regime'] == 'normal')
    )
    df.loc[type1_mask, 'regime'] = 'type-1'
    
    return df

def get_transition_sequence(results_df, event_date, n_before=3, n_after=3):
    """Return epochs around the epoch closest to event_date."""
    results_df = results_df.reset_index(drop=True)
    date_diffs = (results_df['date'] - pd.Timestamp(event_date)).abs()
    center_idx = date_diffs.idxmin()
    start_idx = max(0, center_idx - n_before)
    end_idx = min(len(results_df) - 1, center_idx + n_after)
    return results_df.iloc[start_idx:end_idx + 1]

from scipy.optimize import curve_fit

def assign_phase_regimes(results):
    """
    Assign phase-space-based regimes.
    Returns a copy with a new column: 'phase_regime'.
    """
    df = results.copy()

    H_HM_low = df['abs_H_minus_HM'].quantile(0.15)
    H_HM_high = df['abs_H_minus_HM'].quantile(0.85)
    H_HGR_low = df['abs_H_minus_HGR'].quantile(0.15)
    H_HGR_high = df['abs_H_minus_HGR'].quantile(0.85)

    df['phase_regime'] = 'Normal'

    crash_mask = (df['abs_H_minus_HM'] < H_HM_low) & \
                 (df['abs_H_minus_HGR'] > df['abs_H_minus_HGR'].quantile(0.5))
    df.loc[crash_mask, 'phase_regime'] = 'Crash'

    type1_mask = (df['abs_H_minus_HM'] > H_HM_high) & \
                 (df['abs_H_minus_HGR'] > H_HGR_high)
    df.loc[type1_mask, 'phase_regime'] = 'Type-1'

    type2_mask = (df['abs_H_minus_HM'] > df['abs_H_minus_HM'].quantile(0.5)) & \
                 (df['abs_H_minus_HGR'] < H_HGR_low)
    df.loc[type2_mask, 'phase_regime'] = 'Type-2'

    anomaly_mask = (df['abs_H_minus_HM'] < df['abs_H_minus_HM'].quantile(0.3)) & \
                   (df['abs_H_minus_HGR'] < df['abs_H_minus_HGR'].quantile(0.3))
    df.loc[anomaly_mask, 'phase_regime'] = 'Anomaly'

    return df


def exp_func(x, alpha, beta):
    """Exponential scaling function: alpha * exp(-beta * x)."""
    return alpha * np.exp(-beta * x)


def fit_scaling_relation(results_plot, verbose=True):
    """
    Fit (H - H_M) ~ alpha * exp(-beta * mu).

    Returns a dict with keys:
      - ok (bool)
      - popt (np.ndarray) or None
      - pcov (np.ndarray) or None
      - x_fit (np.ndarray) or None
      - y_fit (np.ndarray) or None
      - mask (np.ndarray boolean used for fit)
      - error (str) if failed
    """
    try:
        mask = (results_plot['H_minus_HM'] > 0.001) & (results_plot['mu'] > 0.01)
        x_data = results_plot.loc[mask, 'mu'].values
        y_data = results_plot.loc[mask, 'H_minus_HM'].values

        popt, pcov = curve_fit(exp_func, x_data, y_data, p0=[0.85, 10], maxfev=10000)

        x_fit = np.linspace(0.01, results_plot['mu'].max(), 100)
        y_fit = exp_func(x_fit, *popt)

        if verbose:
            print(f"Scaling fit: (H - H_M) ~ {popt[0]:.3f} * exp(-{popt[1]:.2f} * μ)")
            print("Paper values: α ≈ 0.85, β ≈ 10.22")

        return {
            "ok": True,
            "popt": popt,
            "pcov": pcov,
            "x_fit": x_fit,
            "y_fit": y_fit,
            "mask": mask,
            "error": None,
        }

    except Exception as e:
        if verbose:
            print(f"Fit failed: {e}")
        return {
            "ok": False,
            "popt": None,
            "pcov": None,
            "x_fit": None,
            "y_fit": None,
            "mask": None,
            "error": str(e),
        }


def assign_phase_regimes_exact(results):
    """
    Assign regimes based on phase space position (matching the notebook logic).
    Adds/overwrites column: 'phase_regime'.
    """
    df = results.copy()

    H_HM_low = df['abs_H_minus_HM'].quantile(0.15)
    H_HM_high = df['abs_H_minus_HM'].quantile(0.85)
    H_HGR_low = df['abs_H_minus_HGR'].quantile(0.15)
    H_HGR_high = df['abs_H_minus_HGR'].quantile(0.85)

    df['phase_regime'] = 'Normal'

    crash_mask = (df['abs_H_minus_HM'] < H_HM_low) & \
                 (df['abs_H_minus_HGR'] > df['abs_H_minus_HGR'].quantile(0.5))
    df.loc[crash_mask, 'phase_regime'] = 'Crash'

    type1_mask = (df['abs_H_minus_HM'] > H_HM_high) & \
                 (df['abs_H_minus_HGR'] > H_HGR_high)
    df.loc[type1_mask, 'phase_regime'] = 'Type-1'

    type2_mask = (df['abs_H_minus_HM'] > df['abs_H_minus_HM'].quantile(0.5)) & \
                 (df['abs_H_minus_HGR'] < H_HGR_low)
    df.loc[type2_mask, 'phase_regime'] = 'Type-2'

    anomaly_mask = (df['abs_H_minus_HM'] < df['abs_H_minus_HM'].quantile(0.3)) & \
                   (df['abs_H_minus_HGR'] < df['abs_H_minus_HGR'].quantile(0.3))
    df.loc[anomaly_mask, 'phase_regime'] = 'Anomaly'

    return df


def get_transition_sequence(results_df, event_date, n_before=3, n_after=3):
    """Return epochs around the epoch closest to event_date."""
    results_df = results_df.reset_index(drop=True)
    date_diffs = (results_df['date'] - pd.Timestamp(event_date)).abs()
    center_idx = date_diffs.idxmin()
    start_idx = max(0, center_idx - n_before)
    end_idx = min(len(results_df) - 1, center_idx + n_after)
    return results_df.iloc[start_idx:end_idx + 1]


from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PhaseThresholds:
    """Quantile thresholds used to define phase-space regions."""
    H_HM_25: float
    H_HM_35: float
    H_HM_50: float
    H_HM_80: float
    H_HGR_20: float
    H_HGR_35: float
    H_HGR_50: float
    H_HGR_80: float


def compute_phase_thresholds_quantiles(df: pd.DataFrame) -> PhaseThresholds:
    """
    Compute the exact quantile thresholds used in the 'new notebook' cell.

    Requires columns:
      - 'abs_H_minus_HM'
      - 'abs_H_minus_HGR'
    """
    return PhaseThresholds(
        H_HM_25=float(df["abs_H_minus_HM"].quantile(0.25)),
        H_HM_35=float(df["abs_H_minus_HM"].quantile(0.35)),
        H_HM_50=float(df["abs_H_minus_HM"].quantile(0.50)),
        H_HM_80=float(df["abs_H_minus_HM"].quantile(0.80)),
        H_HGR_20=float(df["abs_H_minus_HGR"].quantile(0.20)),
        H_HGR_35=float(df["abs_H_minus_HGR"].quantile(0.35)),
        H_HGR_50=float(df["abs_H_minus_HGR"].quantile(0.50)),
        H_HGR_80=float(df["abs_H_minus_HGR"].quantile(0.80)),
    )


def assign_phase_regimes_quantiles(
    results: pd.DataFrame,
    *,
    reset_index: bool = True,
    thresholds: Optional[PhaseThresholds] = None,
) -> Tuple[pd.DataFrame, PhaseThresholds]:
    """
    Assign phase_regime using the NEW notebook's quantile rules.

    Returns:
      (df_labeled, thresholds_used)

    Requires columns:
      - 'abs_H_minus_HM'
      - 'abs_H_minus_HGR'
      - 'date' (only needed later for event sequences; not needed for labeling itself)
    """
    df = results.copy()
    if reset_index:
        df = df.reset_index(drop=True)

    if thresholds is None:
        thresholds = compute_phase_thresholds_quantiles(df)

    df["phase_regime"] = "Normal"

    crash_mask = (df["abs_H_minus_HM"] < thresholds.H_HM_25) & (df["abs_H_minus_HGR"] > thresholds.H_HGR_50)
    df.loc[crash_mask, "phase_regime"] = "Crash"

    type1_mask = (df["abs_H_minus_HM"] > thresholds.H_HM_80) & (df["abs_H_minus_HGR"] > thresholds.H_HGR_80)
    df.loc[type1_mask, "phase_regime"] = "Type-1"

    type2_mask = (df["abs_H_minus_HM"] > thresholds.H_HM_50) & (df["abs_H_minus_HGR"] < thresholds.H_HGR_20)
    df.loc[type2_mask, "phase_regime"] = "Type-2"

    anomaly_mask = (
        (df["abs_H_minus_HM"] < thresholds.H_HM_35)
        & (df["abs_H_minus_HGR"] < thresholds.H_HGR_35)
        & (df["phase_regime"] == "Normal")
    )
    df.loc[anomaly_mask, "phase_regime"] = "Anomaly"

    return df, thresholds


def get_event_sequence(
    df: pd.DataFrame,
    event_date: str | pd.Timestamp,
    *,
    n_before: int = 3,
    n_after: int = 3,
) -> Tuple[pd.DataFrame, int]:
    """
    Return a slice of rows around the epoch closest to event_date.

    Returns:
      (seq_df, event_pos_in_seq)
    """
    event_ts = pd.Timestamp(event_date)
    date_diffs = (pd.to_datetime(df["date"]) - event_ts).abs()
    event_idx = int(date_diffs.idxmin())

    start_idx = max(0, event_idx - n_before)
    end_idx = min(len(df) - 1, event_idx + n_after)

    seq = df.iloc[start_idx : end_idx + 1].copy()
    event_pos = event_idx - start_idx
    return seq, int(event_pos)


def classify_transition(seq: pd.DataFrame) -> Tuple[str, str]:
    """
    Classify transition type from a sequence of phase_regime labels.

    Returns:
      (transition_label, title_color)
    """
    regimes = seq["phase_regime"].tolist()
    has_crash = "Crash" in regimes
    has_anomaly = "Anomaly" in regimes
    has_normal = "Normal" in regimes
    has_type1 = "Type-1" in regimes
    has_type2 = "Type-2" in regimes

    if has_crash and (has_normal or has_anomaly):
        return "CRASH TRANSITION", "green"
    elif has_anomaly and has_normal and not has_crash:
        return "ANOMALY TRANSITION", "orange"
    elif has_type1 or has_type2:
        return "TYPE TRANSITION", "blue"
    else:
        return "NO CLEAR TRANSITION", "red"


def summarize_event_transitions(
    results_labeled: pd.DataFrame,
    events: List[Tuple[str, str]],
    *,
    n_before: int = 3,
    n_after: int = 3,
) -> pd.DataFrame:
    """
    Build a tidy summary table for event transition analysis.

    events: list of (event_date, event_name)

    Returns a DataFrame with:
      event_name, event_date, transition_type, sequence, event_abs_H_minus_HM, event_abs_H_minus_HGR
    """
    rows = []
    for event_date, event_name in events:
        seq, event_pos = get_event_sequence(results_labeled, event_date, n_before=n_before, n_after=n_after)
        transition_type, _color = classify_transition(seq)

        regimes = seq["phase_regime"].tolist()
        event_row = seq.iloc[event_pos]

        rows.append(
            {
                "event_name": event_name,
                "event_date": str(pd.Timestamp(event_date).date()),
                "transition_type": transition_type,
                "sequence": " → ".join(regimes),
                "event_abs_H_minus_HM": float(event_row["abs_H_minus_HM"]),
                "event_abs_H_minus_HGR": float(event_row["abs_H_minus_HGR"]),
            }
        )
    return pd.DataFrame(rows)


