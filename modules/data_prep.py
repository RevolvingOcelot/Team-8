import yfinance as yf
import pandas as pd
import numpy as np


def compute_log_returns(prices):
    """Compute log returns: r_i(τ) = ln P_i(τ) - ln P_i(τ-1)"""
    return np.log(prices / prices.shift(1)).dropna()

def compute_correlation_matrix(returns):
    """Compute Pearson correlation matrix C with NaN handling."""
    C = np.corrcoef(returns.T)
    C = np.nan_to_num(C, nan=0.0, posinf=1.0, neginf=-1.0)
    C = (C + C.T) / 2
    np.fill_diagonal(C, 1.0)
    
    return C

def correlation_to_A_matrix(C):
    """
    Convert correlation matrix C to matrix A = |C|^2.
    This ensures non-negative entries for Perron-Frobenius theorem.
    Paper uses n=2 (squared).
    """
    return np.abs(C) ** 2

def get_correlation_for_date(target_date, returns_df, epoch_size):
    """Return correlation matrix for the epoch ending closest to target_date."""
    target = pd.Timestamp(target_date)

    idx = returns_df.index.searchsorted(target)
    end_idx = min(idx, len(returns_df) - 1)
    start_idx = max(0, end_idx - epoch_size + 1)

    epoch_returns = returns_df.iloc[start_idx:end_idx + 1].values
    C = compute_correlation_matrix(epoch_returns)
    return C, returns_df.index[end_idx]
