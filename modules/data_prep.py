import yfinance as yf
import pandas as pd
import numpy as np

def load_price_data(tickers, start_date, end_date):
    """
    Load adjusted close price data for given tickers and date range.
    """
    # Ensure we have a Python list of strings
    if isinstance(tickers, (pd.Series, pd.Index)):
        tickers = tickers.astype(str).tolist()
    elif isinstance(tickers, str):
        raise ValueError("tickers must be a list of tickers, not a single string")

    tickers = [str(t).strip().upper() for t in tickers if str(t).strip()]

    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        progress=False,
        group_by="column",
        threads=True,
    )

    # Extract adjusted close robustly
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"].copy()
    else:
        # single ticker case
        prices = data[["Adj Close"]].copy()
        prices.columns = [tickers[0]]

    prices = prices.sort_index()

    # Drop tickers with no data at all
    prices = prices.dropna(axis=1, how="all")

    print(f"Requested {len(tickers)} tickers; loaded {prices.shape[1]} with data.")
    return prices

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

def get_correlation_for_date(target_date, returns_df, epoch_size, compute_correlation_matrix):
    """Return correlation matrix for the epoch ending closest to target_date."""
    target = pd.Timestamp(target_date)

    idx = returns_df.index.searchsorted(target)
    end_idx = min(idx, len(returns_df) - 1)
    start_idx = max(0, end_idx - epoch_size + 1)

    epoch_returns = returns_df.iloc[start_idx:end_idx + 1].values
    C = compute_correlation_matrix(epoch_returns)
    return C, returns_df.index[end_idx]
