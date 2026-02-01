import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd
from modules.entropy import assign_phase_regimes_quantiles 
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from scipy.interpolate import griddata
from modules.entropy import get_event_sequence, classify_transition, PhaseThresholds
from modules.config import PHASE_MARKERS

def set_plot_style():
    """Set global matplotlib style for the project."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 11


def plot_correlation_decomposition(C, C_M, C_GR, title='Eigenvalue Decomposition of Correlation Matrix'):
    """Visualize correlation matrix + decomposition."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    titles = ['Full C', 'Market Mode C_M', 'Group-Random C_GR', 'C_M + C_GR']
    matrices = [C, C_M, C_GR, C_M + C_GR]

    for ax, t, mat in zip(axes, titles, matrices):
        im = ax.imshow(mat, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title(t, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.colorbar(im, ax=axes, shrink=0.8, label='Correlation')
    if title:
        plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_regime_correlation_matrices(returns, regime_examples, epoch_size, get_corr_fn, compute_corr_fn):
    """Plot correlation matrices for specified regimes/dates."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    for ax, (regime, date) in zip(axes, regime_examples.items()):
        try:
            C, actual_date = get_corr_fn(date, returns, epoch_size)
            im = ax.imshow(C, cmap='jet', vmin=-1, vmax=1, aspect='equal')

            date_str = actual_date.strftime('%d-%m-%y')
            ax.set_title(f'{regime}\n{date_str}', fontsize=14, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])

            upper_tri = np.triu_indices(C.shape[0], k=1)
            mu = np.mean(C[upper_tri])
            ax.set_xlabel(f'μ = {mu:.2f}', fontsize=11)

        except Exception as e:
            ax.set_title(f'{regime}\n(Error)')
            print(f"Error for {regime}: {e}")

    cbar = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label('Correlation', fontsize=12)
    plt.suptitle('Correlation Matrices by Market Regime', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_ranked_eigen_centralities(C, C_M, C_GR, correlation_to_A_matrix, compute_eigen_centrality, title='Ranked Eigen-centralities'):

    """Plot ranked eigen-centralities for full, market, and group-random matrices."""
    fig, ax = plt.subplots(figsize=(8, 5))

    A = correlation_to_A_matrix(C)
    A_M = correlation_to_A_matrix(C_M)
    A_GR = correlation_to_A_matrix(C_GR)

    p = compute_eigen_centrality(A)
    p_M = compute_eigen_centrality(A_M)
    p_GR = compute_eigen_centrality(A_GR)

    ranks = np.arange(1, len(p) + 1)
    ax.plot(ranks, np.sort(p)[::-1], 'k-', linewidth=2, label='C (full)')
    ax.plot(ranks, np.sort(p_M)[::-1], 'c-', linewidth=2, label='C_M (market)')
    ax.plot(ranks, np.sort(p_GR)[::-1], 'gray', linewidth=2, label='C_GR (group-random)')

    ax.set_xlabel('Rank', fontsize=12)
    ax.set_ylabel('Eigen-centrality $p_i$', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.set_xlim([1, len(p)])
    plt.tight_layout()
    plt.show()



def plot_market_indicators(results, crisis_periods=None, major_crashes=None):
    """
    Plot market indicators:
    μ, H-H_M, -ln(H-H_M), and lambda_max with shaded crisis bands.
    """
    if crisis_periods is None:
        crisis_periods = []
    if major_crashes is None:
        major_crashes = []

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    # Panel 1: Mean correlation μ
    ax = axes[0]
    ax.plot(results['date'], results['mu'], color='orange', linewidth=0.8, label='μ')
    ax.set_ylabel('μ (mean corr)', fontsize=11)
    ax.set_title('Evolution of Market Indicators', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 2: H - H_M
    ax = axes[1]
    ax.plot(results['date'], results['H_minus_HM'], color='blue', linewidth=0.8, label='H - H$_M$')
    ax.set_ylabel('H - H$_M$', fontsize=11)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 3: -ln(H - H_M)
    ax = axes[2]
    neg_ln = -np.log(results['H_minus_HM'].clip(lower=1e-10))
    ax.plot(results['date'], neg_ln, color='red', linewidth=0.8, label='-ln(H - H$_M$) "Fear Gauge"')
    ax.set_ylabel('-ln(H - H$_M$)', fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Panel 4: Largest eigenvalue
    ax = axes[3]
    ax.plot(results['date'], results['lambda_max'], color='green', linewidth=0.8, label='λ$_{max}$')
    ax.set_ylabel('λ$_{max}$', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add crisis bands to all panels
    for ax in axes:
        first = True
        for start, end, _name in crisis_periods:
            try:
                if first:
                    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                               color='lightcoral', alpha=0.3, label='Crisis Period')
                    first = False
                else:
                    ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                               color='lightcoral', alpha=0.3)
            except Exception:
                pass

    # Add vertical lines for major crashes
    for date, label in major_crashes:
        try:
            for ax in axes:
                ax.axvline(pd.Timestamp(date), color='red', linestyle='--', linewidth=1, alpha=0.7)

            axes[0].annotate(
                label,
                xy=(pd.Timestamp(date), axes[0].get_ylim()[1]),
                fontsize=8, ha='center', va='bottom',
                color='red', fontweight='bold'
            )
        except Exception:
            pass

    plt.tight_layout()
    plt.show()




def plot_scaling_relation(results_plot, fit=None, markers=None):
    """
    Plot scaling relation:
      x = mu
      y = H - H_M

    Parameters
    ----------
    results_plot : pd.DataFrame
        Must contain: 'mu', 'H_minus_HM', 'phase_regime'
    fit : dict or None
        Output from fit_scaling_relation(...)
    markers : dict
        Mapping regime -> (marker, color, size)
    """
    if markers is None:
        markers = PHASE_MARKERS

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot colored by regime
    for regime, (marker, color, size) in markers.items():
        mask = results_plot['phase_regime'] == regime
        if mask.any():
            ax.scatter(
                results_plot.loc[mask, 'mu'],
                results_plot.loc[mask, 'H_minus_HM'],
                marker=marker, c=color, s=size, label=regime,
                alpha=0.7, edgecolors='white', linewidth=0.3
            )

    # WOE reference point
    ax.scatter([0.05], [0.65], c='white', marker='*', s=250,
               edgecolors='black', linewidth=1.5, label='WOE', zorder=5)

    # Fit line
    if fit is not None and fit.get("ok", False):
        popt = fit["popt"]
        x_fit = fit["x_fit"]
        y_fit = fit["y_fit"]
        ax.plot(x_fit, y_fit, 'k--', linewidth=2,
                label=f'Fit: α={popt[0]:.2f}, β={popt[1]:.2f}')

    ax.set_xlabel('μ', fontsize=14)
    ax.set_ylabel('H - H$_M$', fontsize=14)
    ax.set_title('Scaling Relation (with log-scale insert)', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.2])

    # Inset with log scale
    inset = ax.inset_axes([0.5, 0.5, 0.45, 0.45])

    for regime, (marker, color, size) in markers.items():
        mask = results_plot['phase_regime'] == regime
        if mask.any():
            inset.scatter(
                results_plot.loc[mask, 'mu'],
                results_plot.loc[mask, 'H_minus_HM'].clip(lower=1e-4),
                marker=marker, c=color, s=size * 0.5, alpha=0.7
            )

    if fit is not None and fit.get("ok", False):
        inset.plot(fit["x_fit"], fit["y_fit"], 'k--', linewidth=1.5)

    inset.set_yscale('log')
    inset.set_xlabel('μ', fontsize=10)
    inset.set_ylabel('H - H$_M$', fontsize=10)
    inset.set_xlim([0, 1])
    inset.set_ylim([1e-4, 1e0])
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_entropy_differences(results, crash_dates, high_mu_q=0.85, very_high_mu_q=0.95):
    """
    Plot time evolution of entropy differences with high-correlation bands.
    Expects columns:
      'date', 'mu', 'abs_H_minus_HM', 'HM_minus_HGR', 'abs_H_minus_HGR'
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Define high correlation periods for colored bands
    high_corr_threshold = results['mu'].quantile(high_mu_q)
    very_high_corr = results['mu'].quantile(very_high_mu_q)

    high_corr_mask = results['mu'] > high_corr_threshold
    very_high_corr_mask = results['mu'] > very_high_corr

    # plot 1: |H - H_M|
    ax = axes[0]
    ax.plot(results['date'], results['abs_H_minus_HM'], color='magenta', linewidth=0.8, label='|H - H$_M$|')
    ax.fill_between(
        results['date'], 0, results['abs_H_minus_HM'].max() * 1.1,
        where=high_corr_mask, alpha=0.15, color='cyan', label='High μ (>85th %ile)'
    )
    ax.fill_between(
        results['date'], 0, results['abs_H_minus_HM'].max() * 1.1,
        where=very_high_corr_mask, alpha=0.25, color='pink', label='Very high μ (>95th %ile)'
    )
    ax.set_ylabel('|H - H$_M$|', fontsize=12, color='magenta')
    ax.set_title('Evolution of Entropy Differences', fontsize=14)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='y', labelcolor='magenta')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # plot 2: |H_M - H_GR|
    ax = axes[1]
    hm_minus_hgr_abs = results['HM_minus_HGR'].abs()
    ax.plot(results['date'], hm_minus_hgr_abs, color='blue', linewidth=0.8, label='|H$_M$ - H$_{GR}$|')
    ax.fill_between(
        results['date'], 0, hm_minus_hgr_abs.max() * 1.1,
        where=high_corr_mask, alpha=0.15, color='cyan'
    )
    ax.fill_between(
        results['date'], 0, hm_minus_hgr_abs.max() * 1.1,
        where=very_high_corr_mask, alpha=0.25, color='pink'
    )
    ax.set_ylabel('|H$_M$ - H$_{GR}$|', fontsize=12, color='blue')
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='y', labelcolor='blue')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # plot 3: |H - H_GR|
    ax = axes[2]
    ax.plot(results['date'], results['abs_H_minus_HGR'], color='green', linewidth=0.8, label='|H - H$_{GR}$|')
    ax.fill_between(
        results['date'], 0, results['abs_H_minus_HGR'].max() * 1.1,
        where=high_corr_mask, alpha=0.15, color='cyan'
    )
    ax.fill_between(
        results['date'], 0, results['abs_H_minus_HGR'].abs().max() * 1.1,
        where=very_high_corr_mask, alpha=0.25, color='pink'
    )
    ax.set_ylabel('|H - H$_{GR}$|', fontsize=12, color='green')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylim(bottom=0)
    ax.tick_params(axis='y', labelcolor='green')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Add crash markers with labels
    for ax_idx, ax in enumerate(axes):
        for cd, name in crash_dates:
            try:
                ax.axvline(pd.Timestamp(cd), color='red', linestyle='--', alpha=0.7, linewidth=1.5)
                if ax_idx == 0:
                    ax.annotate(
                        name,
                        xy=(pd.Timestamp(cd), ax.get_ylim()[1] * 0.95),
                        fontsize=8, ha='center', color='red', fontweight='bold'
                    )
            except Exception:
                pass

    plt.tight_layout()
    plt.show()

TEMP_PLOT_DATA = [
    ('mu', 'μ', 'purple'),
    ('H', 'H', 'blue'),
    ('H_M', 'H$_M$', 'orange'),
    ('H_GR', 'H$_{GR}$', 'olive'),
    ('H_minus_HM', 'H - H$_M$', 'magenta'),
    ('H_minus_HGR', 'H - H$_{GR}$', 'darkblue'),
    ('HM_minus_HGR', 'H$_M$ - H$_{GR}$', 'teal'),
]


def plot_full_temporal_evolution(results, plot_data = TEMP_PLOT_DATA , regime_line_date='2001-04-01'):
    """
    Plot temporal evolution (8 panels).

    Expects columns:
      'date' plus the columns listed in plot_data.
    """
    fig, axes = plt.subplots(8, 1, figsize=(14, 20), sharex=True)

    # Plot first 7 variables
    for ax, (col, label, color) in zip(axes[:7], plot_data):
        ax.plot(results['date'], results[col], color=color, linewidth=0.8)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(True, alpha=0.3)

    #-ln(H - H_M)
    ax = axes[7]
    neg_ln_H = -np.log(results['H_minus_HM'].clip(lower=1e-10))
    ax.plot(results['date'], neg_ln_H, color='cyan', linewidth=0.8)
    ax.set_ylabel('-ln(H - H$_M$)', fontsize=11)
    ax.set_xlabel('Date', fontsize=11)
    ax.grid(True, alpha=0.3)

    # Regime change vertical line
    for ax in axes:
        ax.axvline(pd.Timestamp(regime_line_date), color='gray', linestyle='--', alpha=0.5, linewidth=1)

    plt.suptitle('Temporal Evolution of Market Indicators', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


def plot_entropy_evolution(results, crash_dates, ylim=(4.2, 5.3)):
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(results['date'], results['H'], color='orange', linewidth=1, label='H', alpha=0.9)
    ax.plot(results['date'], results['H_M'], color='black', linewidth=1, label='H$_M$', alpha=0.9)
    ax.plot(results['date'], results['H_GR'], color='olive', linewidth=1, label='H$_{GR}$', alpha=0.9)

    H_max = results['H_max'].iloc[0]
    ax.axhline(
        y=H_max, color='red', linestyle='--', linewidth=1.5,
        label=f'H$_{{max}}$ = ln(N) = {H_max:.2f}'
    )

    for date, _ in crash_dates:
        try:
            ax.axvline(pd.Timestamp(date), color='red', linestyle=':', alpha=0.5, linewidth=1)
        except Exception:
            pass

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Entropy', fontsize=12)
    ax.set_title(
        'Dynamical Evolution of Entropies: H(τ), H$_M$(τ), H$_{GR}$(τ)',
        fontsize=14
    )
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(list(ylim))

    plt.tight_layout()
    plt.show()

def plot_3d_phase_space_wireframe(results_plot, colors_3d, grid_n=30):
    """
    3D phase space: plot H as a surface over (H_M, H_GR) with regime-colored scatter.
    Expects columns: 'H_M', 'H_GR', 'H', 'phase_regime'
    """
    fig = plt.figure(figsize=(16, 7))

    H_M_data = results_plot['H_M'].values
    H_GR_data = results_plot['H_GR'].values
    H_data = results_plot['H'].values

    H_M_range = np.linspace(H_M_data.min(), H_M_data.max(), grid_n)
    H_GR_range = np.linspace(H_GR_data.min(), H_GR_data.max(), grid_n)
    H_M_mesh, H_GR_mesh = np.meshgrid(H_M_range, H_GR_range)

    H_surface = griddata((H_M_data, H_GR_data), H_data, (H_M_mesh, H_GR_mesh), method='cubic')
    H_surface_nearest = griddata((H_M_data, H_GR_data), H_data, (H_M_mesh, H_GR_mesh), method='nearest')
    H_surface = np.where(np.isnan(H_surface), H_surface_nearest, H_surface)

    # TOP VIEW
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_wireframe(H_M_mesh, H_GR_mesh, H_surface,
                       color='olive', alpha=0.3, linewidth=0.5,
                       rstride=2, cstride=2)

    for regime, color in colors_3d.items():
        mask = results_plot['phase_regime'] == regime
        if mask.any():
            ax1.scatter(results_plot.loc[mask, 'H_M'],
                        results_plot.loc[mask, 'H_GR'],
                        results_plot.loc[mask, 'H'],
                        c=color, s=30, alpha=0.9, label=regime,
                        edgecolors='k', linewidth=0.3, zorder=10)

    ax1.set_xlabel('H$_M$', fontsize=11)
    ax1.set_ylabel('H$_{GR}$', fontsize=11)
    ax1.set_zlabel('H', fontsize=11)
    ax1.set_title('3D Phase Space (Top View)', fontsize=12)
    ax1.view_init(elev=75, azim=-90)
    ax1.legend(loc='upper left', fontsize=8)

    # SIDE VIEW
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_wireframe(H_M_mesh, H_GR_mesh, H_surface,
                       color='teal', alpha=0.3, linewidth=0.5,
                       rstride=2, cstride=2)

    for regime, color in colors_3d.items():
        mask = results_plot['phase_regime'] == regime
        if mask.any():
            ax2.scatter(results_plot.loc[mask, 'H_M'],
                        results_plot.loc[mask, 'H_GR'],
                        results_plot.loc[mask, 'H'],
                        c=color, s=30, alpha=0.9, label=regime,
                        edgecolors='k', linewidth=0.3, zorder=10)

    ax2.set_xlabel('H$_M$', fontsize=11)
    ax2.set_ylabel('H$_{GR}$', fontsize=11)
    ax2.set_zlabel('H', fontsize=11)
    ax2.set_title('3D Phase Space (Side View)', fontsize=12)
    ax2.view_init(elev=10, azim=-45)
    ax2.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_3d_entropy_differences(results_plot, markers_3d):
    """
    3D scatter of entropy differences:
      x = H - H_M
      y = H_M - H_GR
      z = H - H_GR
    Colored/marked by 'phase_regime'.
    """
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    for regime, (marker, color, size) in markers_3d.items():
        mask = results_plot['phase_regime'] == regime
        if mask.any():
            ax.scatter(
                results_plot.loc[mask, 'H_minus_HM'],
                results_plot.loc[mask, 'HM_minus_HGR'],
                results_plot.loc[mask, 'H_minus_HGR'],
                marker=marker, c=color, s=size, alpha=0.7, label=regime
            )

    ax.set_xlabel('H - H$_M$', fontsize=11)
    ax.set_ylabel('H$_M$ - H$_{GR}$', fontsize=11)
    ax.set_zlabel('H - H$_{GR}$', fontsize=11)
    ax.set_title('3D Phase Space: Entropy Differences', fontsize=14)
    ax.legend(loc='upper left', fontsize=9)
    ax.view_init(elev=25, azim=-45)

    plt.tight_layout()
    plt.show()

DEFAULT_MARKERS: Dict[str, Tuple[str, str, float]] = {
    "Crash": ("^", "red", 50),
    "Type-1": ("D", "deepskyblue", 40),
    "Type-2": ("s", "blue", 40),
    "Anomaly": ("o", "green", 40),
    "Normal": ("o", "gray", 15),
}


def plot_event_transition_grid(
    results_labeled: pd.DataFrame,
    events: List[Tuple[str, str]],
    thresholds: PhaseThresholds,
    *,
    markers: Optional[Dict[str, Tuple[str, str, float]]] = DEFAULT_MARKERS,
    n_before: int = 3,
    n_after: int = 3,
    ncols: int = 3,
    figsize: Tuple[int, int] = (15, 15),
    xlim: Tuple[float, float] = (1e-4, 1e0),
    ylim: Tuple[float, float] = (1e-4, 1e0),) -> None:

    """
    Plot a grid of phase-space transition paths around each event.

    Requires in results_labeled:
      - 'abs_H_minus_HM', 'abs_H_minus_HGR', 'phase_regime', 'date'
    """
    n = len(events)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)

    # Precompute scatter masks
    scatter_data = []
    for regime, (m, c, s) in markers.items():
        mask = results_labeled["phase_regime"] == regime
        scatter_data.append((regime, mask, m, c, s))

    for idx, ax in enumerate(axes):
        if idx >= n:
            ax.axis("off")
            continue

        event_date, event_name = events[idx]

        # Background scatter by regime
        for _regime, mask, marker, color, size in scatter_data:
            if mask.any():
                ax.scatter(
                    results_labeled.loc[mask, "abs_H_minus_HM"] + 1e-6,
                    results_labeled.loc[mask, "abs_H_minus_HGR"] + 1e-6,
                    marker=marker,
                    c=color,
                    s=size * 0.3,
                    alpha=0.3,
                    zorder=1,
                )

        # Threshold guide lines
        ax.axvline(thresholds.H_HM_25, color="red", linestyle=":", alpha=0.5, linewidth=1)
        ax.axvline(thresholds.H_HM_50, color="gray", linestyle=":", alpha=0.5, linewidth=1)
        ax.axhline(thresholds.H_HGR_50, color="red", linestyle=":", alpha=0.5, linewidth=1)
        ax.axhline(thresholds.H_HGR_35, color="green", linestyle=":", alpha=0.5, linewidth=1)

        # Event sequence path
        seq, event_pos = get_event_sequence(results_labeled, event_date, n_before=n_before, n_after=n_after)
        x = seq["abs_H_minus_HM"].to_numpy() + 1e-6
        y = seq["abs_H_minus_HGR"].to_numpy() + 1e-6

        transition_type, title_color = classify_transition(seq)

        ax.plot(x, y, "--", color="darkred", linewidth=2, zorder=4)
        ax.scatter(x, y, c="darkred", s=60, zorder=5, edgecolors="white", linewidth=1)
        ax.scatter(
            x[event_pos],
            y[event_pos],
            c="black",
            s=120,
            zorder=6,
            edgecolors="red",
            linewidth=2,
        )

        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.annotate(
                str(i + 1),
                (xi, yi),
                fontsize=8,
                fontweight="bold",
                color="darkred",
                xytext=(3, 3),
                textcoords="offset points",
                zorder=7,
            )

        ax.set_title(f"{event_name}\n{event_date}\n{transition_type}", fontsize=9, color=title_color)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(list(xlim))
        ax.set_ylim(list(ylim))
        ax.set_xlabel("|H - H$_M$|", fontsize=8)
        ax.set_ylabel("|H - H$_{GR}$|", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle("Phase Transitions for Major Market Events", fontsize=12, y=1.02)
    plt.show()


def plot_phase_transitions(
    results,
    *,
    crash_reference_date="1987-10-19",
    woe_point=(0.5, 0.005),
    print_tables=True,
    xlim=(1e-4, 1e0),
    ylim=(1e-4, 1e0),
    title="Phase Space with Order-Disorder Transitions",):

    """
    Single function version of the notebook cell:
      - labels phase_regime using assign_phase_regimes_quantiles (reused)
      - finds best RED crash-centered transition (closest to crash_reference_date)
      - finds best BLUE Type-1→Type-2 transition maximizing drop in |H-H_GR|
      - optionally prints the two 7-epoch tables
      - plots the full phase space with both sequences highlighted

    Parameters
    ----------
    results : pd.DataFrame
        Must contain: 'date', 'abs_H_minus_HM', 'abs_H_minus_HGR'
        (phase_regime will be assigned internally)
    crash_reference_date : str
        Date to anchor the RED crash transition (default '1987-10-19')
    print_tables : bool
        If True, print the same tables as the notebook cell.
    """

    # Reuse existing regime assignment from before
    results_plot, _thr = assign_phase_regimes_quantiles(results)
    results_plot = results_plot.reset_index(drop=True)


    # RED: crash transition anchored to reference date (same logic)
    crash_date = pd.Timestamp(crash_reference_date)
    best_red_idx = (results_plot["date"] - crash_date).abs().idxmin()

    # BLUE: best Type-1 + Type-2 7-epoch sequence maximizing drop in |H-H_GR|
    best_blue_idx = None
    best_blue_score = 0

    for i in range(3, len(results_plot) - 3):
        seq = results_plot.iloc[i - 3 : i + 4]
        regimes = seq["phase_regime"].tolist()

        has_type1 = "Type-1" in regimes
        has_type2 = "Type-2" in regimes

        if has_type1 and has_type2:
            t1_pos = [j for j, r in enumerate(regimes) if r == "Type-1"]
            t2_pos = [j for j, r in enumerate(regimes) if r == "Type-2"]

            if min(t1_pos) < max(t2_pos):
                drop = seq["abs_H_minus_HGR"].iloc[0] - seq["abs_H_minus_HGR"].iloc[-1]
                if drop > best_blue_score:
                    best_blue_score = drop
                    best_blue_idx = i

    # Extract sequences
    red_seq = results_plot.iloc[best_red_idx - 3 : best_red_idx + 4]
    blue_seq = None if best_blue_idx is None else results_plot.iloc[best_blue_idx - 3 : best_blue_idx + 4]

    # Event dates
    red_event_date = results_plot.iloc[best_red_idx]["date"].strftime("%Y-%m-%d")
    blue_event_date = None if best_blue_idx is None else results_plot.iloc[best_blue_idx]["date"].strftime("%Y-%m-%d")

    #Table
    if print_tables:
        red_start = red_seq.iloc[0]["date"].strftime("%Y-%m-%d")
        red_end = red_seq.iloc[-1]["date"].strftime("%Y-%m-%d")

        print("=" * 80)
        print(f"RED: Crash Transition ({red_start} to {red_end})")
        print(f"     Event date: {red_event_date}")
        print("=" * 80)
        print(red_seq[["date", "phase_regime", "abs_H_minus_HM", "abs_H_minus_HGR"]])

        if blue_seq is None:
            print("\nNo BLUE (Type-1→Type-2) transition found.")
        else:
            blue_start = blue_seq.iloc[0]["date"].strftime("%Y-%m-%d")
            blue_end = blue_seq.iloc[-1]["date"].strftime("%Y-%m-%d")

            print(f"\n{'=' * 80}")
            print(f"BLUE: Type-1→Type-2 Transition ({blue_start} to {blue_end})")
            print(f"      Event date: {blue_event_date}")
            print("=" * 80)
            print(blue_seq[["date", "phase_regime", "abs_H_minus_HM", "abs_H_minus_HGR"]])


    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    MARKERS = DEFAULT_MARKERS

    for regime, (marker, color, size) in MARKERS.items():
        mask = results_plot["phase_regime"] == regime
        if mask.any():
            ax.scatter(
                results_plot.loc[mask, "abs_H_minus_HM"] + 1e-6,
                results_plot.loc[mask, "abs_H_minus_HGR"] + 1e-6,
                marker=marker,
                c=color,
                s=size,
                label=regime,
                alpha=0.7,
                edgecolors="white",
                linewidth=0.3,
                zorder=2,
            )

    ax.scatter(
        [woe_point[0]],
        [woe_point[1]],
        c="white",
        marker="*",
        s=250,
        edgecolors="black",
        linewidth=1.5,
        label="WOE",
        zorder=5,
    )

    # RED sequence
    x_red = red_seq["abs_H_minus_HM"].values + 1e-6
    y_red = red_seq["abs_H_minus_HGR"].values + 1e-6
    ax.plot(x_red, y_red, "r--", linewidth=2.5, zorder=4)
    ax.scatter(x_red, y_red, c="darkred", s=80, zorder=5, edgecolors="white", linewidth=1)
    ax.scatter(x_red[3], y_red[3], c="black", s=150, zorder=6, edgecolors="darkred", linewidth=2)
    for i, (xi, yi) in enumerate(zip(x_red, y_red)):
        ax.annotate(
            str(i + 1),
            (xi, yi),
            fontsize=9,
            fontweight="bold",
            color="darkred",
            xytext=(5, 5),
            textcoords="offset points",
            zorder=7,
        )

    # BLUE sequence
    if blue_seq is not None:
        x_blue = blue_seq["abs_H_minus_HM"].values + 1e-6
        y_blue = blue_seq["abs_H_minus_HGR"].values + 1e-6
        ax.plot(x_blue, y_blue, "b--", linewidth=2.5, zorder=4)
        ax.scatter(x_blue, y_blue, c="darkblue", s=80, zorder=5, edgecolors="white", linewidth=1)
        ax.scatter(x_blue[3], y_blue[3], c="black", s=150, zorder=6, edgecolors="darkblue", linewidth=2)
        for i, (xi, yi) in enumerate(zip(x_blue, y_blue)):
            ax.annotate(
                str(i + 1),
                (xi, yi),
                fontsize=9,
                fontweight="bold",
                color="darkblue",
                xytext=(5, 5),
                textcoords="offset points",
                zorder=7,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("|H - H$_M$|", fontsize=14)
    ax.set_ylabel("|H - H$_{GR}$|", fontsize=14)
    ax.set_title(title, fontsize=12)
    ax.legend(loc="lower left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(list(xlim))
    ax.set_ylim(list(ylim))

    plt.tight_layout()
    plt.show()

    return {
        "results_plot": results_plot,
        "best_red_idx": int(best_red_idx),
        "best_blue_idx": None if best_blue_idx is None else int(best_blue_idx),
        "best_blue_score": float(best_blue_score),
        "red_seq": red_seq,
        "blue_seq": blue_seq,
        "red_event_date": red_event_date,
        "blue_event_date": blue_event_date,
    }


def report_crash_regime_detections(
    results,
    crash_events,
    *,
    window_radius=2,
    print_crash_periods=True,
):
    """
    Phase-space crash detection report.

    Parameters
    ----------
    results : pd.DataFrame
        Must contain: 'date', 'abs_H_minus_HM', 'abs_H_minus_HGR'
    crash_events : list[tuple[str, str]]
        List of (event_date, event_name), e.g.:
          ("1987-10-19", "Black Monday")
    window_radius : int
        Number of epochs before/after nearest epoch to inspect (default 2)
    print_crash_periods : bool
        Whether to print the full table of all detected crash epochs.

    Returns
    -------
    dict with:
      - results_plot
      - crash_periods
      - detection_summary
    """


    # Reuses existing phase-regime assignment
    results_plot, _ = assign_phase_regimes_quantiles(results)
    results_plot = results_plot.reset_index(drop=True)

    crash_periods = results_plot.loc[
        results_plot["phase_regime"] == "Crash",
        ["date", "abs_H_minus_HM", "abs_H_minus_HGR"],
    ]

    print("=" * 80)
    print("PHASE SPACE 'CRASH' REGIME DETECTIONS")
    print("(LOW |H-H_M| AND HIGH |H-H_GR| simultaneously)")
    print("=" * 80)
    print(f"\nTotal epochs classified as 'Crash': {len(crash_periods)} out of {len(results_plot)}")

    print("\nCrash periods detected:")
    if print_crash_periods:
        print(crash_periods.to_string())
    else:
        print("(printing disabled)")

    detection_summary = []

    # Checks each known crash event
    for event_date, event_name in crash_events:
        event_ts = pd.Timestamp(event_date)

        date_diffs = (results_plot["date"] - event_ts).abs()
        event_idx = date_diffs.idxmin()

        start_idx = max(0, event_idx - window_radius)
        end_idx = min(len(results_plot) - 1, event_idx + window_radius)

        window = results_plot.iloc[start_idx : end_idx + 1]
        regimes = window["phase_regime"].tolist()
        detected = "Crash" in regimes

        status = "✓ DETECTED" if detected else "✗ NOT DETECTED"
        print(
            f"{event_name:18s} ({event_date}): {status}  "
            f"[Regimes: {' → '.join(regimes)}]"
        )

        detection_summary.append(
            {
                "event_date": event_date,
                "event_name": event_name,
                "detected": bool(detected),
                "regime_sequence": " → ".join(regimes),
                "nearest_epoch_date": str(results_plot.iloc[event_idx]["date"].date()),
            }
        )

    return {
        "results_plot": results_plot,
        "crash_periods": crash_periods,
        "detection_summary": detection_summary,

        "well_detected_crashes": {
        d["event_date"]: d["event_name"]
        for d in detection_summary
        if d["detected"]
    }
}


def plot_entropy_crash_detection_panels(
    results_plot,
    thresholds,
    well_detected_crashes,
    *,
    days_threshold=120,
    title="Phase Transition Method: Entropy-Based Crash Detection",):

    """
    3-panel crash-detection

    Parameters
    ----------
    results_plot : pd.DataFrame
        Must already contain:
          - 'date'
          - 'abs_H_minus_HM'
          - 'abs_H_minus_HGR'
          - 'phase_regime'
    thresholds : object
        Must provide:
          - thresholds.H_HM_25
          - thresholds.H_HGR_50
        (This is the PhaseThresholds returned by assign_phase_regimes_quantiles.)
    well_detected_crashes : dict[str, str]
        Mapping: 'YYYY-MM-DD' -> event_name
        (Derived from report_crash_regime_detections output.)
    days_threshold : int
        Used to mark epochs "near" a well-detected crash (default 120)
    """


    # is_near_event 
    def is_near_event(epoch_date, events_dict, days_threshold=120):
        for event_date in events_dict.keys():
            event_ts = pd.Timestamp(event_date)
            if abs((epoch_date - event_ts).days) < days_threshold:
                return True
        return False

    #near_known_crash column
    results_plot = results_plot.copy()
    results_plot["near_known_crash"] = results_plot["date"].apply(
        lambda d: is_near_event(d, well_detected_crashes, days_threshold=days_threshold)
    )

    fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    dates = results_plot["date"]

    # Panel 1 - The TWO entropy indicators
    ax1 = axes[0]
    ax1.plot(dates, results_plot["abs_H_minus_HM"], "r-", linewidth=1.2, label="|H - H$_M$|")
    ax1.axhline(thresholds.H_HM_25, color="darkred", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_ylabel("|H - H$_M$|", fontsize=11, color="red")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.invert_yaxis()

    ax1_twin = ax1.twinx()
    ax1_twin.plot(dates, results_plot["abs_H_minus_HGR"], "purple", linewidth=1.2, label="|H - H$_{GR}$|")
    ax1_twin.axhline(thresholds.H_HGR_50, color="darkmagenta", linestyle="--", linewidth=1, alpha=0.7)
    ax1_twin.set_ylabel("|H - H$_{GR}$|", fontsize=11, color="purple")
    ax1_twin.tick_params(axis="y", labelcolor="purple")

    ax1.set_title(
        "The Two Entropy Indicators\n(Crash = LOW |H-H$_M$| AND HIGH |H-H$_{GR}$| simultaneously)",
        fontsize=12,
        fontweight="bold",
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")


    # Panel 2 - Phase Space
    ax2 = axes[1]
    regime_colors = {
        "Crash": "red",
        "Type-1": "deepskyblue",
        "Type-2": "blue",
        "Anomaly": "green",
        "Normal": "lightgray",
    }

    for i in range(len(results_plot) - 1):
        regime = results_plot.iloc[i]["phase_regime"]
        color = regime_colors[regime]
        ax2.axvspan(dates.iloc[i], dates.iloc[i + 1], alpha=0.7, color=color, linewidth=0)

    ax2.set_ylabel("Regime", fontsize=11)
    ax2.set_yticks([])
    ax2.set_title("Phase Space Regime Classification", fontsize=12, fontweight="bold")

    patches = [mpatches.Patch(color=color, label=regime) for regime, color in regime_colors.items()]
    ax2.legend(handles=patches, loc="upper right", ncol=5)


    # Panel 3
    ax3 = axes[2]
    ax3.set_facecolor("white")

    for i in range(len(results_plot) - 1):
        regime = results_plot.iloc[i]["phase_regime"]
        near_known = results_plot.iloc[i]["near_known_crash"]

        if regime == "Crash" and near_known:
            ax3.axvspan(dates.iloc[i], dates.iloc[i + 1], alpha=0.5, color="red", linewidth=0)

    for event_date, event_name in well_detected_crashes.items():
        event_ts = pd.Timestamp(event_date)
        if dates.min() <= event_ts <= dates.max():
            ax3.axvline(event_ts, color="black", linestyle="-", linewidth=2.5, alpha=0.9)
            ax3.annotate(
                event_name,
                xy=(event_ts, 0.5),
                xytext=(0, 0),
                textcoords="offset points",
                fontsize=10,
                rotation=90,
                ha="center",
                va="center",
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="black",
                    alpha=0.9,
                ),
            )

    ax3.set_yticks([])
    ax3.set_ylim(0, 1)
    ax3.set_title("Major Crashes Successfully Detected by Entropy Method", fontsize=12, fontweight="bold")

    axes[-1].set_xlabel("Date", fontsize=12)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(5))

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()

    return 0

