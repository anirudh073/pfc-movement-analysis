"""Spike visualisation: rasters, ISI histograms, autocorrelograms, etc."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math


def _select_units(spike_counts, units):
    """Return (selected_spike_counts, selected_unit_ids) for a unit subset."""
    all_ids = np.arange(spike_counts.shape[0])
    if units is None:
        return spike_counts, all_ids
    units = np.asarray(units)
    return spike_counts[units], units


def _grid_shape(n):
    """Return (nrows, ncols) for a roughly square grid of n panels."""
    ncols = math.ceil(math.sqrt(n))
    nrows = math.ceil(n / ncols)
    return nrows, ncols


#  1. Raster plot 

def plot_raster(spike_counts, cov_df,
                epoch=None, units=None, plot_position=False,
                figsize=None):
    """Raster plot for a set of units. Use %matplotlib qt for zoom/pan.

    Parameters
    ----------
    spike_counts  : 2-D array (n_units x n_bins)
    cov_df        : DataFrame with 'time_bin_center' column (and 'epoch')
    epoch         : if given, restrict to bins where cov_df["epoch"] == epoch
    units         : list of unit indices to plot (default: all)
    plot_position : if True, add a panel below showing linear position
                    coloured by speed. Requires 'linear_position' and
                    'speed' columns in cov_df.
    figsize       : figure size tuple
    """
    times = cov_df["time_bin_center"].values.copy()
    spk = np.asarray(spike_counts)

    if epoch is not None:
        mask = cov_df["epoch"].values == epoch
        times = times[mask]
        spk = spk[:, mask]
        cov_sub = cov_df.iloc[np.where(mask)[0]]
    else:
        cov_sub = cov_df

    # show time relative to start of the displayed window
    t0 = times[0]
    times = times - t0

    spk, uids = _select_units(spk, units)

    spike_times = [times[spk[i] > 0] for i in range(len(uids))]

    if figsize is None:
        h = max(3, 0.4 * len(uids)) + (2.5 if plot_position else 0)
        figsize = (14, h)

    if plot_position:
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = fig.add_gridspec(
            2, 2,
            height_ratios=[max(3, 0.4 * len(uids)), 2.5],
            width_ratios=[40, 1]
        )
        ax_raster = fig.add_subplot(gs[0, 0])
        ax_pos = fig.add_subplot(gs[1, 0], sharex=ax_raster)
        cax = fig.add_subplot(gs[:, 1])
    else:
        fig, ax_raster = plt.subplots(figsize=figsize, constrained_layout=True)

    ax_raster.eventplot(spike_times, lineoffsets=np.arange(len(uids)),
                        linelengths=0.8, linewidths=0.4, color="k")
    ax_raster.set_yticks(np.arange(len(uids)))
    ax_raster.set_yticklabels(uids, fontsize=7)
    ax_raster.set_ylabel("Unit")
    title = "Raster"
    if epoch is not None:
        title += f" — epoch {epoch}"
    ax_raster.set_title(title)
    sns.despine(ax=ax_raster)

    if plot_position:
        pos = cov_sub["linear_position"].values
        spd = cov_sub["speed"].values
        pos_times = cov_sub["time_bin_center"].values - t0
        sc = ax_pos.scatter(pos_times, pos, c=spd, s=0.3, cmap="viridis",
                            rasterized=True)
        ax_pos.set_ylabel("Linear position")
        ax_pos.set_xlabel("Time (s)")
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label("Speed (cm/s)", fontsize=8)
        sns.despine(ax=ax_pos)
    else:
        ax_raster.set_xlabel("Time (s)")

    if plot_position:
        return fig, (ax_raster, ax_pos)
    return fig, ax_raster


#  2. ISI distribution grid 

def plot_isi_grid(spike_counts, bin_size=0.002, cov_df=None, epoch=None,
                  units=None, max_isi_ms=100, n_hist_bins=100, figsize=None):
    """Plot ISI distributions for a set of units in an auto-sized grid.

    Parameters
    ----------
    spike_counts  : 2-D array (n_units x n_bins)
    bin_size      : bin width in seconds (default 0.002)
    cov_df        : DataFrame with 'epoch' column (needed if epoch is set)
    epoch         : if given, restrict to bins where cov_df["epoch"] == epoch
    units         : list of unit indices to plot (default: all)
    max_isi_ms    : upper ISI limit in ms for the histogram (default 100)
    n_hist_bins   : number of histogram bins (default 100)
    figsize       : figure size tuple
    """
    spk = np.asarray(spike_counts)
    if epoch is not None:
        mask = cov_df["epoch"].values == epoch
        spk = spk[:, mask]
    spk, uids = _select_units(spk, units)
    n = len(uids)
    nrows, ncols = _grid_shape(n)

    if figsize is None:
        figsize = (3.5 * ncols, 2.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for idx in range(n):
        ax = axes.flat[idx]
        spike_bins = np.where(spk[idx] > 0)[0]
        if len(spike_bins) < 2:
            ax.text(0.5, 0.5, "< 2 spikes", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_title(f"Unit {uids[idx]}", fontsize=8)
            continue

        isis_ms = np.diff(spike_bins) * bin_size * 1000
        isis_ms = isis_ms[isis_ms <= max_isi_ms]

        ax.hist(isis_ms, bins=n_hist_bins, color="steelblue", edgecolor="none")
        ax.axvline(bin_size * 1000, color="red", lw=0.8, ls="--",
                   label=f"1 bin ({bin_size*1000:.0f} ms)")
        ax.set_title(f"Unit {uids[idx]}", fontsize=8)
        ax.set_xlabel("ISI (ms)", fontsize=7)
        ax.set_ylabel("Count", fontsize=7)
        ax.tick_params(labelsize=6)
        sns.despine(ax=ax)

    for idx in range(n, nrows * ncols):
        axes.flat[idx].set_visible(False)

    title = f"Inter-spike interval distributions (bin = {bin_size*1000:.0f} ms)"
    if epoch is not None:
        title += f" — epoch {epoch}"
    fig.suptitle(title, fontsize=11)
    return fig, axes


#  3. Autocorrelogram grid 

def plot_acg_grid(spike_counts, bin_size=0.002, cov_df=None, epoch=None,
                  units=None, max_lag_ms=50, plot_bin_ms=None, figsize=None):
    """Plot autocorrelograms for a set of units in an auto-sized grid.

    Parameters
    ----------
    spike_counts : 2-D array (n_units x n_bins)
    bin_size     : spike-count bin width in seconds (default 0.002)
    cov_df       : DataFrame with 'epoch' column (needed if epoch is set)
    epoch        : if given, restrict to bins where cov_df["epoch"] == epoch
    units        : list of unit indices to plot (default: all)
    max_lag_ms   : maximum lag in ms (default 50)
    plot_bin_ms  : display bin width in ms (default: same as bin_size).
                   Must be a multiple of bin_size. Bins are summed.
    figsize      : figure size tuple
    """
    spk = np.asarray(spike_counts)
    if epoch is not None:
        mask = cov_df["epoch"].values == epoch
        spk = spk[:, mask]
    spk, uids = _select_units(spk, units)
    n = len(uids)
    nrows, ncols = _grid_shape(n)

    data_bin_ms = bin_size * 1000
    if plot_bin_ms is None:
        plot_bin_ms = data_bin_ms
    rebin = max(1, round(plot_bin_ms / data_bin_ms))
    plot_bin_ms = rebin * data_bin_ms

    max_lag_bins = int(max_lag_ms / data_bin_ms)

    if figsize is None:
        figsize = (3.5 * ncols, 2.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for idx in range(n):
        ax = axes.flat[idx]
        counts = spk[idx].astype(float)
        n_spikes = counts.sum()
        if n_spikes < 2:
            ax.text(0.5, 0.5, "< 2 spikes", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8)
            ax.set_title(f"Unit {uids[idx]}", fontsize=8)
            continue

        acg = np.correlate(counts, counts, mode="full")
        mid = len(acg) // 2
        acg_sym = acg[mid - max_lag_bins: mid + max_lag_bins + 1].copy()
        acg_sym[max_lag_bins] = 0  # zero lag-0

        # rebin if requested
        if rebin > 1:
            n_neg = max_lag_bins
            neg = acg_sym[:n_neg]
            pos = acg_sym[n_neg + 1:]
            # trim to multiple of rebin
            n_trim = (len(neg) // rebin) * rebin
            neg = neg[-n_trim:].reshape(-1, rebin).sum(axis=1)
            pos = pos[:n_trim].reshape(-1, rebin).sum(axis=1)
            acg_sym = np.concatenate([neg, [0], pos])

        n_half = len(acg_sym) // 2
        lags_ms = np.arange(-n_half, n_half + 1) * plot_bin_ms

        ax.bar(lags_ms, acg_sym,
               width=plot_bin_ms * 0.9, color="steelblue", edgecolor="none")
        ax.set_title(f"Unit {uids[idx]}", fontsize=8)
        ax.set_xlabel("Lag (ms)", fontsize=7)
        ax.set_ylabel("Coincidences", fontsize=7)
        ax.tick_params(labelsize=6)
        sns.despine(ax=ax)

    for idx in range(n, nrows * ncols):
        axes.flat[idx].set_visible(False)

    title = f"Autocorrelograms (±{max_lag_ms} ms, {plot_bin_ms:.0f} ms bins)"
    if epoch is not None:
        title += f" — epoch {epoch}"
    fig.suptitle(title, fontsize=11)
    return fig, axes
