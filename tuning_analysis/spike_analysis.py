import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import spyglass.spikesorting.v1 as sgs
import spikeinterface as si
import spikeinterface.widgets as sw
import spikeinterface.core as sc
from spikeinterface.postprocessing import compute_correlograms
import spyglass.position.v1 as sgp
import spyglass.linearization.v1 as sgpl
from spyglass.position import PositionOutput
from spyglass.spikesorting.analysis.v1.group import UnitSelectionParams
from spyglass.spikesorting.analysis.v1.group import SortedSpikesGroup
from elephant.statistics import time_histogram
from elephant.conversion import BinnedSpikeTrain
from spyglass.spikesorting.spikesorting_merge import SpikeSortingOutput


MIN_OCCUPANCY = 1 #second(s)


def fetch_single_epoch_spikes(nwb_file_name: str,
                              sorted_spikes_group_nane: str):
    group_key = {
        "nwb_file_name": nwb_file_name,
        "sorted_spikes_group_name": sorted_spikes_group_nane
    }
    
    SortedSpikesGroup.Units & group_key
    
    group_key = (SortedSpikesGroup & group_key).fetch1("KEY")
    return SortedSpikesGroup().fetch_spike_data(group_key)

def get_unit_firing_rate(spikes: list, unit: int):
    return len(spikes[unit])/((spikes[unit][-1]) - (spikes[unit][0]))


# ----------------------------- METRIC DEFINITIONS
#
# NOTE: These metrics are used in multiple contexts:
# - simple pairwise comparisons (called with just curve_a, curve_b)
# - generic permutation tests (called as spec.func(curve_a, curve_b, centers, min_bins))
#
# To avoid fragile "dual definitions" and signature mismatches (especially under
# %autoreload), we define a single set of functions with a signature that works
# in both call styles. Unused args are accepted but ignored where appropriate.

def curve_corr(curve_a, curve_b, centers=None, min_bins=3):
    curve_a = np.asarray(curve_a, dtype=float)
    curve_b = np.asarray(curve_b, dtype=float)
    finite = np.isfinite(curve_a) & np.isfinite(curve_b)
    if int(finite.sum()) < int(min_bins):
        return float("nan")
    return float(np.corrcoef(curve_a[finite], curve_b[finite])[0, 1])


def mod_depth(curve):
    curve = np.asarray(curve, dtype=float)
    if np.all(~np.isfinite(curve)):
        return float("nan")
    return float(np.nanmax(curve) - np.nanmin(curve))


def nrmse(curve_a, curve_b, centers=None, min_bins=3, eps=1e-6):
    curve_a = np.asarray(curve_a, dtype=float)
    curve_b = np.asarray(curve_b, dtype=float)
    finite = np.isfinite(curve_a) & np.isfinite(curve_b)
    if int(finite.sum()) < int(min_bins):
        return float("nan")
    rmse = float(np.sqrt(np.mean((curve_a[finite] - curve_b[finite]) ** 2)))
    denom = max(mod_depth(curve_a), mod_depth(curve_b)) + float(eps)
    return float(rmse / denom)


def peak_shift_x(curve_a, curve_b, centers=None, min_bins=3):
    curve_a = np.asarray(curve_a, dtype=float)
    curve_b = np.asarray(curve_b, dtype=float)
    x = np.asarray(centers, dtype=float) if centers is not None else None
    if x is None:
        return float("nan")
    finite = np.isfinite(curve_a) & np.isfinite(curve_b) & np.isfinite(x)
    if int(finite.sum()) < int(min_bins):
        return float("nan")
    idxs = np.where(finite)[0]
    ia = idxs[np.argmax(curve_a[finite])]
    ib = idxs[np.argmax(curve_b[finite])]
    return float(abs(x[ia] - x[ib]))


def peak_bin_shift(curve_a, curve_b, centers=None, min_bins=3):
    curve_a = np.asarray(curve_a, dtype=float)
    curve_b = np.asarray(curve_b, dtype=float)
    finite = np.isfinite(curve_a) & np.isfinite(curve_b)
    if int(finite.sum()) < int(min_bins):
        return float("nan")
    idx_a = np.where(finite)[0][np.argmax(curve_a[finite])]
    idx_b = np.where(finite)[0][np.argmax(curve_b[finite])]
    return float(abs(idx_a - idx_b))


def mean_rate(curve):
    curve = np.asarray(curve, dtype=float)
    return float(np.nanmean(curve)) if np.any(np.isfinite(curve)) else float("nan")


def mean_rate_diff(curve_a, curve_b, centers=None, min_bins=3):
    curve_a = np.asarray(curve_a, dtype=float)
    curve_b = np.asarray(curve_b, dtype=float)
    finite = np.isfinite(curve_a) & np.isfinite(curve_b)
    if int(finite.sum()) < int(min_bins):
        return float("nan")
    ma = float(np.nanmean(curve_a[finite]))
    mb = float(np.nanmean(curve_b[finite]))
    return float(abs(ma - mb))


#------------------------------------------------




def spikes_to_speed(spike_times, timestamps, speed, mask=None):
    """
    Map spike times to speed values at the closest earlier timestamp.
    If mask is provided (boolean array same length as timestamps),
    only keep spikes whose corresponding timestamp has mask == True.
    """
    timestamps = np.asarray(timestamps)
    speed = np.asarray(speed)
    spike_times = np.asarray(spike_times)

    t0, t1 = timestamps[0], timestamps[-1]
    in_window = (spike_times >= t0) & (spike_times <= t1)
    spike_times_win = spike_times[in_window]

    spike_idx = np.searchsorted(timestamps, spike_times_win, side='right') - 1
    spike_idx = np.clip(spike_idx, 0, len(timestamps) - 1)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        keep = mask[spike_idx]
        spike_idx = spike_idx[keep]

    return speed[spike_idx]



def rest_vs_run_firing_rates(spikes_list: list,
                             trialwise_position_df: pd.DataFrame,
                             speed_threshold: int = 5,
                             segment_threshold: int = 3,
                             ):
    """
    Compute per-unit firing rates during rest vs run segments and
    perform a Mann-Whitney U test across segments for each unit.

    Returns a dataframe with columns:
        'unit', 'rest firing rate', 'rest sem',
        'run firing rate', 'run sem',
        'mw_statistic', 'mw_p_value'
    """
    speed_boolean_df = trialwise_position_df.copy()
    speed_boolean_df["is_running"] = speed_boolean_df["speed"] > speed_threshold

    segment_changes = speed_boolean_df["is_running"].ne(
        speed_boolean_df["is_running"].shift(
            fill_value=speed_boolean_df["is_running"].iloc[0]
        )
    )
    speed_boolean_df["speed_segment_id"] = segment_changes.cumsum()
    speed_boolean_df["time"] = speed_boolean_df.index

    segments = (
        speed_boolean_df.groupby(["speed_segment_id", "is_running"])
        .agg(start_time=("time", "first"),
             end_time=("time", "last"),
             n_samples=("time", "size"))
        .reset_index()
    )

    segments["duration"] = segments["end_time"] - segments["start_time"]

    run_segments = segments[
        (segments["is_running"]) & (segments["duration"] >= segment_threshold)
    ]
    rest_segments = segments[
        (~segments["is_running"]) & (segments["duration"] >= segment_threshold)
    ]

    run_intervals = list(zip(run_segments["start_time"], run_segments["end_time"]))
    rest_intervals = list(zip(rest_segments["start_time"], rest_segments["end_time"]))

    unit_ids = []
    mean_rest_fr_list = []
    mean_run_fr_list = []
    mean_rest_sem_list = []
    mean_run_sem_list = []
    mw_stat_list = []
    mw_pval_list = []

    for unit in range(len(spikes_list)):
        rest_fr_list = []
        run_fr_list = []

        for run_interval, rest_interval in zip(run_intervals, rest_intervals):
            rest_timestamps = trialwise_position_df[
                (trialwise_position_df.index > rest_interval[0])
                & (trialwise_position_df.index < rest_interval[-1])
            ].index.to_list()
            run_timestamps = trialwise_position_df[
                (trialwise_position_df.index > run_interval[0])
                & (trialwise_position_df.index < run_interval[-1])
            ].index.to_list()

            if len(rest_timestamps) < 2 or len(run_timestamps) < 2:
                # skip pathological segments with too few samples
                continue

            spikes_mask_rest = (
                (spikes_list[unit] >= rest_timestamps[0])
                & (spikes_list[unit] <= rest_timestamps[-1])
            )
            spikes_mask_run = (
                (spikes_list[unit] >= run_timestamps[0])
                & (spikes_list[unit] <= run_timestamps[-1])
            )

            spikes_rest = spikes_list[unit][spikes_mask_rest]
            spikes_run = spikes_list[unit][spikes_mask_run]

            spike_idx_temp_rest = np.searchsorted(rest_timestamps, spikes_rest)
            spike_idx_temp_run = np.searchsorted(run_timestamps, spikes_run)

            # average firing rate in each interval
            rest_firing_rate = len(spike_idx_temp_rest) / (
                rest_timestamps[-1] - rest_timestamps[0]
            )
            run_firing_rate = len(spike_idx_temp_run) / (
                run_timestamps[-1] - run_timestamps[0]
            )

            rest_fr_list.append(rest_firing_rate)
            run_fr_list.append(run_firing_rate)

        # if no valid segments for this unit, fill with NaNs
        if len(rest_fr_list) == 0 or len(run_fr_list) == 0:
            mean_rest_firing_rate = np.nan
            mean_run_firing_rate = np.nan
            sem_rest_firing_rate = np.nan
            sem_run_firing_rate = np.nan
            mw_stat = np.nan
            mw_p = np.nan
        else:
            # average over intervals
            mean_rest_firing_rate = float(np.mean(rest_fr_list))
            mean_run_firing_rate = float(np.mean(run_fr_list))

            # SEM across segments
            sem_rest_firing_rate = stats.sem(rest_fr_list) if len(rest_fr_list) > 1 else 0.0
            sem_run_firing_rate = stats.sem(run_fr_list) if len(run_fr_list) > 1 else 0.0

            # Mann-Whitney U test across segments (rest vs run)
            try:
                mw_stat, mw_p = stats.mannwhitneyu(
                    rest_fr_list,
                    run_fr_list,
                    alternative="two-sided",
                )
            except ValueError:
                # e.g. all values identical
                mw_stat, mw_p = np.nan, np.nan

        # append
        unit_ids.append(unit + 1)  # offset index to start at 1
        mean_run_fr_list.append(mean_run_firing_rate)
        mean_rest_fr_list.append(mean_rest_firing_rate)
        mean_rest_sem_list.append(sem_rest_firing_rate)
        mean_run_sem_list.append(sem_run_firing_rate)
        mw_stat_list.append(mw_stat)
        mw_pval_list.append(mw_p)

    mean_fr_df = pd.DataFrame(
        {
            "unit": unit_ids,
            "rest firing rate": mean_rest_fr_list,
            "rest sem": mean_rest_sem_list,
            "run firing rate": mean_run_fr_list,
            "run sem": mean_run_sem_list,
            "mw_statistic": mw_stat_list,
            "mw_p_value": mw_pval_list,
        }
    )

    return mean_fr_df





def plot_rest_vs_run_fr(mean_fr_df: pd.DataFrame, alpha: float = 0.05, linewidth: float = 1.5):
    """
    Plot different visualizations of rest vs run firing rates.

    If 'mw_p_value' is present in mean_fr_df, units with p < alpha
    are marked with an asterisk in the per-unit bar plot (type "0").
    """
    plot_types = {
        0: "per-unit paired barplot",
        1: "per-unit connected",
        2: "scatter",
        3: "population_boxplot",
    }

    plot_type = input(f"Enter the index of the plot you want to create: {plot_types}")

    if plot_type == "0":
        plt.figure(figsize=(24, 12), layout="tight")

        unit_ids = mean_fr_df["unit"].values if "unit" in mean_fr_df.columns else np.arange(len(mean_fr_df))
        x = np.arange(len(mean_fr_df))
        width = 0.35

        # bar for rest
        plt.bar(
            x - width / 2,
            mean_fr_df["rest firing rate"],
            width,
            yerr=mean_fr_df["rest sem"],
            label="Rest",
            capsize=3,
            alpha=0.8,
        )

        # bar for run
        plt.bar(
            x + width / 2,
            mean_fr_df["run firing rate"],
            width,
            yerr=mean_fr_df["run sem"],
            label="Run",
            capsize=3,
            alpha=0.8,
        )

        # mark significant units with an asterisk if p-values are available
        if "mw_p_value" in mean_fr_df.columns:
            for i, row in mean_fr_df.iterrows():
                p = row["mw_p_value"]
                if not np.isfinite(p) or p >= alpha:
                    continue

                fr_rest = row["rest firing rate"]
                fr_run = row["run firing rate"]
                sem_rest = row.get("rest sem", 0.0)
                sem_run = row.get("run sem", 0.0)

                # height of the taller bar (mean + sem)
                y_max = max(fr_rest + sem_rest, fr_run + sem_run)
                if not np.isfinite(y_max) or y_max <= 0:
                    y_star = 0.05
                else:
                    y_star = y_max * 1.05

                plt.text(
                    x[i],
                    y_star,
                    "*",
                    ha="center",
                    va="bottom",
                    fontsize=20,
                    color = "red"
                )

        plt.xticks(x, unit_ids, rotation=90)
        plt.ylabel("Firing rate (Hz)")
        plt.xlabel("unit")
        # plt.title("Rest vs run firing rates per unit")
        plt.legend()
        plt.tight_layout()
        plt.show()

    elif plot_type == "1":
        plt.figure(figsize=(24, 12), layout="tight")
        for _, row in mean_fr_df.iterrows():
            plt.plot(
                [0, 1],
                [row["rest firing rate"], row["run firing rate"]],
                "-o",
                color="gray",
                alpha=0.5,
                linewidth=linewidth,
            )

        plt.xticks([0, 1], ["Rest", "Run"])
        plt.ylabel("Firing rate (Hz)")
        plt.title("Per-unit change in firing rate")
        plt.show()

    elif plot_type == "2":
        plt.figure(figsize=(24, 12), layout="tight")

        plt.scatter(
            mean_fr_df["rest firing rate"],
            mean_fr_df["run firing rate"],
            alpha=0.7,
        )

        max_val = max(
            mean_fr_df["rest firing rate"].max(),
            mean_fr_df["run firing rate"].max(),
        )
        plt.plot([0, max_val], [0, max_val], "k--", label="y = x", linewidth=linewidth)

        plt.xlabel("Mean FR during rest (Hz)")
        plt.ylabel("Mean FR during run (Hz)")
        plt.title("Rest vs run firing rates per unit")
        plt.legend()
        plt.axis("equal")
        plt.show()

    elif plot_type == "3":
        df_long = mean_fr_df.melt(
            id_vars="unit", 
            value_vars=["rest firing rate", "run firing rate"],
            var_name="state",
            value_name="mean_fr",
        )

        plt.figure(figsize=(12, 12))
        sns.boxplot(data=df_long, x="state", y="mean_fr")
        sns.stripplot(
            data=df_long,
            x="state",
            y="mean_fr",
            color="black",
            size=3,
            alpha=0.5,
        )

        plt.ylabel("Mean firing rate (Hz)")
        plt.title("Rest vs run firing rate distributions")
        plt.tight_layout()
        plt.show()

    else:
        print("Invalid plot type!")

        
     
     
        
    
def compute_speed_tuning(position_df: pd.DataFrame,
                         spikes_list: list,
                         n_bins: int = 8,
                         mask: np.ndarray = None):
    """
    position_df: full position_trials_merged_df (NOT pre-masked)
    mask: boolean array on position_df.index (e.g. zone=='run' & trial_type=='inbound')
    """

    timestamps = position_df.index.to_numpy()
    speed = position_df['speed'].to_numpy()


    if mask is None:
        mask = np.ones_like(speed, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != speed.shape[0]:
            raise ValueError("mask must have same length as position_df")


    speed_masked = speed[mask]

    dt = np.median(np.diff(timestamps))
    max_speed = np.nanmax(speed_masked)
    speed_bins = np.linspace(0, max_speed, n_bins + 1)
    speed_bin_centers = (speed_bins[:-1] + speed_bins[1:]) / 2

    occupancy_counts, _ = np.histogram(speed_masked, bins=speed_bins)
    occupancy_time = occupancy_counts * dt

    speed_tuning = {}

    for unit, spike_times in enumerate(spikes_list):
        speed_at_spikes = spikes_to_speed(spike_times, timestamps, speed, mask=mask)
        spike_counts, _ = np.histogram(speed_at_spikes, bins=speed_bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            firing_rate = spike_counts / occupancy_time
            firing_rate[occupancy_time == 0] = np.nan

        speed_tuning[unit] = firing_rate

    return speed_tuning, speed_bin_centers

    



def plot_speed_tuning_heatmap(speed_tuning: dict, 
                              vmin: int = 0,
                              vmax: int = 10,
                              cmap: str = "Blues",
                              figsize = (16,16)):
    units = list(speed_tuning.keys())
    fr_matrix = np.vstack([speed_tuning[u] for u in units]) 
    
    fig, ax = plt.subplots(figsize = figsize, layout = "tight")
    sns.heatmap(
        fr_matrix,
        vmin= vmin,
        vmax= vmax,
        cmap="Blues",
        ax=ax,
        cbar_kws={"label": "Firing Rate (Hz)"},
    )
    ax.set_xlabel("Speed (cm/s)")
    ax.set_ylabel("unit#")
    ax.set_yticks(np.arange(len(units)) + 0.5, units, fontsize = 6)
    plt.show()




def plot_speed_tuning_grid(
    speed_tuning: dict,
    position_df: pd.DataFrame,
    spikes_list: list,
    mask: np.ndarray = None,
    n_units: int = -1,
    label=None,
    linewidth: float = 1.0,
):
    """
    Plot per-unit speed tuning curves with 95% CI.

    position_df: full position_trials_merged_df (NOT pre-masked)
    mask: boolean array aligned with position_df.index (e.g. zone=='run' & trial_type=='inbound')
    """

    # --- prepare timebase + mask on the FULL df ---
    timestamps = position_df.index.to_numpy()
    speed = position_df['speed'].to_numpy()

    if mask is None:
        mask = np.ones_like(speed, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != speed.shape[0]:
            raise ValueError("mask must have same length as position_df")

    # infer number of bins from the tuning dict
    first_unit = next(iter(speed_tuning))
    n_bins = len(speed_tuning[first_unit])

    # occupancy only where mask is True
    speed_masked = speed[mask]
    dt = np.median(np.diff(timestamps))

    max_speed = np.nanmax(speed_masked)
    speed_bins = np.linspace(0, max_speed, n_bins + 1)
    speed_bin_centers = (speed_bins[:-1] + speed_bins[1:]) / 2

    occupancy_counts, _ = np.histogram(speed_masked, bins=speed_bins)
    occupancy_time = occupancy_counts * dt  # seconds in each speed bin

    # --- helper: spike counts per speed bin, using the same mask ---
    def _spike_counts_per_speed_bin(spike_times):
        speed_at_spikes = spikes_to_speed(spike_times, timestamps, speed, mask=mask)
        spike_counts, _ = np.histogram(speed_at_spikes, bins=speed_bins)
        return spike_counts

    # --- choose units to plot ---
    all_units = list(speed_tuning.keys())
    if n_units < 0 or n_units > len(all_units):
        units = all_units
    else:
        units = all_units[:n_units]
    n_units = len(units)

    n_cols = 5
    n_rows = int(np.ceil(n_units / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2 * n_rows),
        sharex=True, sharey=True
    )
    axes = np.array(axes).reshape(-1)

    # --- per-unit curves + CIs ---
    for i, unit in enumerate(units):
        spike_times = spikes_list[unit]
        spike_counts = _spike_counts_per_speed_bin(spike_times)

        valid = occupancy_time > MIN_OCCUPANCY

        rate = np.full_like(occupancy_time, np.nan, dtype=float)
        se_rate = np.full_like(occupancy_time, np.nan, dtype=float)

        rate[valid] = spike_counts[valid] / occupancy_time[valid]
        se_rate[valid] = np.sqrt(spike_counts[valid]) / occupancy_time[valid]

        z = 1.96  # ~95% CI
        lower = np.clip(rate - z * se_rate, 0, None)
        upper = rate + z * se_rate

        ax = axes[i]
        ax.plot(speed_bin_centers, rate, marker="o", linewidth=linewidth)
        ax.fill_between(speed_bin_centers, lower, upper,
                        color='C0', alpha=0.3, label='95% CI')

        ax.set_title(str(unit), fontsize=8)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=6)

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'Speed tuning curves {label}', fontsize=14)
    fig.text(0.5, 0.04, 'Speed (cm/s)', ha='center')
    fig.text(0.04, 0.5, 'Firing rate (Hz)', va='center', rotation='vertical')
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.show()






# def compute_speed_tuning_CI(spikes_list: list,
#                             trialwise_position_df: pd.DataFrame,
#                             n_bins: int = 8):
    
    
#     def _spike_counts_per_speed_bin(spike_times, timestamps, speed, speed_bins):
#         speed_at_spikes = spikes_to_speed(spike_times, timestamps, speed)
#         spike_counts, _ = np.histogram(speed_at_spikes, bins=speed_bins)
#         return spike_counts    
        
    
#     timestamps = trialwise_position_df.index.to_list()
#     dt = np.median(np.diff(timestamps))
    
#     speed = trialwise_position_df.speed.to_numpy()
#     max_speed = np.nanmax(speed)  
#     speed_bins = np.linspace(0, max_speed, n_bins + 1)
    
#     occupancy_counts, _ = np.histogram(speed, bins=speed_bins)
#     occupancy_time = occupancy_counts * dt
    
    
#     #for one unit
#     spike_times = spikes_list[unit]
#     spike_counts = _spike_counts_per_speed_bin(spike_times, timestamps, speed, speed_bins)

#     valid = occupancy_time > MIN_OCCUPANCY

#     rate = np.full_like(occupancy_time, np.nan, dtype=float)
#     se_rate = np.full_like(occupancy_time, np.nan, dtype=float)

#     rate[valid] = spike_counts[valid] / occupancy_time[valid]
#     se_rate[valid] = np.sqrt(spike_counts[valid]) / occupancy_time[valid]

#     z = 1.96  # for ~95% CI
#     lower = rate - z * se_rate
#     upper = rate + z * se_rate

#     lower = np.clip(lower, 0, None)
    
#     return rate, lower, upper




def get_si_recording_and_sorting(sorted_group_key: dict):
    spikes, units = SortedSpikesGroup.fetch_spike_data(
        key=sorted_group_key,
        return_unit_ids=True,
    )

    n_units = len(units)
    unit_ids = np.arange(n_units)

    unit_table = pd.DataFrame(units)
    unit_table["group_unit_id"] = unit_ids

    units_by_merge_group = (
        unit_table.groupby("spikesorting_merge_id")["group_unit_id"].apply(list).to_dict()
    )
    merge_ids = list(units_by_merge_group.keys())

    recordings = []
    sortings_filtered = []

    for mid in merge_ids:
        key = dict(merge_id=mid)
        rec = SpikeSortingOutput.get_recording(key)
        rec_times = rec.get_times()
        fs = rec.get_sampling_frequency()

        unit_dict = {}
        group_unit_ids_for_merge = units_by_merge_group[mid]

        for gu in group_unit_ids_for_merge:
            spike_times = np.asarray(spikes[gu])
            spike_times = spike_times[
                (spike_times >= rec_times[0]) & (spike_times <= rec_times[-1])
            ]
            spike_frames = np.searchsorted(rec_times, spike_times)
            unit_dict[gu] = spike_frames.astype("int64")

        sorting_sel = si.NumpySorting.from_unit_dict(
            [unit_dict],
            sampling_frequency=fs,
        )

        recordings.append(rec)
        sortings_filtered.append(sorting_sel)

    # SpikeInterface requires unique channel locations when aggregating channels.
    # Some Spyglass recordings carry duplicated (or placeholder) "location" properties
    # across merge groups, which makes sc.aggregate_channels raise:
    #   AssertionError: Locations are not unique! Cannot aggregate recordings!
    #
    # For tuning analyses we primarily need a consistent channel axis; when locations
    # exist but collide, we make them unique with a tiny deterministic jitter that
    # preserves within-recording geometry.
    try:
        has_any_location = any(
            ("location" in r.get_property_keys()) and (r.get_property("location") is not None)
            for r in recordings
        )
    except Exception:
        has_any_location = False

    if has_any_location:
        all_locations = []
        for r in recordings:
            if "location" not in r.get_property_keys():
                continue
            loc = r.get_property("location")
            if loc is None:
                continue
            loc = np.asarray(loc)
            if loc.ndim != 2 or loc.shape[0] != r.get_num_channels():
                continue
            all_locations.extend([tuple(x) for x in loc])

        locations_unique = (len(set(all_locations)) == len(all_locations)) if all_locations else True
        if not locations_unique:
            for rec_index, r in enumerate(recordings):
                if "location" not in r.get_property_keys():
                    continue
                loc = r.get_property("location")
                if loc is None:
                    continue
                loc = np.asarray(loc)
                if loc.ndim != 2 or loc.shape[0] != r.get_num_channels() or loc.shape[1] < 1:
                    continue

                loc = loc.astype("float64", copy=True)
                # Offset each merge-group recording slightly to avoid cross-recording collisions.
                loc[:, 0] += rec_index * 1e-3
                # Jitter per-channel in a second dimension when present, else the first.
                jitter_dim = 1 if loc.shape[1] > 1 else 0
                loc[:, jitter_dim] += np.arange(loc.shape[0], dtype="float64") * 1e-6
                r.set_property("location", loc)

    try:
        recording = sc.aggregate_channels(recordings)
    except AssertionError as exc:
        # As a last resort, drop locations entirely and retry aggregation.
        if "Locations are not unique" not in str(exc):
            raise
        for r in recordings:
            if "location" not in getattr(r, "get_property_keys", lambda: [])():
                continue
            delete_prop = getattr(r, "delete_property", None)
            if callable(delete_prop):
                try:
                    delete_prop("location")
                except Exception:
                    pass
        recording = sc.aggregate_channels(recordings)

    renamed_unit_ids = np.concatenate([s.get_unit_ids() for s in sortings_filtered])
    sorting = sc.aggregate_units(
        sortings_filtered,
        renamed_unit_ids=renamed_unit_ids,
    )
    sorting.register_recording(recording)
    recording.annotate(is_filtered=True)

    return recording, sorting



"---------------------------------"



# the following function was written by Codex GPT 5.1:

def plot_unit_speed_acg_template_grid(
    unit_ids: list,
    spikes_list: list,
    speed_tuning: dict,
    speed_bin_centers: np.ndarray,
    sorting: si.BaseSorting,
    waveform_extractor: sc.WaveformExtractor = None,
    position_df: pd.DataFrame = None,
    mask: np.ndarray = None,
    plot_speed: bool = True,
    plot_acg: bool = True,
    plot_template: bool = True,
    window_ms: float = 300.0,
    bin_ms: float = 2.0,
    figsize: tuple = None,
    label_prefix: str = "unit",
    suptitle: str = "",
    linewidth: float = 1.0,
):
    """
    Plot speed tuning (with 95% CI), autocorrelogram, and template for selected units.

    Args:
        unit_ids (list): group_unit_id values to plot (indices into spikes_list and sorting).
        spikes_list (list): list of spike time arrays (e.g. output of SortedSpikesGroup.fetch_spike_data).
        speed_tuning (dict): dict[unit_id] -> firing rate over speed bins (from compute_speed_tuning).
        speed_bin_centers (np.ndarray): centers of the speed bins (from compute_speed_tuning).
        sorting (si.BaseSorting): SpikeInterface sorting with unit_ids matching group_unit_id.
        waveform_extractor (WaveformExtractor, optional): WaveformExtractor for templates.
        position_df (pd.DataFrame, optional): full position_trials_merged_df used for speed tuning
                                              (must have 'speed' column and time index).
        mask (np.ndarray, optional): boolean mask aligned with position_df.index (same as used in compute_speed_tuning).
        plot_speed (bool): whether to plot speed tuning column.
        plot_acg (bool): whether to plot autocorrelogram column.
        plot_template (bool): whether to plot template column.
        window_ms (float): correlogram window in ms.
        bin_ms (float): correlogram bin size in ms.
        figsize (tuple): (width, height) of the figure. If None, computed from n_units and n_cols.
        label_prefix (str): text prefix for unit label (e.g. "unit" or "cell").
    """
    if not any([plot_speed, plot_acg, plot_template]):
        raise ValueError("At least one of plot_speed, plot_acg, plot_template must be True.")

    if plot_template and waveform_extractor is None:
        raise ValueError("waveform_extractor must be provided when plot_template=True.")

    if plot_speed and position_df is None:
        raise ValueError("position_df must be provided when plot_speed=True to compute confidence intervals.")

    unit_ids = list(unit_ids)
    n_units = len(unit_ids)

    n_cols = int(plot_speed) + int(plot_acg) + int(plot_template)
    if figsize is None:
        figsize = (3.0 * n_cols, 2.2 * n_units)

    fig, axes = plt.subplots(
        n_units,
        n_cols,
        figsize=figsize,
        squeeze=False,
        sharex="col",   # share x within each column
        sharey="col",   # share y within each column
    )

    # --- precompute occupancy time and speed bins for CIs ---
    if plot_speed:
        timestamps = position_df.index.to_numpy()
        speed = position_df["speed"].to_numpy()

        if mask is None:
            mask_arr = np.ones_like(speed, dtype=bool)
        else:
            mask_arr = np.asarray(mask, dtype=bool)
            if mask_arr.shape[0] != speed.shape[0]:
                raise ValueError("mask must have same length as position_df")

        dt = np.median(np.diff(timestamps))
        n_bins = len(speed_bin_centers)
        if n_bins > 1:
            bin_width = np.median(np.diff(speed_bin_centers))
        else:
            bin_width = speed_bin_centers[0] if speed_bin_centers[0] > 0 else 1.0

        first_edge = max(0.0, speed_bin_centers[0] - bin_width / 2.0)
        speed_bins = first_edge + bin_width * np.arange(n_bins + 1)

        speed_masked = speed[mask_arr]
        occupancy_counts, _ = np.histogram(speed_masked, bins=speed_bins)
        occupancy_time = occupancy_counts * dt

    # --- correlograms for all units once ---
    if plot_acg:
        ccgs, bins = compute_correlograms(
            sorting,
            window_ms=window_ms,
            bin_ms=bin_ms,
        )
        sorting_unit_ids = sorting.get_unit_ids()
        unit_index_map = {u: i for i, u in enumerate(sorting_unit_ids)}
        bin_width_ccg = bins[1] - bins[0]
        bin_x = bins[:-1]

    fs = getattr(waveform_extractor.recording, "sampling_frequency", None) if waveform_extractor is not None else None

    for row_idx, unit in enumerate(unit_ids):
        col_idx = 0
        firing_rate = get_unit_firing_rate(spikes_list, unit)
        unit_label = f"{label_prefix} {unit} ({firing_rate:.2f} Hz)"

        # --- column 1: speed tuning with CI ---
        if plot_speed:
            ax = axes[row_idx, col_idx]
            if unit not in speed_tuning:
                ax.set_axis_off()
            else:
                fr_curve = speed_tuning[unit]

                spike_times = spikes_list[unit]
                speed_at_spikes = spikes_to_speed(spike_times, timestamps, speed, mask=mask_arr)
                spike_counts, _ = np.histogram(speed_at_spikes, bins=speed_bins)

                se_rate = np.full_like(occupancy_time, np.nan, dtype=float)
                valid = occupancy_time > MIN_OCCUPANCY
                se_rate[valid] = np.sqrt(spike_counts[valid]) / occupancy_time[valid]

                z = 1.96
                lower = fr_curve - z * se_rate
                upper = fr_curve + z * se_rate
                lower = np.clip(lower, 0, None)

                ax.plot(speed_bin_centers, fr_curve, marker="o", linewidth=linewidth)
                ax.fill_between(speed_bin_centers, lower, upper, color="C0", alpha=0.3)
                ax.set_ylabel(unit_label)
                if row_idx == n_units - 1:
                    ax.set_xlabel("Speed (cm/s)")
            col_idx += 1

        # --- column 2: autocorrelogram ---
        if plot_acg:
            ax = axes[row_idx, col_idx]
            if unit not in unit_index_map:
                ax.set_axis_off()
            else:
                ui = unit_index_map[unit]
                ccg = ccgs[ui, ui].astype(float)

                # normalize to [0, 1] by max count
                max_ccg = np.nanmax(ccg)
                if max_ccg > 0:
                    ccg = ccg / max_ccg

                ax.bar(bin_x, ccg, width=bin_width_ccg, align="edge", color="k")

                if not plot_speed:
                    ax.set_ylabel(unit_label)
                if row_idx == n_units - 1:
                    ax.set_xlabel("Lag (ms)")

            col_idx += 1


        # --- column 3: template ---
        if plot_template:
            ax = axes[row_idx, col_idx]
            try:
                template = waveform_extractor.get_template(unit, mode="average", force_dense=True)
            except Exception:
                ax.set_axis_off()
            else:
                n_samples, n_channels = template.shape
                t = np.arange(n_samples)
                if fs is not None:
                    t = (t / fs) * 1000.0
                    t_label = "Time (ms)"
                else:
                    t_label = "Samples"

                peak_ch = np.argmax(np.ptp(template, axis=0))
                ax.plot(t, template[:, peak_ch], color="C0", linewidth=linewidth)
                if not (plot_speed or plot_acg):
                    ax.set_ylabel(unit_label)
                if row_idx == n_units - 1:
                    ax.set_xlabel(t_label)
            col_idx += 1

    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()
    return fig, axes



'''----------------------------------------'''

def _mask_to_time_intervals(timestamps, mask):
    """
    Convert a boolean mask over timestamps into a list of (start, end) time intervals.

    Args:
        timestamps (array-like): monotonically increasing time vector.
        mask (array-like of bool): same length as timestamps.

    Returns:
        np.ndarray of shape (n_intervals, 2): [[start0, end0], [start1, end1], ...]
    """
    timestamps = np.asarray(timestamps)
    mask = np.asarray(mask, dtype=bool)

    if timestamps.shape[0] != mask.shape[0]:
        raise ValueError("timestamps and mask must have the same length.")

    if not np.any(mask):
        return np.empty((0, 2), dtype=float)

    mask_int = mask.astype(int)
    diff = np.diff(mask_int)

    # transitions: 0->1 (start), 1->0 (end)
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, len(mask)]

    start_times = timestamps[starts]
    end_times = timestamps[ends - 1]  # inclusive end at last True timestamp

    return np.column_stack([start_times, end_times])


def restrict_sorting_by_position_mask(
    sorting: si.BaseSorting,
    recording: si.BaseRecording,
    position_df: pd.DataFrame,
    mask: np.ndarray,
) -> si.BaseSorting:
    """
    Restrict a SpikeInterface sorting to spikes occurring only when position_df.mask is True.

    Args:
        sorting (si.BaseSorting): original sorting (e.g. epoch2_mpfc_sorting).
        recording (si.BaseRecording): recording registered to this sorting.
        position_df (pd.DataFrame): position dataframe with a time index (same time base as recording).
        mask (array-like of bool): boolean mask aligned with position_df.index
                                   (e.g. position_df['zone'] == 'run').

    Returns:
        si.BaseSorting: new sorting with same unit_ids but spikes only at times
                        where mask is True.
    """
    if sorting.get_num_segments() != 1:
        raise ValueError("restrict_sorting_by_position_mask currently assumes a single segment sorting.")

    timestamps = position_df.index.to_numpy()
    mask_arr = np.asarray(mask, dtype=bool)
    if timestamps.shape[0] != mask_arr.shape[0]:
        raise ValueError("position_df and mask must have the same length.")

    # if mask is all False, return empty sorting with same units
    if not np.any(mask_arr):
        unit_ids = sorting.get_unit_ids()
        empty_unit_dict = {u: np.array([], dtype="int64") for u in unit_ids}
        return si.NumpySorting.from_unit_dict(
            [empty_unit_dict],
            sampling_frequency=sorting.get_sampling_frequency(),
        )

    rec_times = recording.get_times(segment_index=0)
    fs = sorting.get_sampling_frequency()

    unit_ids = sorting.get_unit_ids()
    unit_dict = {}

    t_min = timestamps[0]
    t_max = timestamps[-1]

    for unit_id in unit_ids:
        frames = sorting.get_unit_spike_train(unit_id, segment_index=0)
        if frames.size == 0:
            unit_dict[unit_id] = frames
            continue

        spike_times = rec_times[frames]

        # only consider spikes within the position_df time range
        in_window = (spike_times >= t_min) & (spike_times <= t_max)
        if not np.any(in_window):
            unit_dict[unit_id] = np.array([], dtype="int64")
            continue

        frames_win = frames[in_window]
        times_win = spike_times[in_window]

        # map each spike time to the closest earlier position timestamp
        idx = np.searchsorted(timestamps, times_win, side="right") - 1
        idx = np.clip(idx, 0, len(timestamps) - 1)

        # keep spikes whose corresponding position sample has mask == True
        keep = mask_arr[idx]

        unit_dict[unit_id] = frames_win[keep].astype("int64")

    restricted_sorting = si.NumpySorting.from_unit_dict(
        [unit_dict],
        sampling_frequency=fs,
    )
    return restricted_sorting



# -------------------------------------------------------------------------

def compute_inbound_outbound_firing_rates(
    spikes_list: list,
    trialized_position_df: pd.DataFrame,
    zone_label: str = "run",
) -> pd.DataFrame:
    """
    Compute firing rates for each unit during inbound vs outbound runs.
    Not seperated by trials

    Args:
        spikes_list (list): list of spike time arrays (e.g. epoch2_mpfc_spikes),
                            indexed by group_unit_id.
        trialized_position_df (pd.DataFrame): dataframe with time index and columns
                                              'zone' and 'trial_type'.
        zone_label (str): which zone to treat as "run" for this comparison
                          (default: 'run').

    Returns:
        pd.DataFrame with columns:
            'unit', 'inbound_fr', 'outbound_fr',
            'inbound_fr_se', 'outbound_fr_se',
            'inbound_time', 'outbound_time',
            'n_inbound_spikes', 'n_outbound_spikes'.
    """
    timestamps = trialized_position_df.index.to_numpy()
    zones = trialized_position_df["zone"].to_numpy()
    trial_types = trialized_position_df["trial_type"].to_numpy()

    inbound_mask = (zones == zone_label) & (trial_types == "inbound")
    outbound_mask = (zones == zone_label) & (trial_types == "outbound")

    if not np.any(inbound_mask) and not np.any(outbound_mask):
        raise ValueError(f"No inbound or outbound samples found for zone == '{zone_label}'.")

    dt = np.median(np.diff(timestamps))

    inbound_time = inbound_mask.sum() * dt
    outbound_time = outbound_mask.sum() * dt

    t_min = timestamps[0]
    t_max = timestamps[-1]

    unit_ids = np.arange(len(spikes_list))
    inbound_fr_list = []
    outbound_fr_list = []
    inbound_fr_se_list = []
    outbound_fr_se_list = []
    n_inbound_spikes_list = []
    n_outbound_spikes_list = []
    inbound_time_list = []
    outbound_time_list = []

    for unit in unit_ids:
        spike_times = np.asarray(spikes_list[unit])

        in_window = (spike_times >= t_min) & (spike_times <= t_max)
        spike_times_win = spike_times[in_window]

        if spike_times_win.size == 0:
            inbound_fr_list.append(np.nan)
            outbound_fr_list.append(np.nan)
            inbound_fr_se_list.append(np.nan)
            outbound_fr_se_list.append(np.nan)
            n_inbound_spikes_list.append(0)
            n_outbound_spikes_list.append(0)
            inbound_time_list.append(inbound_time)
            outbound_time_list.append(outbound_time)
            continue

        idx = np.searchsorted(timestamps, spike_times_win, side="right") - 1
        idx = np.clip(idx, 0, len(timestamps) - 1)

        n_inbound_spikes = np.sum(inbound_mask[idx])
        n_outbound_spikes = np.sum(outbound_mask[idx])

        if inbound_time > 0:
            inbound_fr = n_inbound_spikes / inbound_time
            inbound_fr_se = np.sqrt(n_inbound_spikes) / inbound_time if n_inbound_spikes > 0 else 0.0
        else:
            inbound_fr = np.nan
            inbound_fr_se = np.nan

        if outbound_time > 0:
            outbound_fr = n_outbound_spikes / outbound_time
            outbound_fr_se = np.sqrt(n_outbound_spikes) / outbound_time if n_outbound_spikes > 0 else 0.0
        else:
            outbound_fr = np.nan
            outbound_fr_se = np.nan

        inbound_fr_list.append(inbound_fr)
        outbound_fr_list.append(outbound_fr)
        inbound_fr_se_list.append(inbound_fr_se)
        outbound_fr_se_list.append(outbound_fr_se)
        n_inbound_spikes_list.append(n_inbound_spikes)
        n_outbound_spikes_list.append(n_outbound_spikes)
        inbound_time_list.append(inbound_time)
        outbound_time_list.append(outbound_time)

    fr_df = pd.DataFrame(
        {
            "unit": unit_ids,
            "inbound_fr": inbound_fr_list,
            "outbound_fr": outbound_fr_list,
            "inbound_fr_se": inbound_fr_se_list,
            "outbound_fr_se": outbound_fr_se_list,
            "inbound_time": inbound_time_list,
            "outbound_time": outbound_time_list,
            "n_inbound_spikes": n_inbound_spikes_list,
            "n_outbound_spikes": n_outbound_spikes_list,
        }
    )

    return fr_df





def plot_inbound_outbound_fr(fr_df: pd.DataFrame, linewidth: float = 1.5):
    """
    Plot per-unit inbound vs outbound firing rates (no error bars):
      - left: per-unit paired lines
      - right: scatter inbound vs outbound.
    """
    # drop units with NaN (no spikes or zero time)
    fr_df_plot = fr_df.dropna(subset=["inbound_fr", "outbound_fr"]).copy()

    unit_ids = fr_df_plot["unit"].values
    inbound = fr_df_plot["inbound_fr"].values
    outbound = fr_df_plot["outbound_fr"].values
    inbound_se = fr_df_plot["inbound_fr_se"].values
    outbound_se = fr_df_plot["outbound_fr_se"].values

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), tight_layout=True)

    # left: paired lines with error bars
    ax = axes[0]
    for u, fr_in, fr_out, se_in, se_out in zip(unit_ids, inbound, outbound, inbound_se, outbound_se):
        ax.plot([0, 1], [fr_in, fr_out], "-o", color="gray", alpha=0.4, linewidth=linewidth)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Inbound", "Outbound"])
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Per-unit inbound vs outbound firing rates")

    # right: scatter with error bars
    ax = axes[1]
    ax.scatter(inbound, outbound, alpha=0.7)
    max_val = np.nanmax([inbound.max(), outbound.max()]) if inbound.size > 0 else 1.0
    ax.plot([0, max_val], [0, max_val], "k--", label="y = x", linewidth=linewidth)
    ax.set_xlabel("Inbound FR (Hz)")
    ax.set_ylabel("Outbound FR (Hz)")
    ax.set_title("Inbound vs outbound FR")
    ax.axis("equal")
    ax.legend()

    plt.show()

    return fig, axes




def plot_inbound_outbound_fr_side_by_side(fr_df: pd.DataFrame):
    """
    Plot side-by-side inbound vs outbound firing rates for each unit, with error bars.

    Uses Poisson SE from compute_inbound_outbound_firing_rates.
    """
    # drop units with NaN (no spikes or zero time)
    fr_df_plot = fr_df.dropna(subset=["inbound_fr", "outbound_fr"]).copy()

    unit_ids = fr_df_plot["unit"].values
    inbound = fr_df_plot["inbound_fr"].values
    outbound = fr_df_plot["outbound_fr"].values
    inbound_se = fr_df_plot["inbound_fr_se"].values
    outbound_se = fr_df_plot["outbound_fr_se"].values

    n_units = len(unit_ids)
    x = np.arange(n_units)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, 0.3 * n_units), 6), layout="tight")

    ax.bar(
        x - width / 2,
        inbound,
        width,
        yerr=inbound_se,
        label="Inbound",
        capsize=3,
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        outbound,
        width,
        yerr=outbound_se,
        label="Outbound",
        capsize=3,
        alpha=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(unit_ids, rotation=90)
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_xlabel("unit")
    # ax.set_title("Inbound vs outbound firing rate per unit")
    ax.legend()

    plt.show()
    return fig, ax



def compute_trialwise_firing_rates_two_trial_categories(
    spikes_list: list,
    trials_df: pd.DataFrame,
    category_col: str,
    category_a,
    category_b,
    only_correct: bool = True,
    min_duration: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute trial-by-trial firing rates for two trial categories, based on trials_df.

    Each row of trials_df (with trial_start/trial_end) is treated as one trial.
    FR per trial = n_spikes_in_trial / (trial_end - trial_start).

    Args:
        spikes_list (list): list of spike time arrays (e.g. epoch2_mpfc_spikes),
                            indexed by group_unit_id.
        trials_df (pd.DataFrame): output of prepare_trial_data(), with columns
                                  'trial_number', 'trial_start', 'trial_end',
                                  and trial-level metadata (e.g. 'trial_type').
        category_col (str): trial-level column to use (e.g. 'trial_type').
        category_a: first category value (e.g. 'inbound').
        category_b: second category value (e.g. 'outbound').
        only_correct (bool): if True and 'trial_label' exists, keep only
                             trials with trial_label == 'correct'.
        min_duration (float): minimum trial duration (s) to include.

    Returns:
        summary_df (pd.DataFrame): per-unit, per-category summary with columns:
            'unit', 'category', 'fr_mean', 'fr_std', 'fr_sem', 'n_trials',
            'selectivity_index'  # same per unit for both categories
        trialwise_df (pd.DataFrame): per-trial firing rates with columns:
            'unit', 'category', 'trial_number', 'start_time', 'end_time',
            'duration', 'n_spikes', 'fr'.
    """
    # optional: keep only correct trials
    if only_correct and "trial_label" in trials_df.columns:
        valid_trials = trials_df["trial_label"] == "correct"
    else:
        valid_trials = np.ones(len(trials_df), dtype=bool)

    # restrict to the two categories
    trials_use = trials_df[valid_trials & trials_df[category_col].isin([category_a, category_b])].copy()

    if trials_use.empty:
        raise ValueError(
            f"No trials with {category_col} in [{category_a}, {category_b}] "
            "after filtering."
        )

    unit_ids = np.arange(len(spikes_list))
    trial_rows = []

    for _, row in trials_use.iterrows():
        cat_val = row[category_col]
        cat_label = str(cat_val)
        t_start = row["trial_start"]
        t_end = row["trial_end"]
        duration = t_end - t_start
        if duration <= min_duration:
            continue

        trial_num = int(row["trial_number"])

        for unit in unit_ids:
            spike_times = np.asarray(spikes_list[unit])

            in_trial = (spike_times >= t_start) & (spike_times <= t_end)
            n_spikes = int(in_trial.sum())
            fr = n_spikes / duration if duration > 0 else np.nan

            trial_rows.append(
                dict(
                    unit=unit,
                    category=cat_label,
                    trial_number=trial_num,
                    start_time=t_start,
                    end_time=t_end,
                    duration=duration,
                    n_spikes=n_spikes,
                    fr=fr,
                )
            )

    trialwise_df = pd.DataFrame(trial_rows)

    if trialwise_df.empty:
        raise ValueError(
            f"No valid trials for {category_col} in [{category_a}, {category_b}] "
            "after duration filtering."
        )

    # per-unit, per-category summary
    summary_list = []
    grouped = trialwise_df.groupby(["unit", "category"], as_index=False)

    for (unit, cat), group in grouped:
        rates = group["fr"].to_numpy()
        n_trials = len(rates)
        fr_mean = np.nanmean(rates)
        if n_trials > 1:
            fr_std = np.nanstd(rates, ddof=1)
            fr_sem = fr_std / np.sqrt(n_trials)
        else:
            fr_std = 0.0
            fr_sem = 0.0

        summary_list.append(
            dict(
                unit=unit,
                category=cat,
                fr_mean=fr_mean,
                fr_std=fr_std,
                fr_sem=fr_sem,
                n_trials=n_trials,
            )
        )

    summary_df = pd.DataFrame(summary_list)

    # --- add selectivity index per unit ---
    cat_a_str = str(category_a)
    cat_b_str = str(category_b)

    # pivot to wide form: columns are categories, values are fr_mean
    wide = summary_df.pivot(index="unit", columns="category", values="fr_mean")

    # initialize SI with NaN
    si = pd.Series(index=wide.index, dtype=float)

    for unit in wide.index:
        fr_a = wide.loc[unit].get(cat_a_str, np.nan)
        fr_b = wide.loc[unit].get(cat_b_str, np.nan)

        if np.isfinite(fr_a) and np.isfinite(fr_b) and (fr_a + fr_b) > 0:
            si_val = (fr_a - fr_b) / (fr_a + fr_b)
        else:
            si_val = np.nan

        si.loc[unit] = si_val

    si = si.rename("selectivity_index").reset_index()  # columns: ['unit', 'selectivity_index']

    # merge SI back into long summary_df so each row (unit, category) has the unit's SI
    summary_df = summary_df.merge(si, on="unit", how="left")

    return summary_df, trialwise_df



def plot_trialwise_fr_two_categories_side_by_side(
    summary_df: pd.DataFrame,
    category_a,
    category_b,
    use_sem: bool = True,
    title: str = "",
    stats_df: pd.DataFrame = None,
    alpha: float = 0.05,
):
    """
    Side-by-side comparison of trialwise firing rates for two categories, per unit.

    Args:
        summary_df (pd.DataFrame): output of compute_trialwise_firing_rates_two_categories(...)
                                   or compute_trialwise_firing_rates_two_trial_categories(...).
        category_a: first category value (must match 'category' in summary_df).
        category_b: second category value (must match 'category' in summary_df).
        use_sem (bool): if True, use SEM as error bars; if False, use SD.
        title (str): title for the plot.
        stats_df (pd.DataFrame, optional): output of
            test_trialwise_firing_rates_two_trial_categories(...). If provided,
            units with p_value < alpha are marked with an asterisk above their bars.
        alpha (float): significance threshold for marking units as significant.
    """
    cat_a = str(category_a)
    cat_b = str(category_b)

    if use_sem:
        err_col = "fr_sem"
        err_label = "SEM"
    else:
        err_col = "fr_std"
        err_label = "SD"

    df = summary_df[summary_df["category"].isin([cat_a, cat_b])].copy()

    # Only keep units that have both categories
    units_a = set(df[df["category"] == cat_a]["unit"].unique())
    units_b = set(df[df["category"] == cat_b]["unit"].unique())
    common_units = sorted(units_a & units_b)

    rows = []
    for u in common_units:
        row_a = df[(df["unit"] == u) & (df["category"] == cat_a)]
        row_b = df[(df["unit"] == u) & (df["category"] == cat_b)]
        if row_a.empty or row_b.empty:
            continue
        rows.append(
            dict(
                unit=u,
                fr_mean_a=row_a["fr_mean"].iloc[0],
                fr_err_a=row_a[err_col].iloc[0],
                fr_mean_b=row_b["fr_mean"].iloc[0],
                fr_err_b=row_b[err_col].iloc[0],
            )
        )

    if not rows:
        raise ValueError("No units with firing rates in both categories.")

    plot_df = pd.DataFrame(rows)
    unit_ids = plot_df["unit"].values
    fr_a = plot_df["fr_mean_a"].values
    fr_b = plot_df["fr_mean_b"].values
    err_a = plot_df["fr_err_a"].values
    err_b = plot_df["fr_err_b"].values

    n_units = len(unit_ids)
    x = np.arange(n_units)
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, 0.3 * n_units), 6), layout="tight")

    ax.bar(
        x - width / 2,
        fr_a,
        width,
        yerr=err_a,
        label=f"{cat_a} ({err_label})",
        capsize=3,
        alpha=0.8,
    )
    ax.bar(
        x + width / 2,
        fr_b,
        width,
        yerr=err_b,
        label=f"{cat_b} ({err_label})",
        capsize=3,
        alpha=0.8,
    )

    # Mark significant units with an asterisk above the bars, if stats_df is provided
    if stats_df is not None:
        if ("unit" not in stats_df.columns) or ("p_value" not in stats_df.columns):
            raise ValueError("stats_df must contain 'unit' and 'p_value' columns.")

        # map unit -> p_value
        pvals = stats_df.set_index("unit")["p_value"].to_dict()

        for i, u in enumerate(unit_ids):
            p = pvals.get(u, np.nan)
            if not np.isfinite(p) or p >= alpha:
                continue

            # height of the taller bar (mean + error)
            y_max = max(fr_a[i] + err_a[i], fr_b[i] + err_b[i])
            if not np.isfinite(y_max):
                y_max = max(fr_a[i], fr_b[i])

            if not np.isfinite(y_max) or y_max <= 0:
                y_star = 0.05
            else:
                y_star = y_max * 1.05

            ax.text(
                x[i],
                y_star,
                "*",
                ha="center",
                va="bottom",
                fontsize=20,
                color = "red"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(unit_ids, rotation=90)
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_xlabel("unit")
    ax.set_title(title)
    ax.legend()

    plt.show()
    return fig, ax







def test_trialwise_firing_rates_two_trial_categories(
    trialwise_df: pd.DataFrame,
    category_a,
    category_b,
    min_trials_a: int = 3,
    min_trials_b: int = 3,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """
    Per-unit Mann-Whitney U test on trialwise firing rates for two trial categories.

    Args:
        trialwise_df (pd.DataFrame): output of compute_trialwise_firing_rates_two_trial_categories(),
                                     with at least columns:
                                     'unit', 'category', 'trial_number', 'fr'.
        category_a: first category label (e.g. 'inbound').
        category_b: second category label (e.g. 'outbound').
        min_trials_a (int): minimum number of trials in category_a required for the unit.
        min_trials_b (int): minimum number of trials in category_b required for the unit.
        alternative (str): 'two-sided', 'less', or 'greater' (passed to stats.mannwhitneyu).

    Returns:
        pd.DataFrame with one row per unit, columns:
            'unit',
            'n_trials_a', 'n_trials_b',
            'fr_mean_a', 'fr_mean_b',
            'statistic', 'p_value'.
    """
    cat_a = str(category_a)
    cat_b = str(category_b)

    df = trialwise_df[trialwise_df["category"].isin([cat_a, cat_b])].copy()

    results = []
    for unit, group in df.groupby("unit"):
        fr_a = group.loc[group["category"] == cat_a, "fr"].to_numpy()
        fr_b = group.loc[group["category"] == cat_b, "fr"].to_numpy()

        n_a = fr_a.size
        n_b = fr_b.size

        # skip units with too few trials
        if n_a < min_trials_a or n_b < min_trials_b:
            results.append(
                dict(
                    unit=unit,
                    n_trials_a=n_a,
                    n_trials_b=n_b,
                    fr_mean_a=np.nan if n_a == 0 else np.nanmean(fr_a),
                    fr_mean_b=np.nan if n_b == 0 else np.nanmean(fr_b),
                    statistic=np.nan,
                    p_value=np.nan,
                )
            )
            continue

        fr_mean_a = float(np.nanmean(fr_a))
        fr_mean_b = float(np.nanmean(fr_b))

        try:
            stat, p_val = stats.mannwhitneyu(fr_a, fr_b, alternative=alternative)
        except ValueError:
            # e.g. all values equal across both samples
            stat, p_val = np.nan, np.nan

        results.append(
            dict(
                unit=unit,
                n_trials_a=n_a,
                n_trials_b=n_b,
                fr_mean_a=fr_mean_a,
                fr_mean_b=fr_mean_b,
                statistic=stat,
                p_value=p_val,
            )
        )
        

    results_df = pd.DataFrame(results)
    results_df["preference"] = np.where(
    results_df["fr_mean_a"] > results_df["fr_mean_b"],
    category_a,
    category_b
)
    return results_df


def compute_trialwise_firing_rates_two_trial_categories_with_pos_mask(
    spikes_list: list,
    trials_df: pd.DataFrame,
    position_df: pd.DataFrame,
    pos_mask,
    category_col: str,
    category_a,
    category_b,
    only_correct: bool = True,
    min_duration: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Trial-by-trial firing rates for two trial categories, restricted by a
    position-based mask (e.g. track_segment_id == 2).

    Each row of trials_df is still treated as one trial, but:
      - Time denominator is the *total time within that trial* where pos_mask is True.
      - Spikes are counted only when they occur in that trial AND at samples
        where pos_mask is True.

    Args:
        spikes_list (list): list of spike time arrays (e.g. epoch2_mpfc_spikes),
                            indexed by group_unit_id.
        trials_df (pd.DataFrame): output of prepare_trial_data(), with columns
                                  'trial_number', 'trial_start', 'trial_end',
                                  and trial-level metadata (e.g. 'trial_type').
        position_df (pd.DataFrame): trialized_position-like dataframe with time
                                    index and columns like 'trial_number',
                                    'track_segment_id', 'zone', etc.
        pos_mask: boolean mask on position_df (Series or array); for example:
                  (position_df['track_segment_id'] == 2)
                  (position_df['zone'] == 'run') & (position_df['track_segment_id'] == 2)
        category_col (str): trial-level column to use (e.g. 'trial_type').
        category_a: first category value (e.g. 'inbound').
        category_b: second category value (e.g. 'outbound').
        only_correct (bool): if True and 'trial_label' exists, keep only
                             trials with trial_label == 'correct'.
        min_duration (float): minimum *masked* duration (s) to include a trial.

    Returns:
        summary_df (pd.DataFrame): per-unit, per-category summary with columns:
            'unit', 'category', 'fr_mean', 'fr_std', 'fr_sem', 'n_trials',
            'selectivity_index'
        trialwise_df (pd.DataFrame): per-trial firing rates with columns:
            'unit', 'category', 'trial_number', 'start_time', 'end_time',
            'duration', 'n_spikes', 'fr'.
    """
    # --- align position + mask ---
    timestamps = position_df.index.to_numpy()
    if timestamps.size < 2:
        raise ValueError("position_df must have at least 2 timepoints.")

    # approximate dt in seconds from position_df index
    dt = np.median(np.diff(timestamps))

    # ensure pos_mask is boolean aligned with position_df.index
    if isinstance(pos_mask, (pd.Series, pd.Index)):
        pos_mask_arr = pd.Series(pos_mask, index=position_df.index)
        pos_mask_arr = pos_mask_arr.reindex(position_df.index).fillna(False).to_numpy(dtype=bool)
    else:
        pos_mask_arr = np.asarray(pos_mask, dtype=bool)
        if pos_mask_arr.shape[0] != len(position_df):
            raise ValueError("pos_mask must have same length as position_df.")


    if "trial_number" not in position_df.columns:
        raise ValueError("position_df must contain a 'trial_number' column.")

    pos_trial_numbers = position_df["trial_number"].to_numpy()
    t_min, t_max = timestamps[0], timestamps[-1]

    # --- filter trials by correctness and category ---
    if only_correct and "trial_label" in trials_df.columns:
        valid_trials = trials_df["trial_label"] == "correct"
    else:
        valid_trials = np.ones(len(trials_df), dtype=bool)

    trials_use = trials_df[
        valid_trials & trials_df[category_col].isin([category_a, category_b])
    ].copy()

    if trials_use.empty:
        raise ValueError(
            f"No trials with {category_col} in [{category_a}, {category_b}] after filtering."
        )

    unit_ids = np.arange(len(spikes_list))
    trial_rows = []

    for _, tr in trials_use.iterrows():
        trial_num = int(tr["trial_number"])
        cat_val = tr[category_col]
        cat_label = str(cat_val) ##########
        t_start = tr["trial_start"]
        t_end = tr["trial_end"]

        # position samples within this trial AND satisfying pos_mask
        in_this_trial = (pos_trial_numbers == trial_num)
        in_time = (timestamps >= t_start) & (timestamps <= t_end)
        pos_trial_mask = pos_mask_arr & in_this_trial & in_time

        # total time in this trial under the positional condition
        duration = pos_trial_mask.sum() * dt
        if duration <= min_duration:
            # skip trials where the animal doesn't spend enough time in this condition
            continue

        for unit in unit_ids:
            spike_times = np.asarray(spikes_list[unit])

            # first restrict spikes to overall position time window
            in_window = (spike_times >= t_min) & (spike_times <= t_max)
            spike_times_win = spike_times[in_window]

            # then restrict to this trial's time window
            in_trial = (spike_times_win >= t_start) & (spike_times_win <= t_end)
            spike_times_trial = spike_times_win[in_trial]

            if spike_times_trial.size == 0:
                n_spikes = 0
                fr = 0.0
            else:
                # map spikes to nearest earlier position sample
                idx = np.searchsorted(timestamps, spike_times_trial, side="right") - 1
                idx = np.clip(idx, 0, len(timestamps) - 1)

                # keep spikes where the corresponding position sample passes pos_trial_mask
                keep = pos_trial_mask[idx]
                n_spikes = int(keep.sum())
                fr = n_spikes / duration if duration > 0 else np.nan

            trial_rows.append(
                dict(
                    unit=unit,
                    category=cat_label,
                    trial_number=trial_num,
                    start_time=t_start,
                    end_time=t_end,
                    duration=duration,
                    n_spikes=n_spikes,
                    fr=fr,
                )
            )

    trialwise_df = pd.DataFrame(trial_rows)
    if trialwise_df.empty:
        raise ValueError(
            "No valid trials after applying positional mask and duration filtering."
        )

    # --- per-unit, per-category summary (same pattern as your existing function) ---
    summary_list = []
    grouped = trialwise_df.groupby(["unit", "category"], as_index=False)

    for (unit, cat), group in grouped:
        rates = group["fr"].to_numpy()
        n_trials = len(rates)
        fr_mean = np.nanmean(rates)
        if n_trials > 1:
            fr_std = np.nanstd(rates, ddof=1)
            fr_sem = fr_std / np.sqrt(n_trials)
        else:
            fr_std = 0.0
            fr_sem = 0.0

        summary_list.append(
            dict(
                unit=unit,
                category=cat,
                fr_mean=fr_mean,
                fr_std=fr_std,
                fr_sem=fr_sem,
                n_trials=n_trials,
            )
        )

    summary_df = pd.DataFrame(summary_list)

    # --- add selectivity index (same definition as in spike_analysis.py) ---
    cat_a_str = str(category_a)
    cat_b_str = str(category_b)

    wide = summary_df.pivot(index="unit", columns="category", values="fr_mean")

    si = pd.Series(index=wide.index, dtype=float)
    for unit in wide.index:
        fr_a = wide.loc[unit].get(cat_a_str, np.nan)
        fr_b = wide.loc[unit].get(cat_b_str, np.nan)
        if np.isfinite(fr_a) and np.isfinite(fr_b) and (fr_a + fr_b) > 0:
            si_val = (fr_a - fr_b) / (fr_a + fr_b)
        else:
            si_val = np.nan
        si.loc[unit] = si_val

    si = si.rename("selectivity_index").reset_index()
    summary_df = summary_df.merge(si, on="unit", how="left")

    return summary_df, trialwise_df




def spikes_to_position(spike_times, timestamps, position, mask=None):
    """
    Map spike times to speed values at the closest earlier timestamp.
    If mask is provided (boolean array same length as timestamps),
    only keep spikes whose corresponding timestamp has mask == True.
    """
    timestamps = np.asarray(timestamps)
    position = np.asarray(position)
    spike_times = np.asarray(spike_times)

    t0, t1 = timestamps[0], timestamps[-1]
    in_window = (spike_times >= t0) & (spike_times <= t1)
    spike_times_win = spike_times[in_window]

    spike_idx = np.searchsorted(timestamps, spike_times_win, side='right') - 1
    spike_idx = np.clip(spike_idx, 0, len(timestamps) - 1)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        keep = mask[spike_idx]
        spike_idx = spike_idx[keep]

    return position[spike_idx]



def compute_location_tuning(position_df: pd.DataFrame,
                         spikes_list: list,
                         n_bins: int = 20,
                         mask: np.ndarray = None):
    """
    position_df: full position_trials_merged_df (NOT pre-masked)
    mask: boolean array on position_df.index (e.g. zone=='run' & trial_type=='inbound')
    """

    timestamps = position_df.index.to_numpy()
    position = position_df['linear_position'].to_numpy()


    if mask is None:
        mask = np.ones_like(position, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != position.shape[0]:
            raise ValueError("mask must have same length as position_df")


    position_masked = position[mask]

    dt = np.median(np.diff(timestamps))
    max_position = np.nanmax(position_masked)
    position_bins = np.linspace(0, max_position, n_bins + 1)
    position_bin_centers = (position_bins[:-1] + position_bins[1:]) / 2

    occupancy_counts, _ = np.histogram(position_masked, bins=position_bins)
    occupancy_time = occupancy_counts * dt

    position_tuning = {}

    for unit, spike_times in enumerate(spikes_list):
        position_at_spikes = spikes_to_position(spike_times, timestamps, position, mask=mask)
        spike_counts, _ = np.histogram(position_at_spikes, bins=position_bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            firing_rate = spike_counts / occupancy_time
            firing_rate[occupancy_time == 0] = np.nan

        position_tuning[unit] = firing_rate

    return position_tuning, position_bin_centers



def plot_position_tuning_grid(position_tuning: dict,
                           position_df: pd.DataFrame,
                           spikes_list: list,
                           mask: np.ndarray = None,
                           n_units: int = -1,
                           label = None,
                           linewidth: float = 1.0):
    """
    Plot per-unit position tuning curves with 95% CI.

    position_df: full position_trials_merged_df (NOT pre-masked)
    mask: boolean array aligned with position_df.index (e.g. zone=='run' & trial_type=='inbound')
    """

    # --- prepare timebase + mask on the FULL df ---
    timestamps = position_df.index.to_numpy()
    position = position_df['linear_position'].to_numpy()

    if mask is None:
        mask = np.ones_like(position, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != position.shape[0]:
            raise ValueError("mask must have same length as position_df")

    # infer number of bins from the tuning dict
    first_unit = next(iter(position_tuning))
    n_bins = len(position_tuning[first_unit])

    # occupancy only where mask is True
    position_masked = position[mask]
    dt = np.median(np.diff(timestamps))

    max_position = np.nanmax(position_masked)
    position_bins = np.linspace(0, max_position, n_bins + 1)
    position_bin_centers = (position_bins[:-1] + position_bins[1:]) / 2

    occupancy_counts, _ = np.histogram(position_masked, bins=position_bins)
    occupancy_time = occupancy_counts * dt  # seconds in each position bin

    # --- helper: spike counts per position bin, using the same mask ---
    def _spike_counts_per_position_bin(spike_times):
        position_at_spikes = spikes_to_position(spike_times, timestamps, position, mask=mask)
        spike_counts, _ = np.histogram(position_at_spikes, bins=position_bins)
        return spike_counts

    # --- choose units to plot ---
    all_units = list(position_tuning.keys())
    if n_units < 0 or n_units > len(all_units):
        units = all_units
    else:
        units = all_units[:n_units]
    n_units = len(units)

    n_cols = 5
    n_rows = int(np.ceil(n_units / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2 * n_rows),
        sharex=True, sharey=True
    )
    axes = np.array(axes).reshape(-1)

    # --- per-unit curves + CIs ---
    for i, unit in enumerate(units):
        spike_times = spikes_list[unit]
        spike_counts = _spike_counts_per_position_bin(spike_times)

        valid = occupancy_time > MIN_OCCUPANCY

        rate = np.full_like(occupancy_time, np.nan, dtype=float)
        se_rate = np.full_like(occupancy_time, np.nan, dtype=float)

        rate[valid] = spike_counts[valid] / occupancy_time[valid]
        se_rate[valid] = np.sqrt(spike_counts[valid]) / occupancy_time[valid]

        z = 1.96  # ~95% CI
        lower = np.clip(rate - z * se_rate, 0, None)
        upper = rate + z * se_rate

        ax = axes[i]
        ax.plot(position_bin_centers, rate, marker="o", linewidth=linewidth)
        ax.fill_between(position_bin_centers, lower, upper,
                        color='C0', alpha=0.3, label='95% CI')

        ax.set_title(str(unit), fontsize=8)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=6)

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'position tuning curves {label}', fontsize=14)
    fig.text(0.5, 0.04, 'position (cm)', ha='center')
    fig.text(0.04, 0.5, 'Firing rate (Hz)', va='center', rotation='vertical')
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.show()
    
    
    
    
def spikes_to_tuner(spike_times, timestamps, tuner, mask=None):
    """
    Map spike times to speed values at the closest earlier timestamp.
    If mask is provided (boolean array same length as timestamps),
    only keep spikes whose corresponding timestamp has mask == True.
    """
    timestamps = np.asarray(timestamps)
    tuner = np.asarray(tuner)
    spike_times = np.asarray(spike_times)

    t0, t1 = timestamps[0], timestamps[-1]
    in_window = (spike_times >= t0) & (spike_times <= t1)
    spike_times_win = spike_times[in_window]

    spike_idx = np.searchsorted(timestamps, spike_times_win, side='right') - 1
    spike_idx = np.clip(spike_idx, 0, len(timestamps) - 1)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        keep = mask[spike_idx]
        spike_idx = spike_idx[keep]

    return tuner[spike_idx]


def _validate_binning_mode(binning: Optional[str]) -> str:
    if binning is None:
        return "edges"
    binning = str(binning).strip().lower()
    if binning in {"edge", "edges", "discrete", "hist", "histogram"}:
        return "edges"
    if binning in {"sliding", "moving", "moving_window", "moving-window", "continuous"}:
        return "sliding"
    raise ValueError(f"Unknown binning mode {binning!r}. Expected 'edges' or 'sliding'.")


def _counts_in_windows(values: np.ndarray, left_edges: np.ndarray, right_edges: np.ndarray) -> np.ndarray:
    """
    Count how many `values` fall in each half-open interval [left, right).

    This is used for sliding-window ("continuous") binning where windows can overlap.
    """
    left_edges = np.asarray(left_edges, dtype=float)
    right_edges = np.asarray(right_edges, dtype=float)
    if left_edges.shape != right_edges.shape:
        raise ValueError("left_edges and right_edges must have the same shape")

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.zeros(left_edges.shape, dtype=np.int64)

    values_sorted = np.sort(values)
    li = np.searchsorted(values_sorted, left_edges, side="left")
    ri = np.searchsorted(values_sorted, right_edges, side="left")
    return (ri - li).astype(np.int64)


def _prepare_tuner_binning(
    tuner_values_masked: np.ndarray,
    *,
    n_bins: int,
    tuner_bins: Optional[np.ndarray],
    binning: Optional[str],
    window_width: Optional[float],
    window_step: Optional[float],
    tuner_min: Optional[float],
    tuner_max: Optional[float],
) -> Dict[str, Any]:
    """
    Build either traditional histogram bin edges ("edges") or overlapping sliding windows ("sliding").

    Returns a dict with keys:
      - mode: 'edges' or 'sliding'
      - centers: (n_bins,) bin/window centers
      - left: (n_bins,) left edges
      - right: (n_bins,) right edges
      - edges: (n_bins+1,) edges (only for mode='edges')
    """
    mode = _validate_binning_mode(binning)

    tuner_values_masked = np.asarray(tuner_values_masked, dtype=float)
    finite_vals = tuner_values_masked[np.isfinite(tuner_values_masked)]
    if finite_vals.size == 0:
        raise ValueError("No finite tuner values under mask; cannot bin.")

    if mode == "edges":
        if tuner_bins is None:
            max_tuner = float(np.nanmax(finite_vals))
            if not np.isfinite(max_tuner):
                raise ValueError("Non-finite max tuner value; cannot bin.")
            edges = np.linspace(0.0, max_tuner, int(n_bins) + 1)
        else:
            edges = np.asarray(tuner_bins, dtype=float)
            if edges.ndim != 1:
                raise ValueError("tuner_bins must be 1D")
            if len(edges) != int(n_bins) + 1:
                raise ValueError("tuner_bins must have length n_bins+1")
            if not np.all(np.diff(edges) > 0):
                raise ValueError("tuner_bins must be strictly increasing")

        centers = (edges[:-1] + edges[1:]) / 2.0
        return {
            "mode": "edges",
            "centers": centers,
            "left": edges[:-1],
            "right": edges[1:],
            "edges": edges,
        }

    # --- sliding windows ---
    if tuner_bins is not None:
        raise ValueError("tuner_bins is only used for binning='edges'.")

    if window_width is None or not np.isfinite(window_width) or window_width <= 0:
        raise ValueError("For binning='sliding', window_width must be a positive number.")

    if window_step is None:
        window_step = window_width / 2.0
    if not np.isfinite(window_step) or window_step <= 0:
        raise ValueError("For binning='sliding', window_step must be a positive number.")

    if tuner_min is None:
        tuner_min = float(np.nanmin(finite_vals))
    if tuner_max is None:
        tuner_max = float(np.nanmax(finite_vals))
    if not np.isfinite(tuner_min) or not np.isfinite(tuner_max) or tuner_max <= tuner_min:
        raise ValueError("Invalid tuner_min/tuner_max for sliding-window binning.")

    start = tuner_min + window_width / 2.0
    stop = tuner_max - window_width / 2.0
    if stop < start:
        raise ValueError(
            "window_width is larger than the available tuner range under mask; "
            "choose a smaller window_width or specify a wider tuner_min/tuner_max."
        )

    # include stop (within tolerance) so endpoints behave as expected
    centers = np.arange(start, stop + (window_step * 0.5), window_step, dtype=float)
    left = centers - window_width / 2.0
    right = centers + window_width / 2.0
    return {
        "mode": "sliding",
        "centers": centers,
        "left": left,
        "right": right,
        "edges": None,
        "window_width": float(window_width),
        "window_step": float(window_step),
        "tuner_min": float(tuner_min),
        "tuner_max": float(tuner_max),
    }


def _peak_normalize_from_upper_ci(
    rate: np.ndarray,
    lower: Optional[np.ndarray],
    upper: np.ndarray,
) -> tuple[np.ndarray, Optional[np.ndarray], np.ndarray, float]:
    """
    Peak-normalize using a CI-derived scale.

    Scale is defined as the maximum of the *upper* confidence bound across x (ignoring NaNs).
    """
    rate = np.asarray(rate, dtype=float)
    upper = np.asarray(upper, dtype=float)
    lower_arr = None if lower is None else np.asarray(lower, dtype=float)

    scale = float(np.nanmax(upper)) if np.any(np.isfinite(upper)) else float("nan")
    if (not np.isfinite(scale)) or scale <= 0:
        return rate, lower_arr, upper, float("nan")

    rate_n = rate / scale
    upper_n = upper / scale
    lower_n = None if lower_arr is None else (lower_arr / scale)
    return rate_n, lower_n, upper_n, scale


def _peak_normalize_dicts_from_upper_ci(
    *,
    tuning: dict,
    lower_ci: Optional[dict],
    upper_ci: dict,
    slope_boot: Optional[dict] = None,
    curvature_boot: Optional[dict] = None,
) -> dict:
    """
    In-place normalize dict outputs (unit -> array) by per-unit CI-derived scale.

    Returns {unit: scale}.
    """
    scales = {}
    for unit in list(tuning.keys()):
        if unit not in upper_ci:
            continue

        rate = np.asarray(tuning[unit], dtype=float)
        lo = None if lower_ci is None else np.asarray(lower_ci[unit], dtype=float)
        hi = np.asarray(upper_ci[unit], dtype=float)

        rate_n, lo_n, hi_n, scale = _peak_normalize_from_upper_ci(rate, lo, hi)
        if not np.isfinite(scale) or scale <= 0:
            scales[unit] = float("nan")
            continue

        tuning[unit] = rate_n
        upper_ci[unit] = hi_n
        if lower_ci is not None and lo_n is not None:
            lower_ci[unit] = lo_n

        if slope_boot is not None and unit in slope_boot:
            slope_boot[unit] = np.asarray(slope_boot[unit], dtype=float) / scale
        if curvature_boot is not None and unit in curvature_boot:
            curvature_boot[unit] = np.asarray(curvature_boot[unit], dtype=float) / scale

        scales[unit] = scale

    return scales




def compute_tuning(position_df: pd.DataFrame,
                         column: str,
                         spikes_list: list,
                         n_bins: int = 6,
                         mask: np.ndarray = None,
                         tuner_bins: np.ndarray = None,
                         binning: str = "edges",
                         window_width: Optional[float] = None,
                         window_step: Optional[float] = None,
                         tuner_min: Optional[float] = None,
                         tuner_max: Optional[float] = None,
                         peak_normalize: bool = False):
    """
    position_df: full tuner_trials_merged_df (NOT pre-masked)
    mask: boolean array on position_df.index (e.g. zone=='run' & trial_type=='inbound')

    peak_normalize
    --------------
    If True, divide each unit's tuning curve by a per-unit scale defined as:
    `max_x upper_95%CI(x)`, where the 95% CI is a Poisson approximation based on
    spike counts and occupancy time (using `MIN_OCCUPANCY` to avoid unstable bins).
    """

    timestamps = position_df.index.to_numpy()
    tuner = position_df[column].to_numpy()


    if mask is None:
        mask = np.ones_like(tuner, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != tuner.shape[0]:
            raise ValueError("mask must have same length as position df")


    tuner_masked = tuner[mask]

    dt = np.median(np.diff(timestamps))
    bin_spec = _prepare_tuner_binning(
        tuner_masked,
        n_bins=n_bins,
        tuner_bins=tuner_bins,
        binning=binning,
        window_width=window_width,
        window_step=window_step,
        tuner_min=tuner_min,
        tuner_max=tuner_max,
    )
    tuner_bin_centers = bin_spec["centers"]

    if bin_spec["mode"] == "edges":
        occupancy_counts, _ = np.histogram(tuner_masked, bins=bin_spec["edges"])
    else:
        occupancy_counts = _counts_in_windows(tuner_masked, bin_spec["left"], bin_spec["right"])
    occupancy_time = occupancy_counts * dt

    tuner_tuning = {}

    for unit, spike_times in enumerate(spikes_list):
        tuner_at_spikes = spikes_to_tuner(spike_times, timestamps, tuner, mask=mask)
        if bin_spec["mode"] == "edges":
            spike_counts, _ = np.histogram(tuner_at_spikes, bins=bin_spec["edges"])
        else:
            spike_counts = _counts_in_windows(tuner_at_spikes, bin_spec["left"], bin_spec["right"])

        with np.errstate(divide='ignore', invalid='ignore'):
            firing_rate = spike_counts / occupancy_time
            firing_rate[occupancy_time == 0] = np.nan

        if peak_normalize:
            valid = occupancy_time > MIN_OCCUPANCY
            se_rate = np.full_like(occupancy_time, np.nan, dtype=float)
            ok = valid & np.isfinite(occupancy_time) & (occupancy_time > 0)
            se_rate[ok] = np.sqrt(spike_counts[ok]) / occupancy_time[ok]

            z = 1.96  # ~95% CI
            upper = firing_rate + z * se_rate
            upper[~valid] = np.nan
            scale = float(np.nanmax(upper)) if np.any(np.isfinite(upper)) else float("nan")
            if np.isfinite(scale) and scale > 0:
                firing_rate = firing_rate / scale

        tuner_tuning[unit] = firing_rate

    return tuner_tuning, tuner_bin_centers

def compute_tuning_bootstrap_trials(
    position_df: pd.DataFrame,
    column: str,
    spikes_list: list,
    n_bins: int = 6,
    mask: np.ndarray = None,
    trial_column: str = "trial_number",
    n_boot: int = 500,
    ci: float = 0.95,
    random_state: int = 0,
    tuner_bins: np.ndarray = None,
    binning: str = "edges",
    window_width: Optional[float] = None,
    window_step: Optional[float] = None,
    tuner_min: Optional[float] = None,
    tuner_max: Optional[float] = None,
    peak_normalize: bool = False,
):
    """
    Trial-bootstrap tuning curves + bootstrap replicate shape stats.

    Binning modes
    ------------
    - Default (`binning='edges'`): standard non-overlapping histogram bins (optionally via `tuner_bins`).
    - Sliding windows (`binning='sliding'`): overlapping "continuous" bins using a moving window;
      set `window_width` and optionally `window_step`, and (optionally) `tuner_min`/`tuner_max`.

    peak_normalize
    --------------
    If True, divide each unit's pooled curve, CI curves, and (slope/curvature) bootstrap statistics
    by a per-unit scale defined as `max_x upper_CI(x)` (i.e., the maximum of the upper confidence
    bound across bins). This uses whatever `ci` you requested (commonly 0.95).

    Returns
    -------
    tuner_tuning : dict[int, np.ndarray]
        Unit -> pooled firing rate per bin (Hz), same idea as compute_tuning().
    tuner_bin_centers : np.ndarray
        Bin centers for `column`.
    lower_ci : dict[int, np.ndarray]
        Unit -> lower CI per bin (percentile CI across trial bootstraps).
    upper_ci : dict[int, np.ndarray]
        Unit -> upper CI per bin.
    slope_boot : dict[int, np.ndarray]
        Unit -> bootstrap replicate linear slopes (np.nan if not enough valid bins).
    curvature_boot : dict[int, np.ndarray]
        Unit -> bootstrap replicate quadratic curvature (x^2 coefficient; np.nan if not enough valid bins).

    Notes on the slope/curvature definitions
    ---------------------------------------
    - slope is from a linear fit: y = a + b*x  -> returns b
    - curvature is from a quadratic fit: y = a + b*x + q*x^2 -> returns q
      (q < 0 suggests concave-down / “bell-ish”, q > 0 concave-up / “U-ish”)
    """

    if trial_column not in position_df.columns:
        raise KeyError(f"position_df missing required column: {trial_column!r}")

    timestamps = position_df.index.to_numpy()
    tuner = position_df[column].to_numpy()
    trial_ids_raw = position_df[trial_column].to_numpy()

    if len(timestamps) < 2:
        raise ValueError("position_df must have at least 2 timestamped samples")

    if not pd.Index(position_df.index).is_monotonic_increasing:
        raise ValueError(
            "position_df.index must be sorted/monotonic increasing for spike alignment "
            "(call position_df = position_df.sort_index() before building mask)."
        )

    if mask is None:
        mask = np.ones_like(tuner, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != tuner.shape[0]:
            raise ValueError("mask must have same length as position_df")

    # ignore rows without trial ids
    trial_ok = ~pd.isna(trial_ids_raw)
    mask = mask & trial_ok

    dt = np.median(np.diff(timestamps))

    tuner_masked = tuner[mask]
    if tuner_masked.size == 0:
        raise ValueError("Mask leaves zero samples; cannot compute tuning.")

    # max_tuner = np.nanmax(tuner_masked)
    # if not np.isfinite(max_tuner):
    #     raise ValueError(f"Non-finite max for {column!r} under mask; cannot bin.")

    # tuner_bins = np.linspace(0, max_tuner, n_bins + 1)
    # tuner_bin_centers = (tuner_bins[:-1] + tuner_bins[1:]) / 2

    mode = _validate_binning_mode(binning)
    if mode == "edges":
        if tuner_bins is None:
            max_tuner = np.nanmax(tuner_masked)
            tuner_bins = np.linspace(0, max_tuner, n_bins + 1)
        else:
            tuner_bins = np.asarray(tuner_bins, dtype=float)
            if tuner_bins.ndim != 1:
                raise ValueError("tuner_bins must be 1D")
            if len(tuner_bins) != n_bins + 1:
                raise ValueError("tuner_bins must have length n_bins+1")
            if not np.all(np.diff(tuner_bins) > 0):
                raise ValueError("tuner_bins must be strictly increasing")

        tuner_bin_centers = (tuner_bins[:-1] + tuner_bins[1:]) / 2
        bin_spec = {"mode": "edges", "edges": tuner_bins, "centers": tuner_bin_centers}
    else:
        if tuner_bins is not None:
            raise ValueError("tuner_bins is only used for binning='edges'.")

        bin_spec = _prepare_tuner_binning(
            tuner_masked,
            n_bins=n_bins,
            tuner_bins=None,
            binning="sliding",
            window_width=window_width,
            window_step=window_step,
            tuner_min=tuner_min,
            tuner_max=tuner_max,
        )
        tuner_bin_centers = bin_spec["centers"]
        n_bins = int(tuner_bin_centers.size)



    # Trials list
    trial_ids_masked = np.asarray(trial_ids_raw[mask]).astype(int)
    trial_ids = np.unique(trial_ids_masked)
    trial_ids = np.sort(trial_ids)
    n_trials = len(trial_ids)
    if n_trials < 2:
        raise ValueError(f"Need >=2 trials for trial bootstrap; found {n_trials}.")

    if mode == "edges":
        # Map masked samples -> trial indices 0..n_trials-1
        trial_idx_per_sample = np.searchsorted(trial_ids, trial_ids_masked)

        # Map masked samples -> bin indices 0..n_bins-1
        tuner_masked_finite = np.isfinite(tuner_masked)
        bin_idx = np.searchsorted(tuner_bins, tuner_masked, side="right") - 1
        bin_idx[bin_idx == n_bins] = n_bins - 1  # exact right edge
        valid_bin = tuner_masked_finite & (bin_idx >= 0) & (bin_idx < n_bins)

        # Occupancy per (trial, bin)
        flat_occ = np.bincount(
            trial_idx_per_sample[valid_bin] * n_bins + bin_idx[valid_bin],
            minlength=n_trials * n_bins,
        )
        occupancy_counts_trials = flat_occ.reshape(n_trials, n_bins)
        occupancy_time_trials = occupancy_counts_trials * dt  # seconds
    else:
        # Sliding windows: samples can contribute to multiple windows, so we count per trial per window.
        left_edges = bin_spec["left"]
        right_edges = bin_spec["right"]

        tuner_masked_vals = np.asarray(tuner_masked, dtype=float)
        trial_ids_masked_int = np.asarray(trial_ids_masked, dtype=int)

        finite = np.isfinite(tuner_masked_vals)
        tuner_masked_vals = tuner_masked_vals[finite]
        trial_ids_masked_int = trial_ids_masked_int[finite]

        trial_idx = np.searchsorted(trial_ids, trial_ids_masked_int)
        order = np.argsort(trial_idx, kind="mergesort")
        trial_idx_sorted = trial_idx[order]
        tuner_sorted_by_trial = tuner_masked_vals[order]

        occupancy_counts_trials = np.zeros((n_trials, n_bins), dtype=np.int64)
        for t in range(n_trials):
            s = np.searchsorted(trial_idx_sorted, t, side="left")
            e = np.searchsorted(trial_idx_sorted, t, side="right")
            if e <= s:
                continue
            vals = tuner_sorted_by_trial[s:e]
            occupancy_counts_trials[t, :] = _counts_in_windows(vals, left_edges, right_edges)

        occupancy_time_trials = occupancy_counts_trials * dt  # seconds

    occupancy_time_total = occupancy_time_trials.sum(axis=0)

    # Bootstrap weights (multinomial == resampling trials with replacement)
    rng = np.random.default_rng(random_state)
    weights = rng.multinomial(n_trials, np.ones(n_trials) / n_trials, size=n_boot).astype(float)

    occupancy_time_boot = weights @ occupancy_time_trials  # (n_boot, n_bins)

    # CI quantiles
    alpha = 1.0 - float(ci)
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)

    # ---- helpers for per-replicate slope/curvature ----
    x = np.asarray(tuner_bin_centers, dtype=float)

    def _linear_slope(xv, yv):
        # returns b in y = a + b*x
        if xv.size < 2:
            return np.nan
        x_mean = np.mean(xv)
        y_mean = np.mean(yv)
        denom = np.sum((xv - x_mean) ** 2)
        if denom <= 0:
            return np.nan
        return np.sum((xv - x_mean) * (yv - y_mean)) / denom

    def _quadratic_curvature(xv, yv):
        # returns q in y = a + b*x + q*x^2
        if xv.size < 3:
            return np.nan
        X = np.column_stack([np.ones_like(xv), xv, xv ** 2])
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        return float(beta[2])

    # ---- spike counts per (trial, bin) per unit ----
    def _spike_counts_trials_for_unit(spike_times: np.ndarray) -> np.ndarray:
        spike_times = np.asarray(spike_times)

        t0, t1 = timestamps[0], timestamps[-1]
        in_window = (spike_times >= t0) & (spike_times <= t1)
        spike_times_win = spike_times[in_window]
        if spike_times_win.size == 0:
            return np.zeros((n_trials, n_bins), dtype=np.int64)

        spike_pos_idx = np.searchsorted(timestamps, spike_times_win, side="right") - 1
        spike_pos_idx = np.clip(spike_pos_idx, 0, len(timestamps) - 1)

        keep = mask[spike_pos_idx]
        if not np.any(keep):
            return np.zeros((n_trials, n_bins), dtype=np.int64)

        spike_pos_idx = spike_pos_idx[keep]

        spike_trials = trial_ids_raw[spike_pos_idx] #trial number for each spike
        spike_tuner = tuner[spike_pos_idx] # tuner value for each spike

        ok = (~pd.isna(spike_trials)) & np.isfinite(spike_tuner)
        if not np.any(ok):
            return np.zeros((n_trials, n_bins), dtype=np.int64)

        spike_trials = spike_trials[ok].astype(int)
        spike_tuner = spike_tuner[ok]

        spike_trial_idx = np.searchsorted(trial_ids, spike_trials)
        valid_trial = (
            (spike_trial_idx >= 0)
            & (spike_trial_idx < n_trials)
            & (trial_ids[spike_trial_idx] == spike_trials)
        )
        if not np.any(valid_trial):
            return np.zeros((n_trials, n_bins), dtype=np.int64)

        spike_trial_idx = spike_trial_idx[valid_trial]
        spike_tuner = spike_tuner[valid_trial]

        if mode == "edges":
            spike_bin_idx = np.searchsorted(tuner_bins, spike_tuner, side="right") - 1
            spike_bin_idx[spike_bin_idx == n_bins] = n_bins - 1

            valid = (spike_bin_idx >= 0) & (spike_bin_idx < n_bins)
            if not np.any(valid):
                return np.zeros((n_trials, n_bins), dtype=np.int64)

            flat = np.bincount(
                spike_trial_idx[valid] * n_bins + spike_bin_idx[valid],
                minlength=n_trials * n_bins,
            )
            return flat.reshape(n_trials, n_bins)

        # sliding-window mode
        left_edges = bin_spec["left"]
        right_edges = bin_spec["right"]
        out = np.zeros((n_trials, n_bins), dtype=np.int64)

        order = np.argsort(spike_trial_idx, kind="mergesort")
        trial_idx_sorted = spike_trial_idx[order]
        tuner_sorted_by_trial = spike_tuner[order]

        for t in range(n_trials):
            s = np.searchsorted(trial_idx_sorted, t, side="left")
            e = np.searchsorted(trial_idx_sorted, t, side="right")
            if e <= s:
                continue
            vals = tuner_sorted_by_trial[s:e]
            out[t, :] = _counts_in_windows(vals, left_edges, right_edges)

        return out

    # Precompute spike counts by trial/bin for all units (fast boot later)
    spike_counts_trials_units = []
    for spike_times in spikes_list:
        spike_counts_trials_units.append(_spike_counts_trials_for_unit(spike_times))

    # ---- outputs ----
    tuner_tuning = {}
    lower_ci = {}
    upper_ci = {}
    slope_boot = {}
    curvature_boot = {}

    pooled_valid_bins = (occupancy_time_total > MIN_OCCUPANCY) & np.isfinite(x)

    for unit, counts_trials in enumerate(spike_counts_trials_units):
        # pooled (point estimate)
        spike_counts_total = counts_trials.sum(axis=0)
        rate = np.full(n_bins, np.nan, dtype=float)
        valid_occ = occupancy_time_total > MIN_OCCUPANCY
        rate[valid_occ] = spike_counts_total[valid_occ] / occupancy_time_total[valid_occ]
        tuner_tuning[unit] = rate

        # bootstrap pooled rates: (n_boot, n_bins)
        spike_counts_boot = weights @ counts_trials
        with np.errstate(divide="ignore", invalid="ignore"):
            rates_boot = spike_counts_boot / occupancy_time_boot

        # apply occupancy threshold per bootstrap replicate/bin
        rates_boot[occupancy_time_boot <= MIN_OCCUPANCY] = np.nan

        lower_ci[unit] = np.nanpercentile(rates_boot, lo_q, axis=0)
        upper_ci[unit] = np.nanpercentile(rates_boot, hi_q, axis=0)

        # shape stats per bootstrap replicate (no long-term storage of full curves needed)
        s_arr = np.full(n_boot, np.nan, dtype=float)
        q_arr = np.full(n_boot, np.nan, dtype=float)

        # Compute per replicate using whatever bins are finite (and pooled-valid)
        for r in range(n_boot):
            y = rates_boot[r, :]
            ok = pooled_valid_bins & np.isfinite(y)
            xv = x[ok]
            yv = y[ok]
            s_arr[r] = _linear_slope(xv, yv)
            q_arr[r] = _quadratic_curvature(xv, yv)

        slope_boot[unit] = s_arr
        curvature_boot[unit] = q_arr

    if peak_normalize:
        _peak_normalize_dicts_from_upper_ci(
            tuning=tuner_tuning,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            slope_boot=slope_boot,
            curvature_boot=curvature_boot,
        )

    return tuner_tuning, tuner_bin_centers, lower_ci, upper_ci, slope_boot, curvature_boot



#plot SINGLE UNIT tuning
def plot_unit_tuning_bootstrap(
    unit_id,
    tuning_dict,
    centers,
    lower_ci_dict,
    upper_ci_dict,
    *,
    ax=None,
    color="C0",
    marker="o",
    s=3.0,            
    linewidth=1.5,
    ci_alpha=0.25,
    show_ci=True,
    show_stderr=False,   
    ci=0.95,
    stderr_dict=None,   
    peak_normalize=False,  
    xlabel=None,
    ylabel=None,
    title=None,
    ylim=None,
):
    """
    Plot a single unit's bootstrapped tuning curve with CI band.

    Parameters
    ----------
    unit_id : int (or whatever keys your dicts use)
    tuning_dict/lower_ci_dict/upper_ci_dict : dict[unit_id] -> 1D array
    centers : 1D array of bin centers
    ax : matplotlib Axes (optional)
    s : markersize (NOT scatter 's')
    peak_normalize : uses scale = max(upper_ci) for that unit
    """
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(5, 3))

    if unit_id not in tuning_dict:
        raise KeyError(f"unit_id {unit_id!r} not found in tuning_dict keys (n={len(tuning_dict)}).")
    if unit_id not in lower_ci_dict or unit_id not in upper_ci_dict:
        raise KeyError(f"unit_id {unit_id!r} missing from lower/upper CI dicts.")

    x = np.asarray(centers, dtype=float)
    y = np.asarray(tuning_dict[unit_id], dtype=float)
    lo = np.asarray(lower_ci_dict[unit_id], dtype=float)
    hi = np.asarray(upper_ci_dict[unit_id], dtype=float)

    # optional peak normalization using CI-derived scale
    if peak_normalize:
        scale = np.nanmax(hi) if np.any(np.isfinite(hi)) else np.nan
        if np.isfinite(scale) and scale > 0:
            y = y / scale
            lo = lo / scale
            hi = hi / scale

            if stderr_dict is not None and unit_id in stderr_dict:
                stderr_dict = dict(stderr_dict)
                stderr_dict[unit_id] = np.asarray(stderr_dict[unit_id], dtype=float) / scale

    # main curve
    ax.plot(
        x, y,
        color=color,
        marker=marker,
        markersize=s,
        linewidth=linewidth,
    )

    # CI band
    if show_ci:
        ax.fill_between(x, lo, hi, color=color, alpha=ci_alpha, linewidth=0)

    # standard error (either provided, or inferred from CI width assuming normality)
    if show_stderr:
        if stderr_dict is not None and unit_id in stderr_dict:
            se = np.asarray(stderr_dict[unit_id], dtype=float)
        else:
            try:
                from statistics import NormalDist
                alpha = 1.0 - float(ci)
                z = NormalDist().inv_cdf(1.0 - alpha / 2.0)
            except Exception:
                z = 1.96  # fallback for ~95%
            se = (hi - lo) / (2.0 * z) if (np.isfinite(z) and z > 0) else np.full_like(y, np.nan)

        ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(se) & (se >= 0)
        if np.any(ok):
            ax.errorbar(
                x[ok], y[ok],
                yerr=se[ok],
                fmt="none",
                ecolor=color,
                elinewidth=max(0.75, linewidth * 0.75),
                alpha=0.8,
                capsize=0,
            )

    # labels
    if xlabel is None:
        xlabel = "speed_norm"  # change if you're plotting something else
    if ylabel is None:
        ylabel = "Normalized firing rate" if peak_normalize else "Firing rate (Hz)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        title = f"unit {unit_id}"
    ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(ylim)
    elif peak_normalize:
        ax.set_ylim(-0.05, 1.05)

    ax.axhline(0, color="k", linewidth=0.5, alpha=0.5)
    return ax



def illustrate_pvalue_rule_for_unit(
    unit_id,
    *,
    null_metric_dict,          # e.g. null_curve_correlation
    contig_metric_dict,        # e.g. contiguous_curve_correlation
    alpha=0.05,
    direction="lower",         # "lower" for corr (low is worse), "upper" for nrmse/peak_shift (high is worse)
    bins=50,
    ax=None,
    hist_color="0.35",
    hist_alpha=0.75,
    contig_color="tab:blue",
    thresh_color="tab:red",
    shade_color="tab:red",
    shade_alpha=0.15,
    title=None,
    xlabel=None,
):
    null_vals = np.asarray(null_metric_dict[unit_id], dtype=float)
    null_vals = null_vals[np.isfinite(null_vals)]
    contig_val = float(contig_metric_dict[unit_id])

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    else:
        fig = ax.figure

    if null_vals.size == 0 or not np.isfinite(contig_val):
        ax.set_title(f"unit {unit_id}: insufficient data")
        return fig, ax

    # p-value rule
    if direction == "lower":
        p = float(np.mean(null_vals <= contig_val))
        q = 100.0 * float(alpha)            # left-tail threshold
    elif direction == "upper":
        p = float(np.mean(null_vals >= contig_val))
        q = 100.0 * (1.0 - float(alpha))    # right-tail threshold
    else:
        raise ValueError("direction must be 'lower' or 'upper'")

    sig = p < alpha
    thresh = float(np.nanpercentile(null_vals, q))

    # histogram FIRST (sets reasonable x-limits)
    ax.hist(null_vals, bins=bins, color=hist_color, alpha=hist_alpha, edgecolor="white", linewidth=0.5)

    # get finite axis limits and compute shading span
    xmin, xmax = ax.get_xlim()
    if not np.isfinite(xmin) or not np.isfinite(xmax):
        xmin = float(np.nanmin(null_vals))
        xmax = float(np.nanmax(null_vals))
        ax.set_xlim(xmin, xmax)
        xmin, xmax = ax.get_xlim()

    if direction == "lower":
        x0 = xmin
        x1 = min(thresh, xmax)
    else:
        x0 = max(thresh, xmin)
        x1 = xmax

    # only shade if the span has positive width
    if np.isfinite(x0) and np.isfinite(x1) and (x1 > x0):
        ax.axvspan(
            x0, x1,
            color=shade_color,
            alpha=shade_alpha,
            zorder=0,
            label=f"rejection region (α={alpha})",
        )

    # draw threshold + observed value
    ax.axvline(thresh, color=thresh_color, linewidth=2, linestyle="--",
               label=f"null threshold = {thresh:.3g}")
    ax.axvline(contig_val, color=contig_color, linewidth=2,
               label=f"contig = {contig_val:.3g}")

    # annotate p-value
    ax.text(
        0.02, 0.98,
        f"p = {p:.4g}\n(sig @ α={alpha}: {sig})",
        transform=ax.transAxes,
        va="top", ha="left",
        # fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="0.8"),
    )

    if xlabel is None:
        xlabel = "metric"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")

    if title is None:
        title = f"Unit {unit_id}: null vs contig ({direction}-tail test)"
    ax.set_title(title)
    ax.legend(frameon=False)
        # font sizes
    ax.tick_params(axis="both", which="both", labelsize=30)
    ax.xaxis.label.set_size(30); ax.yaxis.label.set_size(30); ax.title.set_size(30)

    return fig, ax




def compare_speed_tuning_across_progress_segments(
    trialized_position_df: pd.DataFrame,
    spikes_list: list,
    base_mask: np.ndarray,
    progress_col: str,
    progress_edges: np.ndarray,
    progress_labels: list,
    speed_col: str = "speed",
    trial_col: str = "trial_number",
    tuner_bins: np.ndarray = None,  # used for binning='edges' OR (by default) for stratification when binning='sliding'
    n_boot: int = 500,
    random_state: int = 0,
    # occupancy / stratification controls
    min_speedbin_occupancy_s: float = 1.0,
    stratify_equalize_speedbin_occupancy: bool = True,
    # NEW: make this compatible with compute_tuning_bootstrap_trials(binning="sliding")
    binning: str = "sliding",
    window_width: float = 20.0,
    window_step: float = 2.0,
    tuner_min: float = 0.0,
    tuner_max: float = 120.0,
    stratify_bins: np.ndarray = None,
    peak_normalize: bool = True,
):
    """
    Returns
    -------
    segment_results[label] = dict(
        tuning, lo, hi, centers, mask,
        occ_s_per_tuningbin,
        slope_boot, curv_boot,
    )
    valid_speed_bins_mask : np.ndarray[bool]
        Mask over *tuning bins* (i.e., histogram bins for binning='edges' OR sliding-window centers for binning='sliding')
        that have >= min_speedbin_occupancy_s in ALL segments after applying segment_masks.
    pairwise_metrics[(label_i, label_j)] : dict
        Per-unit curve comparison metrics, computed after restricting curves to valid_speed_bins_mask.
    """
    def _validate_binning_mode(m: str) -> str:
        if m is None:
            return "edges"
        m = str(m).strip().lower()
        if m in {"edge", "edges", "discrete", "hist", "histogram"}:
            return "edges"
        if m in {"sliding", "moving", "moving_window", "moving-window", "continuous"}:
            return "sliding"
        raise ValueError(f"Unknown binning mode {m!r}. Expected 'edges' or 'sliding'.")

    def _counts_in_windows(values: np.ndarray, left_edges: np.ndarray, right_edges: np.ndarray) -> np.ndarray:
        # counts in half-open intervals [left, right)
        left_edges = np.asarray(left_edges, dtype=float)
        right_edges = np.asarray(right_edges, dtype=float)
        if left_edges.shape != right_edges.shape:
            raise ValueError("left_edges and right_edges must have the same shape")
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return np.zeros(left_edges.shape, dtype=np.int64)
        values_sorted = np.sort(values)
        li = np.searchsorted(values_sorted, left_edges, side="left")
        ri = np.searchsorted(values_sorted, right_edges, side="left")
        return (ri - li).astype(np.int64)

    df = trialized_position_df.copy()

    mode = _validate_binning_mode(binning)

    # dt from index
    timestamps = df.index.to_numpy()
    if len(timestamps) < 2:
        raise ValueError("df must have at least 2 timestamped samples")
    dt = float(np.median(np.diff(timestamps)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Bad dt; ensure df.index sorted and in seconds.")

    # base validity
    base_mask = np.asarray(base_mask, bool).copy()
    base_mask &= df[trial_col].notna().to_numpy()
    speed_vals = df[speed_col].to_numpy(dtype=float)
    prog_vals_raw = df[progress_col].to_numpy(dtype=float)
    base_mask &= np.isfinite(speed_vals)
    base_mask &= np.isfinite(prog_vals_raw)

    # For sliding windows, avoid "stratify bins grabbing out-of-range samples that never contribute to tuning"
    # by restricting to the analysis range.
    if mode == "edges":
        if tuner_bins is None:
            raise ValueError("Pass tuner_bins when binning='edges'.")
        tuner_bins = np.asarray(tuner_bins, dtype=float)
        if tuner_bins.ndim != 1 or tuner_bins.size < 2 or not np.all(np.diff(tuner_bins) > 0):
            raise ValueError("tuner_bins must be 1D and strictly increasing.")
        base_mask &= (speed_vals >= tuner_bins[0]) & (speed_vals <= tuner_bins[-1])
    else:
        if window_width is None or (not np.isfinite(window_width)) or window_width <= 0:
            raise ValueError("For binning='sliding', window_width must be a positive number.")
        if window_step is None:
            window_step = window_width / 2.0
        if (not np.isfinite(window_step)) or window_step <= 0:
            raise ValueError("For binning='sliding', window_step must be a positive number.")
        if tuner_min is None or tuner_max is None:
            # If user leaves these as None, lock them globally (across ALL segments) so centers match.
            masked_vals = speed_vals[base_mask]
            if masked_vals.size == 0:
                raise ValueError("Mask leaves zero samples; cannot infer tuner_min/tuner_max.")
            if tuner_min is None:
                tuner_min = float(np.nanmin(masked_vals))
            if tuner_max is None:
                tuner_max = float(np.nanmax(masked_vals))
        if (not np.isfinite(tuner_min)) or (not np.isfinite(tuner_max)) or tuner_max <= tuner_min:
            raise ValueError("Invalid tuner_min/tuner_max for sliding-window binning.")
        base_mask &= (speed_vals >= tuner_min) & (speed_vals <= tuner_max)

    # assign segment labels
    progress_vals = np.clip(prog_vals_raw, 0.0, 1.0)
    seg_series = pd.cut(
        progress_vals,
        bins=np.asarray(progress_edges, float),
        include_lowest=True,
        right=True,
        labels=progress_labels,
    )
    seg_labels = seg_series.to_numpy()

    # ---- define tuning bins/windows (must be FIXED across segments) ----
    if mode == "edges":
        tuning_centers = (tuner_bins[:-1] + tuner_bins[1:]) / 2.0
        tuning_left = tuner_bins[:-1]
        tuning_right = tuner_bins[1:]
        n_tuning_bins = int(tuning_centers.size)
    else:
        start = float(tuner_min) + float(window_width) / 2.0
        stop = float(tuner_max) - float(window_width) / 2.0
        if stop < start:
            raise ValueError(
                "window_width is larger than the available tuner range; "
                "choose smaller window_width or widen tuner_min/tuner_max."
            )
        tuning_centers = np.arange(start, stop + (float(window_step) * 0.5), float(window_step), dtype=float)
        tuning_left = tuning_centers - float(window_width) / 2.0
        tuning_right = tuning_centers + float(window_width) / 2.0
        n_tuning_bins = int(tuning_centers.size)
        if n_tuning_bins < 2:
            raise ValueError("Sliding-window setup produced <2 tuning bins; adjust range/step/width.")

    # ---- stratification bins (NON-overlapping) used only for downsampling ----
    if stratify_bins is not None:
        stratify_bins = np.asarray(stratify_bins, dtype=float)
    elif tuner_bins is not None:
        # Keep old behavior: if you passed tuner_bins, use it for stratification by default.
        stratify_bins = np.asarray(tuner_bins, dtype=float)
    else:
        # Reasonable default for sliding: non-overlapping bins at window_step resolution.
        if mode == "sliding":
            bw = float(window_step)
            if bw <= 0 or (not np.isfinite(bw)):
                bw = float(window_width) / 2.0
            stratify_bins = np.arange(float(tuner_min), float(tuner_max) + bw, bw, dtype=float)
        else:
            raise ValueError("Need tuner_bins or stratify_bins for stratification.")

    if stratify_bins.ndim != 1 or stratify_bins.size < 2 or not np.all(np.diff(stratify_bins) > 0):
        raise ValueError("stratify_bins must be 1D and strictly increasing.")
    n_strat_bins = int(stratify_bins.size - 1)

    # assign stratification bin index per sample (0..n_strat_bins-1), mark out-of-range invalid
    in_strat_range = (speed_vals >= stratify_bins[0]) & (speed_vals <= stratify_bins[-1]) & np.isfinite(speed_vals)
    base_mask &= in_strat_range

    strat_bin_idx = np.searchsorted(stratify_bins, speed_vals, side="right") - 1
    strat_bin_idx[strat_bin_idx == n_strat_bins] = n_strat_bins - 1  # exact right edge
    strat_bin_idx = np.clip(strat_bin_idx, 0, n_strat_bins - 1)

    # ---- gather indices per segment x stratbin ----
    occ_s_by_segment_strat = {}
    sample_indices_by_segment_and_stratbin = {}

    for label in progress_labels:
        seg_mask = base_mask & (seg_labels == label)
        seg_idx = np.where(seg_mask)[0]

        b = strat_bin_idx[seg_idx]
        counts = np.bincount(b, minlength=n_strat_bins)
        occ_s = counts * dt
        occ_s_by_segment_strat[label] = occ_s

        sample_indices_by_segment_and_stratbin[label] = [
            seg_idx[b == bin_id] for bin_id in range(n_strat_bins)
        ]

    # valid strat bins: enough occupancy in ALL segments (this is what we can *fairly* equalize over)
    valid_strat_bins_mask = np.ones(n_strat_bins, dtype=bool)
    for label in progress_labels:
        valid_strat_bins_mask &= (occ_s_by_segment_strat[label] >= float(min_speedbin_occupancy_s))

    if valid_strat_bins_mask.sum() < 2:
        raise ValueError(
            f"Too few overlapping stratification bins across segments with >= {min_speedbin_occupancy_s}s occupancy. "
            f"Try lowering min_speedbin_occupancy_s, widening stratify bins, or using fewer progress segments."
        )

    # ---- build per-segment sample masks (optionally equalized across strat bins) ----
    segment_masks = {}
    rng = np.random.default_rng(random_state)

    if stratify_equalize_speedbin_occupancy:
        target_occ_s_per_bin = np.zeros(n_strat_bins, dtype=float)
        for bin_id in range(n_strat_bins):
            if not valid_strat_bins_mask[bin_id]:
                continue
            target_occ_s_per_bin[bin_id] = min(
                occ_s_by_segment_strat[label][bin_id] for label in progress_labels
            )
        target_counts_per_bin = np.floor(target_occ_s_per_bin / dt).astype(int)

        for label in progress_labels:
            keep_indices = []
            for bin_id in range(n_strat_bins):
                if not valid_strat_bins_mask[bin_id]:
                    continue
                bin_indices = sample_indices_by_segment_and_stratbin[label][bin_id]
                k = int(target_counts_per_bin[bin_id])
                if k <= 0 or bin_indices.size == 0:
                    continue
                if bin_indices.size <= k:
                    chosen = bin_indices
                else:
                    chosen = rng.choice(bin_indices, size=k, replace=False)
                keep_indices.append(chosen)

            keep_indices = np.concatenate(keep_indices) if keep_indices else np.array([], dtype=int)
            mask = np.zeros(len(df), dtype=bool)
            mask[keep_indices] = True
            segment_masks[label] = mask
    else:
        # "common support only" at strat-bin level: keep all samples in segment but restrict to valid strat bins
        for label in progress_labels:
            seg_mask = base_mask & (seg_labels == label) & valid_strat_bins_mask[strat_bin_idx]
            segment_masks[label] = seg_mask

    # ---- occupancy per *tuning bin/window* AFTER applying segment_masks ----
    occ_s_by_segment_tuning = {}
    for label in progress_labels:
        vals = speed_vals[segment_masks[label]]
        if mode == "edges":
            occ_counts = np.histogram(vals, bins=tuner_bins)[0].astype(np.int64)
        else:
            occ_counts = _counts_in_windows(vals, tuning_left, tuning_right).astype(np.int64)
        occ_s_by_segment_tuning[label] = occ_counts * dt

    # valid tuning bins/windows are those with enough occupancy in ALL segments
    valid_speed_bins_mask = np.ones(n_tuning_bins, dtype=bool)
    for label in progress_labels:
        valid_speed_bins_mask &= (occ_s_by_segment_tuning[label] >= float(min_speedbin_occupancy_s))

    if valid_speed_bins_mask.sum() < 2:
        raise ValueError(
            f"Too few overlapping tuning bins/windows across segments with >= {min_speedbin_occupancy_s}s occupancy. "
            f"Try lowering min_speedbin_occupancy_s, using a larger window_width, or widening tuner_min/tuner_max."
        )

    # ---- compute tuning per segment ----
    segment_results = {}
    for label in progress_labels:
        out = compute_tuning_bootstrap_trials(
            df,
            column=speed_col,
            spikes_list=spikes_list,
            n_bins=n_tuning_bins,
            mask=segment_masks[label],
            trial_column=trial_col,
            n_boot=n_boot,
            ci=0.95,
            random_state=random_state,
            tuner_bins=(tuner_bins if mode == "edges" else None),
            binning=mode,
            window_step=(float(window_step) if mode == "sliding" else None),
            window_width=(float(window_width) if mode == "sliding" else None),
            tuner_min=(float(tuner_min) if mode == "sliding" else None),
            tuner_max=(float(tuner_max) if mode == "sliding" else None),
            peak_normalize=peak_normalize,
        )
        tuning, centers, lo, hi, slope_boot, curv_boot = out

        # sanity: centers should be identical across segments if we fixed params
        if centers.shape[0] != n_tuning_bins:
            raise ValueError(
                f"compute_tuning_bootstrap_trials returned {centers.shape[0]} bins, "
                f"but compare_* expected {n_tuning_bins}."
            )

        segment_results[label] = dict(
            tuning=tuning,
            centers=centers,
            lo=lo,
            hi=hi,
            mask=segment_masks[label],
            occ_s_per_tuningbin=occ_s_by_segment_tuning[label],
            slope_boot=slope_boot,
            curv_boot=curv_boot,
        )

    # ---- pairwise curve-comparison metrics (per unit), restricted to valid tuning bins ----
    valid_bins = valid_speed_bins_mask

    def restrict_curve(curve):
        c = np.asarray(curve, dtype=float).copy()
        if c.shape[0] != valid_bins.shape[0]:
            raise ValueError("Curve length != valid_bins length; check binning params.")
        c[~valid_bins] = np.nan
        return c

    pairwise_metrics = {}
    for i in range(len(progress_labels)):
        for j in range(i + 1, len(progress_labels)):
            li = progress_labels[i]
            lj = progress_labels[j]

            units_i = set(segment_results[li]["tuning"].keys())
            units_j = set(segment_results[lj]["tuning"].keys())
            common_units = sorted(units_i & units_j)

            metrics_per_unit = {}
            for unit_id in common_units:
                ci = restrict_curve(segment_results[li]["tuning"][unit_id])
                cj = restrict_curve(segment_results[lj]["tuning"][unit_id])

                metrics_per_unit[unit_id] = dict(
                    corr=curve_corr(ci, cj),
                    nrmse=nrmse(ci, cj),
                    peak_shift_bins=peak_bin_shift(ci, cj),
                    mean_rate_diff=mean_rate_diff(ci, cj),
                )

            pairwise_metrics[(li, lj)] = metrics_per_unit

    return segment_results, valid_speed_bins_mask, pairwise_metrics




# =============================================================================
# Generic permutation / null-distribution significance tests for tuning curves
# =============================================================================
#
# Designed for re-use in unit_tuning.ipynb for comparisons like:
#   - inbound vs outbound (shuffle trial labels)
#   - left vs right choice (shuffle trial labels)
#   - early vs late progress segments (paired within-trial swap)
#
# Key idea: precompute per-trial occupancy + spike counts once, then re-aggregate
# under permutations; this is much faster than recomputing tuning from scratch.


@dataclass(frozen=True)
class CurveMetricSpec:
    """
    Defines a curve comparison metric and how it is converted to a "difference statistic"
    where larger = more different.

    The permutation p-value is computed as upper-tail:
        p = P(null_stat >= observed_stat)
    """

    name: str
    func: Callable[[np.ndarray, np.ndarray, np.ndarray, int], float]
    to_diff_stat: Callable[[float], float]


def default_curve_metric_specs(*, use_peak_shift_x: bool = True) -> Tuple[CurveMetricSpec, ...]:
    """
    Default metric set used in unit_tuning.ipynb-style comparisons.

    Difference-stat convention:
      - corr: lower corr => more different => stat = 1 - corr
      - nrmse: higher => more different => stat = nrmse
      - peak shift: higher => more different => stat = peak_shift_x or peak_bin_shift
      - mean_rate_diff: higher => more different => stat = mean_rate_diff
    """
    peak_func = peak_shift_x if use_peak_shift_x else peak_bin_shift
    peak_name = "peak_shift_x" if use_peak_shift_x else "peak_shift_bins"
    return (
        CurveMetricSpec("corr", curve_corr, to_diff_stat=lambda r: (1.0 - r) if np.isfinite(r) else float("nan")),
        CurveMetricSpec("nrmse", lambda a, b, x, m: nrmse(a, b, x, m), to_diff_stat=lambda e: e),
        CurveMetricSpec(peak_name, peak_func, to_diff_stat=lambda p: p),
        CurveMetricSpec("mean_rate_diff", mean_rate_diff, to_diff_stat=lambda d: d),
    )


def fdr_bh(p_values: Union[np.ndarray, Sequence[float]], *, alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Benjamini–Hochberg FDR correction.

    Returns:
      reject: boolean array same shape as p_values
      q: adjusted q-values (NaN where p is NaN)
    """
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    reject = np.zeros_like(p, dtype=bool)

    finite = np.isfinite(p)
    if not np.any(finite):
        return reject, q

    p_f = p[finite]
    m = int(p_f.size)
    order = np.argsort(p_f)
    ranked = p_f[order]

    q_ranked = ranked * (m / (np.arange(1, m + 1)))
    q_ranked = np.minimum.accumulate(q_ranked[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0.0, 1.0)

    q_f = np.empty_like(p_f)
    q_f[order] = q_ranked
    q[finite] = q_f
    reject[finite] = q_f <= float(alpha)
    return reject, q


def _peak_normalize_rates_max(rates_units_bins: np.ndarray) -> np.ndarray:
    """
    Peak-normalize per unit using max(rate) across bins (ignoring NaNs).
    (This differs from compute_tuning_bootstrap_trials peak_normalize, which uses a CI-derived scale.)
    """
    rates_units_bins = np.asarray(rates_units_bins, dtype=float)
    scale = np.nanmax(rates_units_bins, axis=1)
    out = rates_units_bins.copy()
    good = np.isfinite(scale) & (scale > 0)
    out[good, :] = out[good, :] / scale[good, None]
    return out


def precompute_trial_binned_counts(
    position_df: pd.DataFrame,
    *,
    column: str,
    spikes_list: Sequence[np.ndarray],
    mask: np.ndarray,
    trial_column: str = "trial_number",
    n_bins: int = 8,
    tuner_bins: Optional[np.ndarray] = None,
    binning: str = "sliding",
    window_width: Optional[float] = None,
    window_step: Optional[float] = None,
    tuner_min: Optional[float] = None,
    tuner_max: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Precompute per-trial occupancy and spike counts in tuner bins/windows.

    Output dict keys:
      - trial_ids: (n_trials,) int trial ids
      - centers: (n_bins_eff,) bin centers
      - dt: float seconds/sample
      - mode: 'edges' or 'sliding'
      - occupancy_time_trials: (n_trials, n_bins_eff) seconds
      - spike_counts_trials_units: (n_units, n_trials, n_bins_eff) counts
    """
    if trial_column not in position_df.columns:
        raise KeyError(f"position_df missing required column: {trial_column!r}")

    timestamps = position_df.index.to_numpy()
    tuner = position_df[column].to_numpy(dtype=float)
    trial_ids_raw = position_df[trial_column].to_numpy()

    if len(timestamps) < 2:
        raise ValueError("position_df must have at least 2 timestamped samples")

    if not pd.Index(position_df.index).is_monotonic_increasing:
        raise ValueError(
            "position_df.index must be sorted/monotonic increasing for spike alignment "
            "(call position_df = position_df.sort_index() before building mask)."
        )

    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (tuner.shape[0],):
        raise ValueError("mask must have same length as position_df")
    mask = mask & (~pd.isna(trial_ids_raw))

    dt = float(np.median(np.diff(timestamps)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Bad dt derived from index; ensure df.index sorted and in seconds.")

    tuner_masked = tuner[mask]
    if tuner_masked.size == 0:
        raise ValueError("Mask leaves zero samples; cannot precompute.")

    mode = _validate_binning_mode(binning)
    bin_spec = _prepare_tuner_binning(
        tuner_masked,
        n_bins=int(n_bins),
        tuner_bins=tuner_bins,
        binning=mode,
        window_width=window_width,
        window_step=window_step,
        tuner_min=tuner_min,
        tuner_max=tuner_max,
    )
    centers = np.asarray(bin_spec["centers"], dtype=float)
    n_bins_eff = int(centers.size)

    trial_ids_masked = np.asarray(trial_ids_raw[mask]).astype(int)
    trial_ids = np.sort(np.unique(trial_ids_masked))
    n_trials = int(trial_ids.size)
    if n_trials < 1:
        raise ValueError("No trials remain after mask; cannot precompute.")

    # ---- occupancy per (trial, bin) ----
    if mode == "edges":
        edges = np.asarray(bin_spec["edges"], dtype=float)
        trial_idx_per_sample = np.searchsorted(trial_ids, trial_ids_masked)
        tuner_masked_finite = np.isfinite(tuner_masked)
        bin_idx = np.searchsorted(edges, tuner_masked, side="right") - 1
        bin_idx[bin_idx == n_bins_eff] = n_bins_eff - 1
        valid_bin = tuner_masked_finite & (bin_idx >= 0) & (bin_idx < n_bins_eff)

        flat_occ = np.bincount(
            trial_idx_per_sample[valid_bin] * n_bins_eff + bin_idx[valid_bin],
            minlength=n_trials * n_bins_eff,
        )
        occupancy_counts_trials = flat_occ.reshape(n_trials, n_bins_eff)
    else:
        left_edges = np.asarray(bin_spec["left"], dtype=float)
        right_edges = np.asarray(bin_spec["right"], dtype=float)

        tuner_masked_vals = np.asarray(tuner_masked, dtype=float)
        trial_ids_masked_int = np.asarray(trial_ids_masked, dtype=int)
        finite = np.isfinite(tuner_masked_vals)
        tuner_masked_vals = tuner_masked_vals[finite]
        trial_ids_masked_int = trial_ids_masked_int[finite]

        trial_idx = np.searchsorted(trial_ids, trial_ids_masked_int)
        order = np.argsort(trial_idx, kind="mergesort")
        trial_idx_sorted = trial_idx[order]
        tuner_sorted_by_trial = tuner_masked_vals[order]

        occupancy_counts_trials = np.zeros((n_trials, n_bins_eff), dtype=np.int64)
        for t in range(n_trials):
            s = np.searchsorted(trial_idx_sorted, t, side="left")
            e = np.searchsorted(trial_idx_sorted, t, side="right")
            if e <= s:
                continue
            vals = tuner_sorted_by_trial[s:e]
            occupancy_counts_trials[t, :] = _counts_in_windows(vals, left_edges, right_edges)

    occupancy_time_trials = occupancy_counts_trials.astype(float) * dt

    # ---- spike counts per (trial, bin) per unit ----
    t0, t1 = timestamps[0], timestamps[-1]
    spike_counts_trials_units = []
    for spike_times in spikes_list:
        spike_times = np.asarray(spike_times)
        in_window = (spike_times >= t0) & (spike_times <= t1)
        st = spike_times[in_window]
        if st.size == 0:
            spike_counts_trials_units.append(np.zeros((n_trials, n_bins_eff), dtype=np.int64))
            continue

        spike_pos_idx = np.searchsorted(timestamps, st, side="right") - 1
        spike_pos_idx = np.clip(spike_pos_idx, 0, len(timestamps) - 1)

        keep = mask[spike_pos_idx]
        if not np.any(keep):
            spike_counts_trials_units.append(np.zeros((n_trials, n_bins_eff), dtype=np.int64))
            continue

        spike_pos_idx = spike_pos_idx[keep]
        spike_trials = trial_ids_raw[spike_pos_idx]
        spike_tuner = tuner[spike_pos_idx]

        ok = (~pd.isna(spike_trials)) & np.isfinite(spike_tuner)
        if not np.any(ok):
            spike_counts_trials_units.append(np.zeros((n_trials, n_bins_eff), dtype=np.int64))
            continue

        spike_trials = spike_trials[ok].astype(int)
        spike_tuner = spike_tuner[ok].astype(float)

        spike_trial_idx = np.searchsorted(trial_ids, spike_trials)
        valid_trial = (
            (spike_trial_idx >= 0)
            & (spike_trial_idx < n_trials)
            & (trial_ids[spike_trial_idx] == spike_trials)
        )
        if not np.any(valid_trial):
            spike_counts_trials_units.append(np.zeros((n_trials, n_bins_eff), dtype=np.int64))
            continue

        spike_trial_idx = spike_trial_idx[valid_trial]
        spike_tuner = spike_tuner[valid_trial]

        if mode == "edges":
            edges = np.asarray(bin_spec["edges"], dtype=float)
            spike_bin_idx = np.searchsorted(edges, spike_tuner, side="right") - 1
            spike_bin_idx[spike_bin_idx == n_bins_eff] = n_bins_eff - 1
            valid = (spike_bin_idx >= 0) & (spike_bin_idx < n_bins_eff)
            if not np.any(valid):
                spike_counts_trials_units.append(np.zeros((n_trials, n_bins_eff), dtype=np.int64))
                continue
            flat = np.bincount(
                spike_trial_idx[valid] * n_bins_eff + spike_bin_idx[valid],
                minlength=n_trials * n_bins_eff,
            )
            spike_counts_trials_units.append(flat.reshape(n_trials, n_bins_eff))
        else:
            left_edges = np.asarray(bin_spec["left"], dtype=float)
            right_edges = np.asarray(bin_spec["right"], dtype=float)

            out = np.zeros((n_trials, n_bins_eff), dtype=np.int64)
            order = np.argsort(spike_trial_idx, kind="mergesort")
            trial_idx_sorted = spike_trial_idx[order]
            tuner_sorted_by_trial = spike_tuner[order]

            for t in range(n_trials):
                s = np.searchsorted(trial_idx_sorted, t, side="left")
                e = np.searchsorted(trial_idx_sorted, t, side="right")
                if e <= s:
                    continue
                vals = tuner_sorted_by_trial[s:e]
                out[t, :] = _counts_in_windows(vals, left_edges, right_edges)

            spike_counts_trials_units.append(out)

    spike_counts_trials_units = np.stack(spike_counts_trials_units, axis=0)  # (n_units, n_trials, n_bins)

    return dict(
        trial_ids=trial_ids,
        centers=centers,
        dt=dt,
        mode=mode,
        occupancy_time_trials=occupancy_time_trials,
        spike_counts_trials_units=spike_counts_trials_units,
    )


def _trial_labels_constant_within_trial(
    position_df: pd.DataFrame,
    *,
    trial_column: str,
    label_column: str,
    mask: np.ndarray,
) -> pd.Series:
    """
    Return trial_id -> label, checking that label is constant within each trial.
    """
    if label_column not in position_df.columns:
        raise KeyError(f"position_df missing label_column: {label_column!r}")

    sub = position_df.loc[mask, [trial_column, label_column]].dropna(subset=[trial_column])
    if sub.empty:
        raise ValueError("No rows remain after mask for extracting trial labels.")

    def _unique_or_bad(s: pd.Series) -> Any:
        vals = pd.unique(s.dropna())
        if len(vals) == 0:
            return np.nan
        if len(vals) > 1:
            return "__NON_CONSTANT__"
        return vals[0]

    labels_by_trial = sub.groupby(trial_column, sort=True)[label_column].apply(_unique_or_bad)
    if (labels_by_trial == "__NON_CONSTANT__").any():
        bad_trials = labels_by_trial.index[labels_by_trial == "__NON_CONSTANT__"].to_numpy()
        raise ValueError(
            f"label_column={label_column!r} is not constant within at least one trial. "
            f"Example bad trial ids: {bad_trials[:10]!r}"
        )
    return labels_by_trial


def _pooled_rates_for_trials(
    pre: Dict[str, Any],
    trial_indices: np.ndarray,
    *,
    min_occupancy_s: float,
    peak_normalize: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate per-trial counts into a pooled curve for each unit.

    Returns:
      rates_units_bins: (n_units, n_bins) rates (Hz) or peak-normalized rates
      occupancy_bins_s: (n_bins,) seconds
    """
    trial_indices = np.asarray(trial_indices, dtype=int)
    occ = np.asarray(pre["occupancy_time_trials"], dtype=float)[trial_indices, :].sum(axis=0)
    counts_trials_units = np.asarray(pre["spike_counts_trials_units"], dtype=float)
    counts = counts_trials_units[:, trial_indices, :].sum(axis=1)

    rates = np.full_like(counts, np.nan, dtype=float)
    valid = occ > float(min_occupancy_s)
    if np.any(valid):
        rates[:, valid] = counts[:, valid] / occ[valid]

    if peak_normalize:
        rates = _peak_normalize_rates_max(rates)
    return rates, occ


def _compute_metrics_and_stats(
    *,
    rates_a: np.ndarray,
    rates_b: np.ndarray,
    centers: np.ndarray,
    metric_specs: Sequence[CurveMetricSpec],
    min_bins: int,
) -> Tuple[np.ndarray, np.ndarray, Sequence[str]]:
    """
    Returns:
      metrics: (n_units, n_metrics)
      stats:   (n_units, n_metrics) where larger = more different
      names:   metric names
    """
    rates_a = np.asarray(rates_a, dtype=float)
    rates_b = np.asarray(rates_b, dtype=float)
    centers = np.asarray(centers, dtype=float)

    n_units = int(rates_a.shape[0])
    metric_specs = tuple(metric_specs)
    n_metrics = int(len(metric_specs))

    metrics = np.full((n_units, n_metrics), np.nan, dtype=float)
    stats = np.full((n_units, n_metrics), np.nan, dtype=float)

    for u in range(n_units):
        ca = rates_a[u, :]
        cb = rates_b[u, :]
        for mi, spec in enumerate(metric_specs):
            mv = spec.func(ca, cb, centers, int(min_bins))
            metrics[u, mi] = mv
            stats[u, mi] = spec.to_diff_stat(mv)

    return metrics, stats, [s.name for s in metric_specs]


def permutation_test_tuning_difference_trial_labels(
    position_df: pd.DataFrame,
    *,
    column: str,
    spikes_list: Sequence[np.ndarray],
    base_mask: np.ndarray,
    trial_column: str,
    label_column: str,
    label_a: Any,
    label_b: Any,
    # binning controls (fixed across permutations)
    binning: str = "sliding",
    n_bins: int = 8,
    tuner_bins: Optional[np.ndarray] = None,
    window_width: Optional[float] = None,
    window_step: Optional[float] = None,
    tuner_min: Optional[float] = None,
    tuner_max: Optional[float] = None,
    # test controls
    metric_specs: Optional[Sequence[CurveMetricSpec]] = None,
    min_bins: int = 3,
    min_occupancy_s: float = 0.25,
    peak_normalize: bool = True,
    n_perm: int = 2000,
    random_state: int = 0,
    strata_column: Optional[str] = None,
    units: Optional[Sequence[int]] = None,
    return_null: bool = False,
) -> Dict[str, Any]:
    """
    Unpaired permutation test for a 2-label trial-level comparison.

    Null distribution is built by shuffling which trials are assigned to A vs B
    (optionally within strata like epoch).

    Returns dict with DataFrames:
      - observed_metrics: per-unit metric values (corr, nrmse, peak shift, ...)
      - observed_stats: the transformed "difference stats" used for p-values
      - p_values: p-value per metric
      - p_max: omnibus p for max(stat across metrics) per unit
      - q_values, q_max: BH-FDR across units
    """
    if metric_specs is None:
        metric_specs = default_curve_metric_specs(use_peak_shift_x=True)

    df = position_df
    base_mask = np.asarray(base_mask, dtype=bool)
    base_mask = base_mask & df[trial_column].notna().to_numpy()

    labels_by_trial = _trial_labels_constant_within_trial(
        df, trial_column=trial_column, label_column=label_column, mask=base_mask
    )
    labels_by_trial = labels_by_trial[labels_by_trial.isin([label_a, label_b])]
    if labels_by_trial.empty:
        raise ValueError(f"No trials found with labels {label_a!r}/{label_b!r} under base_mask.")

    trial_ids_ab = labels_by_trial.index.to_numpy(dtype=int)
    is_a = (labels_by_trial.to_numpy() == label_a).astype(bool)

    if int(np.sum(is_a)) < 2 or int(np.sum(~is_a)) < 2:
        raise ValueError(f"Need >=2 trials in each group; got nA={int(np.sum(is_a))}, nB={int(np.sum(~is_a))}.")

    strata_per_trial = None
    if strata_column is not None:
        strata_by_trial = _trial_labels_constant_within_trial(
            df, trial_column=trial_column, label_column=strata_column, mask=base_mask
        )
        strata_per_trial = strata_by_trial.reindex(trial_ids_ab).to_numpy()

    pre_mask = base_mask & df[trial_column].isin(trial_ids_ab).to_numpy()
    pre = precompute_trial_binned_counts(
        df,
        column=column,
        spikes_list=spikes_list,
        mask=pre_mask,
        trial_column=trial_column,
        n_bins=n_bins,
        tuner_bins=tuner_bins,
        binning=binning,
        window_width=window_width,
        window_step=window_step,
        tuner_min=tuner_min,
        tuner_max=tuner_max,
    )

    tid_to_i = {int(tid): i for i, tid in enumerate(pre["trial_ids"])}
    keep = np.array([tid in tid_to_i for tid in trial_ids_ab], dtype=bool)
    trial_ids_ab = trial_ids_ab[keep]
    is_a = is_a[keep]
    if strata_per_trial is not None:
        strata_per_trial = np.asarray(strata_per_trial)[keep]
    trial_idx = np.array([tid_to_i[int(t)] for t in trial_ids_ab], dtype=int)

    # unit selection
    if units is not None:
        units = np.asarray(list(units), dtype=int)
        pre = dict(pre)
        pre["spike_counts_trials_units"] = np.asarray(pre["spike_counts_trials_units"])[units, :, :]
        unit_ids = units
    else:
        unit_ids = np.arange(np.asarray(pre["spike_counts_trials_units"]).shape[0], dtype=int)

    idx_a_obs = trial_idx[is_a]
    idx_b_obs = trial_idx[~is_a]
    rates_a_obs, occ_a_obs = _pooled_rates_for_trials(
        pre, idx_a_obs, min_occupancy_s=min_occupancy_s, peak_normalize=peak_normalize
    )
    rates_b_obs, occ_b_obs = _pooled_rates_for_trials(
        pre, idx_b_obs, min_occupancy_s=min_occupancy_s, peak_normalize=peak_normalize
    )

    metrics_obs, stats_obs, metric_names = _compute_metrics_and_stats(
        rates_a=rates_a_obs,
        rates_b=rates_b_obs,
        centers=np.asarray(pre["centers"]),
        metric_specs=metric_specs,
        min_bins=min_bins,
    )

    n_units = int(metrics_obs.shape[0])
    n_metrics = int(metrics_obs.shape[1])

    ge_counts = np.zeros((n_units, n_metrics), dtype=np.int64)
    denom_counts = np.zeros((n_units, n_metrics), dtype=np.int64)
    ge_counts_max = np.zeros(n_units, dtype=np.int64)
    denom_counts_max = np.zeros(n_units, dtype=np.int64)

    obs_max = np.nanmax(stats_obs, axis=1)

    rng = np.random.default_rng(int(random_state))
    base_group = is_a.astype(bool).copy()

    null_stats = None
    null_max = None
    if return_null:
        null_stats = np.full((n_units, n_metrics, int(n_perm)), np.nan, dtype=float)
        null_max = np.full((n_units, int(n_perm)), np.nan, dtype=float)

    if strata_per_trial is None:
        for r in range(int(n_perm)):
            perm = rng.permutation(base_group)
            idx_a = trial_idx[perm]
            idx_b = trial_idx[~perm]
            ra, _ = _pooled_rates_for_trials(pre, idx_a, min_occupancy_s=min_occupancy_s, peak_normalize=peak_normalize)
            rb, _ = _pooled_rates_for_trials(pre, idx_b, min_occupancy_s=min_occupancy_s, peak_normalize=peak_normalize)
            _, s, _ = _compute_metrics_and_stats(
                rates_a=ra, rates_b=rb, centers=np.asarray(pre["centers"]), metric_specs=metric_specs, min_bins=min_bins
            )

            if return_null:
                null_stats[:, :, r] = s
                null_max[:, r] = np.nanmax(s, axis=1)

            finite = np.isfinite(s) & np.isfinite(stats_obs)
            denom_counts += finite.astype(np.int64)
            ge_counts += (finite & (s >= stats_obs)).astype(np.int64)

            smax = np.nanmax(s, axis=1)
            finite_m = np.isfinite(smax) & np.isfinite(obs_max)
            denom_counts_max += finite_m.astype(np.int64)
            ge_counts_max += (finite_m & (smax >= obs_max)).astype(np.int64)
    else:
        strata = np.asarray(strata_per_trial)
        uniq = pd.unique(strata)
        idxs_by = [np.where(strata == u)[0] for u in uniq]
        for r in range(int(n_perm)):
            perm = base_group.copy()
            for idxs in idxs_by:
                perm[idxs] = perm[idxs][rng.permutation(idxs.size)]
            idx_a = trial_idx[perm]
            idx_b = trial_idx[~perm]
            ra, _ = _pooled_rates_for_trials(pre, idx_a, min_occupancy_s=min_occupancy_s, peak_normalize=peak_normalize)
            rb, _ = _pooled_rates_for_trials(pre, idx_b, min_occupancy_s=min_occupancy_s, peak_normalize=peak_normalize)
            _, s, _ = _compute_metrics_and_stats(
                rates_a=ra, rates_b=rb, centers=np.asarray(pre["centers"]), metric_specs=metric_specs, min_bins=min_bins
            )

            if return_null:
                null_stats[:, :, r] = s
                null_max[:, r] = np.nanmax(s, axis=1)

            finite = np.isfinite(s) & np.isfinite(stats_obs)
            denom_counts += finite.astype(np.int64)
            ge_counts += (finite & (s >= stats_obs)).astype(np.int64)

            smax = np.nanmax(s, axis=1)
            finite_m = np.isfinite(smax) & np.isfinite(obs_max)
            denom_counts_max += finite_m.astype(np.int64)
            ge_counts_max += (finite_m & (smax >= obs_max)).astype(np.int64)

    p = (ge_counts + 1.0) / (denom_counts + 1.0)
    p_max = (ge_counts_max + 1.0) / (denom_counts_max + 1.0)

    obs_df = pd.DataFrame(metrics_obs, index=unit_ids, columns=metric_names)
    obs_stat_df = pd.DataFrame(stats_obs, index=unit_ids, columns=[f"{n}_stat" for n in metric_names])
    p_df = pd.DataFrame(p, index=unit_ids, columns=[f"p_{n}" for n in metric_names])
    p_max_s = pd.Series(p_max, index=unit_ids, name="p_max")

    q_df = pd.DataFrame(index=unit_ids)
    for col in p_df.columns:
        _, q = fdr_bh(p_df[col].to_numpy(), alpha=0.05)
        q_df[col.replace("p_", "q_")] = q
    _, qmax = fdr_bh(p_max_s.to_numpy(), alpha=0.05)
    q_max_s = pd.Series(qmax, index=unit_ids, name="q_max")

    out: Dict[str, Any] = dict(
        precompute=pre,
        unit_ids=unit_ids,
        centers=np.asarray(pre["centers"]),
        label_a=label_a,
        label_b=label_b,
        n_trials_a=int(idx_a_obs.size),
        n_trials_b=int(idx_b_obs.size),
        observed_metrics=obs_df,
        observed_stats=obs_stat_df,
        p_values=p_df,
        q_values=q_df,
        p_max=p_max_s,
        q_max=q_max_s,
        valid_bins_mask_observed=(np.asarray(occ_a_obs) > float(min_occupancy_s))
        & (np.asarray(occ_b_obs) > float(min_occupancy_s))
        & np.isfinite(np.asarray(pre["centers"])),
    )
    if return_null:
        out["null_stats"] = null_stats
        out["null_max"] = null_max
    return out


def paired_permutation_test_tuning_difference_masks(
    position_df: pd.DataFrame,
    *,
    column: str,
    spikes_list: Sequence[np.ndarray],
    base_mask: np.ndarray,
    trial_column: str,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    label_a: str = "A",
    label_b: str = "B",
    # binning controls (fixed across both conditions and permutations)
    binning: str = "sliding",
    n_bins: int = 8,
    tuner_bins: Optional[np.ndarray] = None,
    window_width: Optional[float] = None,
    window_step: Optional[float] = None,
    tuner_min: Optional[float] = None,
    tuner_max: Optional[float] = None,
    # test controls
    metric_specs: Optional[Sequence[CurveMetricSpec]] = None,
    min_bins: int = 3,
    min_occupancy_s: float = 0.25,
    min_trial_occupancy_s: float = 0.0,
    peak_normalize: bool = True,
    n_perm: int = 2000,
    random_state: int = 0,
    units: Optional[Sequence[int]] = None,
    return_null: bool = False,
) -> Dict[str, Any]:
    """
    Paired permutation test for within-trial comparisons (e.g., early vs late segments).

    You supply two sample-level masks (len(position_df)) defining A vs B.
    The null is built by swapping A/B within each trial with p=0.5.
    """
    if metric_specs is None:
        metric_specs = default_curve_metric_specs(use_peak_shift_x=True)

    df = position_df
    base_mask = np.asarray(base_mask, dtype=bool)
    mask_a = np.asarray(mask_a, dtype=bool)
    mask_b = np.asarray(mask_b, dtype=bool)
    if base_mask.shape != mask_a.shape or base_mask.shape != mask_b.shape:
        raise ValueError("base_mask, mask_a, and mask_b must have same shape.")

    pre_a = precompute_trial_binned_counts(
        df,
        column=column,
        spikes_list=spikes_list,
        mask=(base_mask & mask_a),
        trial_column=trial_column,
        n_bins=n_bins,
        tuner_bins=tuner_bins,
        binning=binning,
        window_width=window_width,
        window_step=window_step,
        tuner_min=tuner_min,
        tuner_max=tuner_max,
    )
    pre_b = precompute_trial_binned_counts(
        df,
        column=column,
        spikes_list=spikes_list,
        mask=(base_mask & mask_b),
        trial_column=trial_column,
        n_bins=n_bins,
        tuner_bins=tuner_bins,
        binning=binning,
        window_width=window_width,
        window_step=window_step,
        tuner_min=tuner_min,
        tuner_max=tuner_max,
    )

    if not np.allclose(np.asarray(pre_a["centers"]), np.asarray(pre_b["centers"]), equal_nan=True):
        raise ValueError("Binning centers differ between A and B; ensure identical binning params.")

    trials_common = np.intersect1d(np.asarray(pre_a["trial_ids"]), np.asarray(pre_b["trial_ids"]))
    if trials_common.size < 2:
        raise ValueError(f"Need >=2 paired trials with data in both masks; found {int(trials_common.size)}.")

    ia = np.searchsorted(np.asarray(pre_a["trial_ids"]), trials_common)
    ib = np.searchsorted(np.asarray(pre_b["trial_ids"]), trials_common)

    occ_a_trials = np.asarray(pre_a["occupancy_time_trials"], dtype=float)[ia, :]
    occ_b_trials = np.asarray(pre_b["occupancy_time_trials"], dtype=float)[ib, :]

    keep = (occ_a_trials.sum(axis=1) >= float(min_trial_occupancy_s)) & (occ_b_trials.sum(axis=1) >= float(min_trial_occupancy_s))
    trials_common = trials_common[keep]
    ia = ia[keep]
    ib = ib[keep]
    occ_a_trials = occ_a_trials[keep, :]
    occ_b_trials = occ_b_trials[keep, :]
    if trials_common.size < 2:
        raise ValueError("Need >=2 paired trials after min_trial_occupancy_s filter.")

    sc_a_all = np.asarray(pre_a["spike_counts_trials_units"], dtype=float)[:, ia, :]
    sc_b_all = np.asarray(pre_b["spike_counts_trials_units"], dtype=float)[:, ib, :]

    if units is not None:
        units = np.asarray(list(units), dtype=int)
        sc_a_all = sc_a_all[units, :, :]
        sc_b_all = sc_b_all[units, :, :]
        unit_ids = units
    else:
        unit_ids = np.arange(sc_a_all.shape[0], dtype=int)

    # observed pooled curves
    occ_a_obs = occ_a_trials.sum(axis=0)
    occ_b_obs = occ_b_trials.sum(axis=0)
    counts_a_obs = sc_a_all.sum(axis=1)
    counts_b_obs = sc_b_all.sum(axis=1)

    rates_a_obs = np.full_like(counts_a_obs, np.nan, dtype=float)
    rates_b_obs = np.full_like(counts_b_obs, np.nan, dtype=float)
    valid_a = occ_a_obs > float(min_occupancy_s)
    valid_b = occ_b_obs > float(min_occupancy_s)
    if np.any(valid_a):
        rates_a_obs[:, valid_a] = counts_a_obs[:, valid_a] / occ_a_obs[valid_a]
    if np.any(valid_b):
        rates_b_obs[:, valid_b] = counts_b_obs[:, valid_b] / occ_b_obs[valid_b]
    if peak_normalize:
        rates_a_obs = _peak_normalize_rates_max(rates_a_obs)
        rates_b_obs = _peak_normalize_rates_max(rates_b_obs)

    metrics_obs, stats_obs, metric_names = _compute_metrics_and_stats(
        rates_a=rates_a_obs,
        rates_b=rates_b_obs,
        centers=np.asarray(pre_a["centers"]),
        metric_specs=metric_specs,
        min_bins=min_bins,
    )

    n_units = int(metrics_obs.shape[0])
    n_metrics = int(metrics_obs.shape[1])

    ge_counts = np.zeros((n_units, n_metrics), dtype=np.int64)
    denom_counts = np.zeros((n_units, n_metrics), dtype=np.int64)
    ge_counts_max = np.zeros(n_units, dtype=np.int64)
    denom_counts_max = np.zeros(n_units, dtype=np.int64)

    obs_max = np.nanmax(stats_obs, axis=1)

    rng = np.random.default_rng(int(random_state))
    n_trials = int(trials_common.size)

    null_stats = None
    null_max = None
    if return_null:
        null_stats = np.full((n_units, n_metrics, int(n_perm)), np.nan, dtype=float)
        null_max = np.full((n_units, int(n_perm)), np.nan, dtype=float)

    for r in range(int(n_perm)):
        swap = (rng.random(n_trials) < 0.5).astype(float)  # (n_trials,)
        w = swap[:, None]  # (n_trials, 1)

        occ_a = ((1.0 - w) * occ_a_trials + w * occ_b_trials).sum(axis=0)
        occ_b = ((1.0 - w) * occ_b_trials + w * occ_a_trials).sum(axis=0)

        w3 = swap[None, :, None]  # (1, n_trials, 1)
        counts_a = ((1.0 - w3) * sc_a_all + w3 * sc_b_all).sum(axis=1)
        counts_b = ((1.0 - w3) * sc_b_all + w3 * sc_a_all).sum(axis=1)

        ra = np.full_like(counts_a, np.nan, dtype=float)
        rb = np.full_like(counts_b, np.nan, dtype=float)
        va = occ_a > float(min_occupancy_s)
        vb = occ_b > float(min_occupancy_s)
        if np.any(va):
            ra[:, va] = counts_a[:, va] / occ_a[va]
        if np.any(vb):
            rb[:, vb] = counts_b[:, vb] / occ_b[vb]
        if peak_normalize:
            ra = _peak_normalize_rates_max(ra)
            rb = _peak_normalize_rates_max(rb)

        _, s, _ = _compute_metrics_and_stats(
            rates_a=ra, rates_b=rb, centers=np.asarray(pre_a["centers"]), metric_specs=metric_specs, min_bins=min_bins
        )

        if return_null:
            null_stats[:, :, r] = s
            null_max[:, r] = np.nanmax(s, axis=1)

        finite = np.isfinite(s) & np.isfinite(stats_obs)
        denom_counts += finite.astype(np.int64)
        ge_counts += (finite & (s >= stats_obs)).astype(np.int64)

        smax = np.nanmax(s, axis=1)
        finite_m = np.isfinite(smax) & np.isfinite(obs_max)
        denom_counts_max += finite_m.astype(np.int64)
        ge_counts_max += (finite_m & (smax >= obs_max)).astype(np.int64)

    p = (ge_counts + 1.0) / (denom_counts + 1.0)
    p_max = (ge_counts_max + 1.0) / (denom_counts_max + 1.0)

    obs_df = pd.DataFrame(metrics_obs, index=unit_ids, columns=metric_names)
    obs_stat_df = pd.DataFrame(stats_obs, index=unit_ids, columns=[f"{n}_stat" for n in metric_names])
    p_df = pd.DataFrame(p, index=unit_ids, columns=[f"p_{n}" for n in metric_names])
    p_max_s = pd.Series(p_max, index=unit_ids, name="p_max")

    q_df = pd.DataFrame(index=unit_ids)
    for col in p_df.columns:
        _, q = fdr_bh(p_df[col].to_numpy(), alpha=0.05)
        q_df[col.replace("p_", "q_")] = q
    _, qmax = fdr_bh(p_max_s.to_numpy(), alpha=0.05)
    q_max_s = pd.Series(qmax, index=unit_ids, name="q_max")

    out: Dict[str, Any] = dict(
        precompute_a=pre_a,
        precompute_b=pre_b,
        unit_ids=unit_ids,
        centers=np.asarray(pre_a["centers"]),
        label_a=label_a,
        label_b=label_b,
        n_trials_paired=int(trials_common.size),
        paired_trial_ids=trials_common,
        observed_metrics=obs_df,
        observed_stats=obs_stat_df,
        p_values=p_df,
        q_values=q_df,
        p_max=p_max_s,
        q_max=q_max_s,
        valid_bins_mask_observed=(np.asarray(occ_a_obs) > float(min_occupancy_s))
        & (np.asarray(occ_b_obs) > float(min_occupancy_s))
        & np.isfinite(np.asarray(pre_a["centers"])),
    )
    if return_null:
        out["null_stats"] = null_stats
        out["null_max"] = null_max
    return out


def plot_tuning_grid_bootstrap(
    tuner_tuning: dict,
    lower_ci: dict,
    upper_ci: dict,
    tuner_bin_centers: np.ndarray,
    column: str,
    n_units: int = -1,
    label=None,
    indices = None,
    s: Optional[float] = None,
    color: Optional[str] = None,
    ci: float = 0.95,
    stderr: Optional[dict] = None,
    show_stderr: bool = False,
    peak_normalize: bool = False,
    ylims: tuple = (0,1),
    linewidth: float = 1.0,
):
    # choose units to plot 
    all_units = list(tuner_tuning.keys())
    
    if indices == None:
        if n_units < 0 or n_units > len(all_units):
            units = all_units
        else:
            units = all_units[:n_units]
    else:
        units = [all_units[idx] for idx in indices]
        units = units[:n_units]        
    
    
    n_units = len(units)

    n_cols = 5
    n_rows = int(np.ceil(n_units / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2 * n_rows),
        sharex=True, sharey=True
    )
    axes = np.array(axes).reshape(-1)

    for i, unit in enumerate(units):
        rate = np.asarray(tuner_tuning[unit], dtype=float)
        lo = np.asarray(lower_ci[unit], dtype=float)
        hi = np.asarray(upper_ci[unit], dtype=float)

        unit_scale = None
        if peak_normalize:
            rate, lo_opt, hi, unit_scale = _peak_normalize_from_upper_ci(rate, lo, hi)
            lo = lo_opt if lo_opt is not None else lo

        ax = axes[i]
        plot_kwargs = {"marker": "o", "linewidth": linewidth}
        if s is not None:
            plot_kwargs["markersize"] = s
        if color is not None:
            plot_kwargs["color"] = color

        line = ax.plot(tuner_bin_centers, rate, **plot_kwargs)[0]
        fill_color = color if color is not None else line.get_color()

        # Always show the bootstrap percentile CI band
        ax.fill_between(
            tuner_bin_centers,
            lo,
            hi,
            color=fill_color,
            alpha=0.3,
            label=f"{int(round(100 * float(ci)))}% CI",
        )

        # Note: we intentionally do not draw error bars on top of the CI band.
        # Keep `stderr`/`show_stderr` args for backward compatibility.
        _ = stderr
        _ = show_stderr

        ax.set_title(str(unit), fontsize=8)
        ax.axhline(0, color="k", linewidth=0.5)
        ax.tick_params(axis="both", labelsize=6)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle(f"{column} tuning curves {label}", fontsize=14)
    fig.text(0.5, 0.04, f"{column}", ha="center")
    ylab = "Normalized firing rate" if peak_normalize else "Firing rate (Hz)"
    fig.text(0.04, 0.5, ylab, va="center", rotation="vertical")
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    if peak_normalize:
        plt.ylim([-0.05, 1.05])
    else:
        plt.ylim([-5, 20])
    plt.show()




def plot_tuning_grid(tuner_tuning: dict,
                           column: str,
                           position_df: pd.DataFrame,
                           spikes_list: list,
                           mask: np.ndarray = None,
                           n_units: int = -1,
                           label = None,
                           tuner_bins: np.ndarray = None,
                           binning: str = "edges",
                           window_width: Optional[float] = None,
                           window_step: Optional[float] = None,
                           tuner_min: Optional[float] = None,
                           tuner_max: Optional[float] = None,
                           s: Optional[float] = None,
                           color: Optional[str] = None,
                           peak_normalize: bool = False,
                           linewidth: float = 1.0):
    """
    Plot per-unit tuner tuning curves with 95% CI.

    position_df: full tuner_trials_merged_df (NOT pre-masked)
    mask: boolean array aligned with position_df.index (e.g. zone=='run' & trial_type=='inbound')
    """

    # --- prepare timebase + mask on the FULL df ---
    timestamps = position_df.index.to_numpy()
    tuner = position_df[column].to_numpy()

    if mask is None:
        mask = np.ones_like(tuner, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != tuner.shape[0]:
            raise ValueError("mask must have same length as position_df")

    # infer number of bins from the tuning dict
    first_unit = next(iter(tuner_tuning))
    n_bins = len(tuner_tuning[first_unit])

    # occupancy only where mask is True
    tuner_masked = tuner[mask]
    dt = np.median(np.diff(timestamps))
    bin_spec = _prepare_tuner_binning(
        tuner_masked,
        n_bins=n_bins,
        tuner_bins=tuner_bins,
        binning=binning,
        window_width=window_width,
        window_step=window_step,
        tuner_min=tuner_min,
        tuner_max=tuner_max,
    )
    tuner_bin_centers = bin_spec["centers"]
    if tuner_bin_centers.size != n_bins:
        raise ValueError(
            f"Binning parameters produce {tuner_bin_centers.size} bins, but tuner_tuning has {n_bins}. "
            "Pass the same binning parameters used to compute tuner_tuning."
        )

    if bin_spec["mode"] == "edges":
        occupancy_counts, _ = np.histogram(tuner_masked, bins=bin_spec["edges"])
    else:
        occupancy_counts = _counts_in_windows(tuner_masked, bin_spec["left"], bin_spec["right"])
    occupancy_time = occupancy_counts * dt  # seconds in each tuner bin/window

    # --- helper: spike counts per tuner bin, using the same mask ---
    def _spike_counts_per_tuner_bin(spike_times):
        tuner_at_spikes = spikes_to_tuner(spike_times, timestamps, tuner, mask=mask)
        if bin_spec["mode"] == "edges":
            spike_counts, _ = np.histogram(tuner_at_spikes, bins=bin_spec["edges"])
        else:
            spike_counts = _counts_in_windows(tuner_at_spikes, bin_spec["left"], bin_spec["right"])
        return spike_counts

    # --- choose units to plot ---
    all_units = list(tuner_tuning.keys())
    if n_units < 0 or n_units > len(all_units):
        units = all_units
    else:
        units = all_units[:n_units]
    n_units = len(units)

    n_cols = 5
    n_rows = int(np.ceil(n_units / n_cols))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(2.5 * n_cols, 2 * n_rows),
        sharex=True, sharey=True
    )
    axes = np.array(axes).reshape(-1)

    # --- per-unit curves + CIs ---
    for i, unit in enumerate(units):
        spike_times = spikes_list[unit]
        spike_counts = _spike_counts_per_tuner_bin(spike_times)

        valid = occupancy_time > MIN_OCCUPANCY

        rate = np.full_like(occupancy_time, np.nan, dtype=float)
        se_rate = np.full_like(occupancy_time, np.nan, dtype=float)

        rate[valid] = spike_counts[valid] / occupancy_time[valid]
        se_rate[valid] = np.sqrt(spike_counts[valid]) / occupancy_time[valid]

        z = 1.96  # ~95% CI
        lower = np.clip(rate - z * se_rate, 0, None)
        upper = rate + z * se_rate

        if peak_normalize:
            rate, lower_opt, upper, _scale = _peak_normalize_from_upper_ci(rate, lower, upper)
            lower = lower_opt if lower_opt is not None else lower

        ax = axes[i]
        plot_kwargs = {"marker": "o", "linewidth": linewidth}
        if s is not None:
            plot_kwargs["markersize"] = s
        if color is not None:
            plot_kwargs["color"] = color

        line = ax.plot(tuner_bin_centers, rate, **plot_kwargs)[0]
        fill_color = color if color is not None else line.get_color()
        ax.fill_between(
            tuner_bin_centers,
            lower,
            upper,
            color=fill_color,
            alpha=0.3,
            label="95% CI",
        )

        ax.set_title(str(unit), fontsize=8)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=6)

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'{column} tuning curves {label}', fontsize=14)
    fig.text(0.5, 0.04, f'{column}', ha='center')
    ylab = "Normalized firing rate" if peak_normalize else "Firing rate (Hz)"
    fig.text(0.04, 0.5, ylab, va='center', rotation='vertical')
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    if peak_normalize:
        plt.ylim([-0.05, 1.05])
    else:
        plt.ylim(-10, 20)
    plt.show()
    
    
    


def classify_tuning_shapes_bootstraps(
    slope_boot,
    curvature_boot,
    *,
    bin_centers: Optional[np.ndarray] = None,
    x_min=None,
    x_max=None,
    peak_loc_boot=None,
    ci=0.95,
    slope_eps=0.01,
    curvature_eps=0.005,
    peak_margin=0.0,
    min_valid_boot=50,
    prefer_peak_for_bell=True,
    normalize_x: bool = False,
):
    """
    Improved bootstrap shape classification.

    Inputs
    - slope_boot[unit]: bootstrap slopes (currently from linear fit; later you may replace with quadratic b or derivative)
    - curvature_boot[unit]: bootstrap quadratic curvatures (q)
    - peak_loc_boot[unit] (optional): bootstrap peak locations x_peak = -b/(2q)
    - x_min/x_max: min/max of speed bin centers (only needed if peak_loc_boot is provided)
    """

    # If bin centers are provided (discrete bins or sliding windows), use them to infer x range
    # so downstream logic can reason in the correct x-units even when binning becomes "continuous".
    inferred_x_min = None
    inferred_x_max = None
    inferred_x_range = None
    if bin_centers is not None:
        bc = np.asarray(bin_centers, dtype=float).ravel()
        bc = bc[np.isfinite(bc)]
        if bc.size >= 2:
            inferred_x_min = float(np.min(bc))
            inferred_x_max = float(np.max(bc))
            inferred_x_range = float(inferred_x_max - inferred_x_min)

    if x_min is None and inferred_x_min is not None:
        x_min = inferred_x_min
    if x_max is None and inferred_x_max is not None:
        x_max = inferred_x_max

    x_range = None
    if (x_min is not None) and (x_max is not None):
        try:
            x_range = float(x_max) - float(x_min)
        except Exception:
            x_range = None
    if x_range is not None and (not np.isfinite(x_range) or x_range <= 0):
        x_range = None

    alpha = 1.0 - float(ci)
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)

    units = list(slope_boot.keys()) if isinstance(slope_boot, dict) else list(range(np.asarray(slope_boot).shape[0]))
    shape = {}
    stats = {}

    for u in units:
        s = np.asarray(slope_boot[u], float)
        q = np.asarray(curvature_boot[u], float)

        s = s[np.isfinite(s)]
        q = q[np.isfinite(q)]

        # Optional normalization: convert slope/curvature to an x-normalized coordinate system so
        # thresholds don't depend on the units/range of x.
        #
        # If x' = (x - x_min) / x_range, then:
        #   slope'     = slope * x_range
        #   curvature' = curvature * x_range^2
        if normalize_x and x_range is not None:
            s = s * x_range
            q = q * (x_range ** 2)

        if (s.size < min_valid_boot) or (q.size < min_valid_boot):
            shape[u] = "insufficient_boot"
            stats[u] = {}
            continue

        s_lo, s_hi = np.percentile(s, [lo_q, hi_q])
        q_lo, q_hi = np.percentile(q, [lo_q, hi_q])

        p_inc = float(np.mean(s > slope_eps))
        p_dec = float(np.mean(s < -slope_eps))
        p_bell_curv = float(np.mean(q < -curvature_eps))
        p_u_curv = float(np.mean(q > curvature_eps))

        # Optional: use interior peak criterion if available
        p_bell_peak = np.nan
        p_u_peak = np.nan
        if peak_loc_boot is not None and x_min is not None and x_max is not None:
            xp = np.asarray(peak_loc_boot[u], float)
            xp = xp[np.isfinite(xp)]
            if xp.size >= min_valid_boot:
                interior = (xp > (x_min + peak_margin)) & (xp < (x_max - peak_margin))
                p_bell_peak = float(np.mean(interior & (q[:xp.size] < -curvature_eps)))
                p_u_peak = float(np.mean(interior & (q[:xp.size] > curvature_eps)))

        # Decision logic:
        # - If we have peak info, let bell/U win when curvature sign is stable AND peak is interior.
        # - Otherwise, call bell/U only when curvature is stable AND slope is NOT strongly monotonic.
        if peak_loc_boot is not None and prefer_peak_for_bell and np.isfinite(p_bell_peak):
            if q_hi < -curvature_eps and p_bell_peak > 0.8:
                label = "bell"
            elif q_lo > curvature_eps and p_u_peak > 0.8:
                label = "U"
            elif s_lo > slope_eps:
                label = "increasing"
            elif s_hi < -slope_eps:
                label = "decreasing"
            else:
                label = "flat/complex"
        else:
            # no peak info: be conservative about bell/U
            if s_lo > slope_eps:
                label = "increasing"
            elif s_hi < -slope_eps:
                label = "decreasing"
            elif (q_hi < -curvature_eps) and not (s_lo > slope_eps or s_hi < -slope_eps):
                label = "bell"
            elif (q_lo > curvature_eps) and not (s_lo > slope_eps or s_hi < -slope_eps):
                label = "U"
            else:
                label = "flat/complex"

        shape[u] = label
        stats[u] = {
            "slope_ci": (float(s_lo), float(s_hi)),
            "curvature_ci": (float(q_lo), float(q_hi)),
            "p_increasing": p_inc,
            "p_decreasing": p_dec,
            "p_bell_curv": p_bell_curv,
            "p_u_curv": p_u_curv,
            "p_bell_peak": p_bell_peak,
            "p_u_peak": p_u_peak,
            "x_min": None if x_min is None else float(x_min),
            "x_max": None if x_max is None else float(x_max),
            "x_range": None if x_range is None else float(x_range),
            "normalize_x": bool(normalize_x),
        }

    return shape, stats


# -----------------------------------------------------------------------------
# Backwards-compatibility aliases
# -----------------------------------------------------------------------------

def classify_tuning_shapes_from_bootstraps(
    slope_boot,
    curvature_boot,
    *,
    bin_centers: Optional[np.ndarray] = None,
    ci=0.95,
    min_valid_boot=50,
    slope_eps=0.0,
    curvature_eps=0.0,
    normalize_x: bool = False,
    **kwargs,
):
    """
    Backwards-compatible alias for `classify_tuning_shapes_bootstraps`.

    This name is used in older notebooks (e.g. `unit_tuning.ipynb`).
    """
    return classify_tuning_shapes_bootstraps(
        slope_boot,
        curvature_boot,
        bin_centers=bin_centers,
        ci=ci,
        min_valid_boot=min_valid_boot,
        slope_eps=slope_eps,
        curvature_eps=curvature_eps,
        normalize_x=normalize_x,
        **kwargs,
    )


# -----------------------------------------------------------------------------
# FigURL helpers for tuning-curve comparison
# -----------------------------------------------------------------------------

def _get_unit_curve(curves, unit_id: int):
    """
    Support passing tuning curves as:
      - dict[unit_id] -> 1D array
      - list/tuple indexed by unit_id
      - np.ndarray with shape (n_units, n_bins)
    """
    if curves is None:
        return None
    if isinstance(curves, dict):
        return curves.get(unit_id, None)
    if isinstance(curves, (list, tuple)):
        if unit_id < 0 or unit_id >= len(curves):
            return None
        return curves[unit_id]
    arr = np.asarray(curves)
    if arr.ndim == 2:
        if unit_id < 0 or unit_id >= arr.shape[0]:
            return None
        return arr[unit_id, :]
    raise TypeError(f"Unsupported curves container type: {type(curves)}")


def select_units_by_shape(
    shape_by_unit: dict,
    *,
    include_shapes,
    unit_ids=None,
):
    """
    Convenience selector for unit IDs based on `shape_by_unit`.

    Parameters
    ----------
    shape_by_unit : dict[int, str]
        Mapping from unit id -> shape label (e.g. 'bell', 'increasing', ...).
    include_shapes : str | list[str] | tuple[str, ...] | set[str]
        Shape(s) to include.
    unit_ids : list[int] | None
        Optional candidate units to filter; defaults to all keys in shape_by_unit.
    """
    if isinstance(include_shapes, str):
        include = {include_shapes}
    else:
        include = set(include_shapes)

    if unit_ids is None:
        unit_ids = list(shape_by_unit.keys())

    return [u for u in unit_ids if shape_by_unit.get(u) in include]



def compute_speed_tuning_progress_balanced_bootstrap_trials(
    position_df: pd.DataFrame,
    *,
    speed_col: str = "speed",
    progress_col: str = "trial_progress",
    spikes_list: list,
    n_speed_bins: int = 8,
    n_progress_bins: int = 5,
    mask: np.ndarray = None,
    trial_column: str = "trial_number",
    n_boot: int = 500,
    ci: float = 0.95,
    random_state: int = 0,
    progress_weights: str = "uniform",
    speed_bins: np.ndarray = None,         
    progress_bins: np.ndarray = None,       
    min_occupancy_cell_s: float = 0.25,    
    min_occupancy_speed_s: float = 1.0,     
    speed_binning: str = "edges",
    speed_window_width: Optional[float] = None,
    speed_window_step: Optional[float] = None,
    speed_min: Optional[float] = None,
    speed_max: Optional[float] = None,
    peak_normalize: bool = False,
):
    """
    Progress-balanced speed tuning with trial bootstrap.

    Concept (plain English)
    -----------------------
    1) Bin the data into a 2D grid: (speed_bin i, progress_bin j).
    2) For each cell (i,j), compute a cell-wise firing rate:
         R[i,j] = spikes[i,j] / time[i,j]
       (only if time[i,j] is large enough).
    3) For each speed bin i, combine the cell-wise rates across progress bins using
       target progress weights w[j] (e.g. uniform), renormalized over available cells:
         rate[i] = sum_j w[j]*R[i,j] / sum_j w[j]   (over valid cells only)
    4) Use trial bootstrap (resample trials with replacement) to get confidence intervals
       and bootstrap distributions of slope/curvature.

    Returns (mirrors compute_tuning_bootstrap_trials)
    -------------------------------------------------
    speed_tuning : dict[int, np.ndarray]  (n_speed_bins,)
    speed_centers : np.ndarray           (n_speed_bins,)
    lower_ci : dict[int, np.ndarray]     (n_speed_bins,)
    upper_ci : dict[int, np.ndarray]     (n_speed_bins,)
    slope_boot : dict[int, np.ndarray]   (n_boot,)
    curvature_boot : dict[int, np.ndarray] (n_boot,)

    peak_normalize
    --------------
    If True, divide each unit's pooled balanced curve, CI curves, and (slope/curvature) bootstrap
    statistics by a per-unit scale defined as `max_x upper_CI(x)` for that unit.
    """

    # ----------------------------
    # Input checks / setup
    # ----------------------------
    if trial_column not in position_df.columns:
        raise KeyError(f"position_df missing required column: {trial_column!r}")
    if speed_col not in position_df.columns:
        raise KeyError(f"position_df missing required column: {speed_col!r}")
    if progress_col not in position_df.columns:
        raise KeyError(f"position_df missing required column: {progress_col!r}")

    if len(position_df.index) < 2:
        raise ValueError("position_df must have at least 2 timestamped samples")

    if not pd.Index(position_df.index).is_monotonic_increasing:
        raise ValueError("position_df.index must be sorted (call position_df = position_df.sort_index()).")

    timestamps = position_df.index.to_numpy()
    speed = position_df[speed_col].to_numpy()
    progress = position_df[progress_col].to_numpy()
    trial_ids_raw = position_df[trial_column].to_numpy()

    if mask is None:
        mask = np.ones_like(speed, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != speed.shape[0]:
            raise ValueError("mask must have same length as position_df")

    # require valid trial id, finite speed, finite progress
    trial_ok = ~pd.isna(trial_ids_raw)
    mask = mask & trial_ok & np.isfinite(speed) & np.isfinite(progress)

    if not np.any(mask):
        raise ValueError("Mask leaves zero usable samples (after requiring trial_ok and finite speed/progress).")

    # approximate dt (works if timestamps are near-uniform)
    dt = float(np.median(np.diff(timestamps)))
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Non-finite or non-positive dt from timestamps; check position_df.index units and sorting.")

    # ----------------------------
    # Binning (speed + progress)
    # ----------------------------
    speed_mode = _validate_binning_mode(speed_binning)

    # speed bins default: [0, max_speed] (matches legacy behavior)
    speed_masked = speed[mask]
    max_speed = float(np.nanmax(speed_masked))
    if not np.isfinite(max_speed):
        raise ValueError("Non-finite max_speed under mask; cannot bin.")

    if speed_mode == "edges":
        if speed_bins is None:
            speed_bins = np.linspace(0.0, max_speed, int(n_speed_bins) + 1)
        else:
            speed_bins = np.asarray(speed_bins, dtype=float).ravel()
            if speed_bins.ndim != 1:
                raise ValueError("speed_bins must be 1D")
            if not np.all(np.diff(speed_bins) > 0):
                raise ValueError("speed_bins must be strictly increasing")
            n_speed_bins = len(speed_bins) - 1
            if n_speed_bins < 1:
                raise ValueError("speed_bins must define at least 1 bin")

        speed_centers = (speed_bins[:-1] + speed_bins[1:]) / 2.0
        speed_bin_spec = {"mode": "edges", "edges": speed_bins, "centers": speed_centers}
    else:
        if speed_bins is not None:
            raise ValueError("speed_bins is only used for speed_binning='edges'.")

        # Default min/max mirror legacy [0, max_speed] rather than [min(speed), max(speed)]
        if speed_min is None:
            speed_min = 0.0
        if speed_max is None:
            speed_max = max_speed

        speed_bin_spec = _prepare_tuner_binning(
            speed_masked,
            n_bins=n_speed_bins,
            tuner_bins=None,
            binning="sliding",
            window_width=speed_window_width,
            window_step=speed_window_step,
            tuner_min=speed_min,
            tuner_max=speed_max,
        )
        speed_centers = np.asarray(speed_bin_spec["centers"], dtype=float)
        n_speed_bins = int(speed_centers.size)

    # progress bins default: [0,1] uniform bins
    # (optionally user-supplied progress_bins)
    if progress_bins is None:
        prog = np.clip(progress, 0.0, 1.0)
        progress_bins = np.linspace(0.0, 1.0, n_progress_bins + 1)
    else:
        progress_bins = np.asarray(progress_bins, dtype=float).ravel()
        if progress_bins.ndim != 1:
            raise ValueError("progress_bins must be 1D")
        if not np.all(np.diff(progress_bins) > 0):
            raise ValueError("progress_bins must be strictly increasing")
        n_progress_bins = len(progress_bins) - 1
        if n_progress_bins < 1:
            raise ValueError("progress_bins must define at least 1 bin")
        prog = progress  # no clipping; caller is responsible for bin choice

    # ----------------------------
    # Trials under mask
    # ----------------------------
    trial_ids_masked = np.asarray(trial_ids_raw[mask]).astype(int)
    trial_ids = np.unique(trial_ids_masked)
    trial_ids = np.sort(trial_ids)
    n_trials = len(trial_ids)
    if n_trials < 2:
        raise ValueError(f"Need >=2 trials for trial bootstrap; found {n_trials}.")

    # Map masked samples -> trial indices 0..n_trials-1
    trial_idx_per_sample_masked = np.searchsorted(trial_ids, trial_ids_masked)

    # ----------------------------
    # Progress bin index for *all* samples (then we mask)
    # ----------------------------
    prog_idx = np.searchsorted(progress_bins, prog, side="right") - 1
    prog_idx[prog_idx == n_progress_bins] = n_progress_bins - 1

    valid = mask & (prog_idx >= 0) & (prog_idx < n_progress_bins)

    if speed_mode == "edges":
        # Single-bin assignment (legacy)
        speed_idx = np.searchsorted(speed_bins, speed, side="right") - 1
        speed_idx[speed_idx == n_speed_bins] = n_speed_bins - 1
        valid = valid & (speed_idx >= 0) & (speed_idx < n_speed_bins)
    else:
        # For sliding windows, we still enforce a global speed range so "occupancy" matches the window domain.
        speed_left = float(np.min(speed_bin_spec["left"]))
        speed_right = float(np.max(speed_bin_spec["right"]))
        valid = valid & np.isfinite(speed) & (speed >= speed_left) & (speed < speed_right)

    if not np.any(valid):
        raise ValueError("No samples remain after bin-index validity checks; check bins and mask.")

    # ----------------------------
    # Occupancy time per (trial, speed, progress)
    # ----------------------------
    # We build this using ONLY masked samples, because trial_idx_per_sample is only defined under mask.
    # So we must index with [mask] consistently.
    valid_under_mask = valid[mask]  # boolean array aligned to masked samples

    if speed_mode == "edges":
        flat_index_occ = (
            trial_idx_per_sample_masked[valid_under_mask] * (n_speed_bins * n_progress_bins)
            + speed_idx[mask][valid_under_mask] * n_progress_bins
            + prog_idx[mask][valid_under_mask]
        )

        occ_counts_flat = np.bincount(
            flat_index_occ,
            minlength=n_trials * n_speed_bins * n_progress_bins,
        )
        occ_counts_trials_2d = occ_counts_flat.reshape(n_trials, n_speed_bins, n_progress_bins)
        occ_time_trials_2d = occ_counts_trials_2d * dt  # seconds
    else:
        # Sliding windows: count samples per window within each (trial, progress) group.
        occ_counts_trials_2d = np.zeros((n_trials, n_speed_bins, n_progress_bins), dtype=np.int64)

        trial_idx_m = trial_idx_per_sample_masked[valid_under_mask]
        prog_idx_m = prog_idx[mask][valid_under_mask].astype(int)
        speed_m = speed[mask][valid_under_mask].astype(float)

        group_id = trial_idx_m * n_progress_bins + prog_idx_m
        order = np.argsort(group_id, kind="mergesort")
        group_id = group_id[order]
        speed_m = speed_m[order]

        unique_groups, start_idx = np.unique(group_id, return_index=True)
        start_idx = np.append(start_idx, speed_m.size)

        left_edges = speed_bin_spec["left"]
        right_edges = speed_bin_spec["right"]

        for k, gid in enumerate(unique_groups):
            s = start_idx[k]
            e = start_idx[k + 1]
            if e <= s:
                continue
            t = int(gid // n_progress_bins)
            p = int(gid % n_progress_bins)
            occ_counts_trials_2d[t, :, p] = _counts_in_windows(speed_m[s:e], left_edges, right_edges)

        occ_time_trials_2d = occ_counts_trials_2d * dt  # seconds

    # pooled occupancy (speed, progress)
    occ_time_total_2d = occ_time_trials_2d.sum(axis=0)

    # ----------------------------
    # Progress weights w[j]
    # ----------------------------
    if progress_weights == "uniform":
        w = np.ones(n_progress_bins, dtype=float) / n_progress_bins
    elif progress_weights == "global":
        # global progress occupancy under mask (diagnostic; less aggressive than uniform)
        global_prog_counts, _ = np.histogram(prog[mask], bins=progress_bins)
        if global_prog_counts.sum() == 0:
            raise ValueError("No global progress occupancy under mask.")
        w = global_prog_counts.astype(float) / global_prog_counts.sum()
    else:
        raise ValueError("progress_weights must be 'uniform' or 'global'")

    # ----------------------------
    # Trial bootstrap weights
    # ----------------------------
    rng = np.random.default_rng(random_state)
    boot_weights = rng.multinomial(n_trials, np.ones(n_trials) / n_trials, size=n_boot).astype(float)

    # ----------------------------
    # Spike counts per (trial, speed, progress) for each unit
    # ----------------------------
    def _spike_counts_trials_2d_for_unit(spike_times: np.ndarray) -> np.ndarray:
        spike_times = np.asarray(spike_times)

        t0, t1 = timestamps[0], timestamps[-1]
        in_window = (spike_times >= t0) & (spike_times <= t1)
        spike_times_win = spike_times[in_window]
        if spike_times_win.size == 0:
            return np.zeros((n_trials, n_speed_bins, n_progress_bins), dtype=np.int64)

        spike_pos_idx = np.searchsorted(timestamps, spike_times_win, side="right") - 1
        spike_pos_idx = np.clip(spike_pos_idx, 0, len(timestamps) - 1)

        # keep spikes whose mapped position sample is valid
        keep = valid[spike_pos_idx]
        if not np.any(keep):
            return np.zeros((n_trials, n_speed_bins, n_progress_bins), dtype=np.int64)
        spike_pos_idx = spike_pos_idx[keep]

        spike_trials = trial_ids_raw[spike_pos_idx]
        p_idx = prog_idx[spike_pos_idx]

        ok = (~pd.isna(spike_trials)) & (p_idx >= 0) & (p_idx < n_progress_bins)
        if speed_mode == "edges":
            s_idx = speed_idx[spike_pos_idx]
            ok = ok & (s_idx >= 0) & (s_idx < n_speed_bins)

        if not np.any(ok):
            return np.zeros((n_trials, n_speed_bins, n_progress_bins), dtype=np.int64)

        spike_trials = spike_trials[ok].astype(int)
        p_idx = p_idx[ok].astype(int)

        spike_trial_idx = np.searchsorted(trial_ids, spike_trials)
        ok2 = (
            (spike_trial_idx >= 0)
            & (spike_trial_idx < n_trials)
            & (trial_ids[spike_trial_idx] == spike_trials)
        )
        if not np.any(ok2):
            return np.zeros((n_trials, n_speed_bins, n_progress_bins), dtype=np.int64)

        spike_trial_idx = spike_trial_idx[ok2].astype(int)
        p_idx = p_idx[ok2].astype(int)

        if speed_mode == "edges":
            s_idx = speed_idx[spike_pos_idx][ok][ok2].astype(int)
            flat = np.bincount(
                spike_trial_idx * (n_speed_bins * n_progress_bins) + s_idx * n_progress_bins + p_idx,
                minlength=n_trials * n_speed_bins * n_progress_bins,
            )
            return flat.reshape(n_trials, n_speed_bins, n_progress_bins)

        # Sliding windows: count spikes per window within each (trial, progress) group.
        out = np.zeros((n_trials, n_speed_bins, n_progress_bins), dtype=np.int64)
        spike_speed = speed[spike_pos_idx][ok][ok2].astype(float)

        group_id = spike_trial_idx * n_progress_bins + p_idx
        order = np.argsort(group_id, kind="mergesort")
        group_id = group_id[order]
        spike_speed = spike_speed[order]

        unique_groups, start_idx = np.unique(group_id, return_index=True)
        start_idx = np.append(start_idx, spike_speed.size)

        left_edges = speed_bin_spec["left"]
        right_edges = speed_bin_spec["right"]
        for k, gid in enumerate(unique_groups):
            s = start_idx[k]
            e = start_idx[k + 1]
            if e <= s:
                continue
            t = int(gid // n_progress_bins)
            p = int(gid % n_progress_bins)
            out[t, :, p] = _counts_in_windows(spike_speed[s:e], left_edges, right_edges)

        return out

    spike_counts_trials_units_2d = [_spike_counts_trials_2d_for_unit(st) for st in spikes_list]

    # ----------------------------
    # Balanced rate helper (two occupancy thresholds)
    # ----------------------------
    def _balanced_rate(spike_2d, occ_2d):
        """
        Returns:
          rate_1d: (..., n_speed_bins)
          valid_speed: (..., n_speed_bins)
        """
        spike_2d = np.asarray(spike_2d)
        occ_2d = np.asarray(occ_2d, dtype=float)

        if spike_2d.shape != occ_2d.shape:
            raise ValueError(f"spike_2d and occ_2d shape mismatch: {spike_2d.shape} vs {occ_2d.shape}")

        # normalize weights
        w_local = np.asarray(w, dtype=float).ravel()
        w_local = w_local / np.sum(w_local)

        with np.errstate(divide="ignore", invalid="ignore"):
            rate_2d = spike_2d / occ_2d

        # cell validity uses min_occupancy_cell_s
        valid_cell = (occ_2d > float(min_occupancy_cell_s)) & np.isfinite(rate_2d)

        # broadcast w to (..., speed, progress)
        w_b = w_local.reshape((1,) * (rate_2d.ndim - 1) + (n_progress_bins,))

        numerator = np.sum(w_b * np.where(valid_cell, rate_2d, 0.0), axis=-1)
        denom_w = np.sum(w_b * valid_cell.astype(float), axis=-1)  # available weight mass per speed bin

        rate_1d = np.full_like(denom_w, np.nan, dtype=float)
        has_any_valid_progress = denom_w > 0
        rate_1d[has_any_valid_progress] = numerator[has_any_valid_progress] / denom_w[has_any_valid_progress]

        # speed-bin total occupancy uses min_occupancy_speed_s
        total_time_speed = np.sum(occ_2d, axis=-1)  # (..., n_speed_bins)
        enough_speed_time = total_time_speed > float(min_occupancy_speed_s)

        valid_speed = has_any_valid_progress & enough_speed_time & np.isfinite(rate_1d)
        rate_1d[~valid_speed] = np.nan

        return rate_1d, valid_speed

    # pooled valid bins (for slope/curvature fitting)
    _, pooled_valid_bins = _balanced_rate(
        np.zeros_like(occ_time_total_2d),
        occ_time_total_2d,
    )
    pooled_valid_bins = pooled_valid_bins & np.isfinite(speed_centers)

    # ----------------------------
    # CI quantiles
    # ----------------------------
    alpha = 1.0 - float(ci)
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)

    x = np.asarray(speed_centers, dtype=float)

    def _linear_slope(xv, yv):
        if xv.size < 2:
            return np.nan
        x_mean = np.mean(xv)
        y_mean = np.mean(yv)
        denom = np.sum((xv - x_mean) ** 2)
        if denom <= 0:
            return np.nan
        return np.sum((xv - x_mean) * (yv - y_mean)) / denom

    def _quadratic_curvature(xv, yv):
        if xv.size < 3:
            return np.nan
        X = np.column_stack([np.ones_like(xv), xv, xv ** 2])
        beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
        return float(beta[2])

    # ----------------------------
    # Outputs
    # ----------------------------
    speed_tuning = {}
    lower_ci = {}
    upper_ci = {}
    slope_boot = {}
    curvature_boot = {}

    for unit, counts_trials_2d in enumerate(spike_counts_trials_units_2d):
        # pooled 2D spike counts
        spike_total_2d = counts_trials_2d.sum(axis=0)

        # pooled balanced curve
        rate_pooled, _ = _balanced_rate(spike_total_2d, occ_time_total_2d)
        speed_tuning[unit] = rate_pooled

        # bootstrap 2D spike and occupancy
        spike_boot_2d = np.tensordot(boot_weights, counts_trials_2d, axes=([1], [0]))      # (n_boot, speed, prog)
        occ_boot_2d = np.tensordot(boot_weights, occ_time_trials_2d, axes=([1], [0]))     # (n_boot, speed, prog)

        # balanced bootstrap curves
        rates_boot, _ = _balanced_rate(spike_boot_2d, occ_boot_2d)  # (n_boot, speed)

        lower_ci[unit] = np.nanpercentile(rates_boot, lo_q, axis=0)
        upper_ci[unit] = np.nanpercentile(rates_boot, hi_q, axis=0)

        # slope/curvature per bootstrap replicate
        s_arr = np.full(n_boot, np.nan, dtype=float)
        q_arr = np.full(n_boot, np.nan, dtype=float)

        for r in range(n_boot):
            y = rates_boot[r, :]
            ok = pooled_valid_bins & np.isfinite(y)
            xv = x[ok]
            yv = y[ok]
            s_arr[r] = _linear_slope(xv, yv)
            q_arr[r] = _quadratic_curvature(xv, yv)

        slope_boot[unit] = s_arr
        curvature_boot[unit] = q_arr

    if peak_normalize:
        _peak_normalize_dicts_from_upper_ci(
            tuning=speed_tuning,
            lower_ci=lower_ci,
            upper_ci=upper_ci,
            slope_boot=slope_boot,
            curvature_boot=curvature_boot,
        )

    return speed_tuning, speed_centers, lower_ci, upper_ci, slope_boot, curvature_boot
































def make_tuning_comparison_plotly_figure(
    *,
    unit_ids,
    speed_tuning,
    speed_centers,
    progress_tuning,
    progress_centers,
    location_tuning,
    location_centers,
    speed_lower=None,
    speed_upper=None,
    progress_lower=None,
    progress_upper=None,
    location_lower=None,
    location_upper=None,
    speed_shape_by_unit=None,
    progress_shape_by_unit=None,
    location_shape_by_unit=None,
    title=None,
    height_per_unit: int = 220,
):
    """
    Build a 3-column Plotly figure (speed / progress / location) with 1 row per unit.

    This is intended to be embedded in FigURL via `figurl.Plotly`.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    unit_ids = list(unit_ids)
    if len(unit_ids) == 0:
        raise ValueError("unit_ids is empty")

    def _shape_str(shape_map, unit_id):
        if not shape_map:
            return None
        s = shape_map.get(unit_id, None)
        return None if s is None else str(s)

    row_titles = []
    for u in unit_ids:
        parts = [f"unit {u}"]
        ss = _shape_str(speed_shape_by_unit, u)
        ps = _shape_str(progress_shape_by_unit, u)
        ls = _shape_str(location_shape_by_unit, u)
        if ss is not None:
            parts.append(f"speed={ss}")
        if ps is not None:
            parts.append(f"progress={ps}")
        if ls is not None:
            parts.append(f"location={ls}")
        row_titles.append(" | ".join(parts))

    fig = make_subplots(
        rows=len(unit_ids),
        cols=3,
        shared_xaxes=False,
        shared_yaxes=False,
        vertical_spacing=0.04,
        horizontal_spacing=0.05,
        row_titles=row_titles,
        column_titles=["Speed", "Progress", "Location"],
    )

    def _add_curve(
        *,
        row: int,
        col: int,
        x,
        y,
        lo=None,
        hi=None,
        show_legend: bool,
    ):
        if x is None:
            raise ValueError("Missing x (bin centers) for subplot")
        if y is None:
            raise ValueError("Missing tuning curve for requested unit_ids")
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if lo is not None and hi is not None:
            lo = np.asarray(lo, dtype=float)
            hi = np.asarray(hi, dtype=float)
            fig.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([hi, lo[::-1]]),
                    fill="toself",
                    fillcolor="rgba(31, 119, 180, 0.20)",
                    line=dict(width=0),
                    name="CI",
                    showlegend=show_legend,
                    hoverinfo="skip",
                ),
                row=row,
                col=col,
            )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines+markers",
                line=dict(color="rgb(31, 119, 180)", width=2),
                marker=dict(size=6),
                name="Mean",
                showlegend=show_legend,
            ),
            row=row,
            col=col,
        )

    for i, unit_id in enumerate(unit_ids):
        row = i + 1

        show_legend = (row == 1)

        _add_curve(
            row=row,
            col=1,
            x=speed_centers,
            y=_get_unit_curve(speed_tuning, unit_id),
            lo=_get_unit_curve(speed_lower, unit_id),
            hi=_get_unit_curve(speed_upper, unit_id),
            show_legend=show_legend,
        )
        _add_curve(
            row=row,
            col=2,
            x=progress_centers,
            y=_get_unit_curve(progress_tuning, unit_id),
            lo=_get_unit_curve(progress_lower, unit_id),
            hi=_get_unit_curve(progress_upper, unit_id),
            show_legend=False,
        )
        _add_curve(
            row=row,
            col=3,
            x=location_centers,
            y=_get_unit_curve(location_tuning, unit_id),
            lo=_get_unit_curve(location_lower, unit_id),
            hi=_get_unit_curve(location_upper, unit_id),
            show_legend=False,
        )

    fig.update_layout(
        template="simple_white",
        height=max(350, height_per_unit * len(unit_ids)),
        width=1400,
        title=title or "Tuning curves",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=70, b=40),
    )

    # Axis labels
    # Use a single y-axis title (middle row, first column) to reduce clutter.
    mid_row = int(np.ceil(len(unit_ids) / 2))
    fig.update_yaxes(title_text="Firing rate (Hz)", row=mid_row, col=1)
    fig.update_xaxes(title_text="speed", row=len(unit_ids), col=1)
    fig.update_xaxes(title_text="trial_progress", row=len(unit_ids), col=2)
    fig.update_xaxes(title_text="linear_position", row=len(unit_ids), col=3)

    return fig


def make_tuning_comparison_figurl(
    *,
    unit_ids,
    speed_tuning,
    speed_centers,
    progress_tuning,
    progress_centers,
    location_tuning,
    location_centers,
    speed_lower=None,
    speed_upper=None,
    progress_lower=None,
    progress_upper=None,
    location_lower=None,
    location_upper=None,
    speed_shape_by_unit=None,
    progress_shape_by_unit=None,
    location_shape_by_unit=None,
    label: str = "tuning-comparison",
    base_url=None,
    hide_app_bar: bool = False,
    title=None,
    height_per_unit: int = 220,
    speed_x_label: str = "speed",
    progress_x_label: str = "trial_progress",
    location_x_label: str = "linear_position",
):
    """
    Create a FigURL showing speed/progress/location tuning curves side-by-side.

    Typical usage (after you compute the tunings/bootstraps in a notebook)
    --------------------------------------------------------------------
    url = make_tuning_comparison_figurl(
        unit_ids=[0, 1, 2],
        speed_tuning=speed_tuning,
        speed_centers=speed_centers,
        speed_lower=speed_lower,
        speed_upper=speed_upper,
        progress_tuning=progress_tuning,
        progress_centers=progress_centers,
        progress_lower=progress_lower,
        progress_upper=progress_upper,
        location_tuning=position_tuning,
        location_centers=position_bin_centers,
        speed_shape_by_unit=speed_shape_by_unit,        # optional
        progress_shape_by_unit=progress_shape_by_unit,  # optional
        label="wilbur20210512_epoch8_run",
        title="Epoch 8 tuning (run)",
    )
    print(url)

    To plot distance-normalized trial progress, compute and pass:
      - column='trial_progress_distance' into compute_tuning_bootstrap_trials
      - progress_x_label='trial_progress_distance' here

    Returns
    -------
    url : str
        A `https://figurl.org/...` URL (or another base_url if provided).
    """
    # NOTE: figurl's built-in Plotly plugin in this environment only preserves trace data,
    # not the full Plotly layout. That means Plotly subplots collapse/overlay.
    # To keep the 3-column layout (and optionally support scrolling), use sortingview's
    # PlotlyFigure + Box layout, and let it generate the FigURL.

    import plotly.graph_objects as go
    from sortingview.views import Box, LayoutItem, PlotlyFigure

    unit_ids = list(unit_ids)
    if len(unit_ids) == 0:
        raise ValueError("unit_ids is empty")

    def _shape_str(shape_map, unit_id):
        if not shape_map:
            return None
        s = shape_map.get(unit_id, None)
        return None if s is None else str(s)

    def _as_float_list(x):
        if x is None:
            return None
        return [float(v) for v in np.asarray(x, dtype=float).ravel()]

    def _make_single_panel(*, x, y, lo, hi, x_label: str):
        if y is None:
            raise ValueError("Missing tuning curve for a requested unit_id")
        x = np.asarray(x, dtype=float).ravel()
        y = np.asarray(y, dtype=float).ravel()
        lo = None if lo is None else np.asarray(lo, dtype=float).ravel()
        hi = None if hi is None else np.asarray(hi, dtype=float).ravel()

        if x.shape != y.shape:
            raise ValueError(f"x/y length mismatch for {x_label}: {x.shape} vs {y.shape}")
        if (lo is not None) and (lo.shape != x.shape):
            raise ValueError(f"x/lo length mismatch for {x_label}: {x.shape} vs {lo.shape}")
        if (hi is not None) and (hi.shape != x.shape):
            raise ValueError(f"x/hi length mismatch for {x_label}: {x.shape} vs {hi.shape}")

        def _true_runs(mask: np.ndarray):
            # yields (start, end) with end exclusive
            mask = np.asarray(mask, dtype=bool)
            if mask.size == 0:
                return
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                return
            breaks = np.where(np.diff(idx) > 1)[0]
            starts = np.concatenate([[idx[0]], idx[breaks + 1]])
            ends = np.concatenate([idx[breaks] + 1, [idx[-1] + 1]])
            for a, b in zip(starts, ends):
                yield int(a), int(b)

        fig = go.Figure()

        # CI band
        if (lo is not None) and (hi is not None):
            mask_ci = np.isfinite(x) & np.isfinite(lo) & np.isfinite(hi)
            for a, b in _true_runs(mask_ci):
                if b - a < 2:
                    continue
                xv = x[a:b]
                lov = lo[a:b]
                hiv = hi[a:b]
                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([xv, xv[::-1]]),
                        y=np.concatenate([hiv, lov[::-1]]),
                        fill="toself",
                        fillcolor="rgba(31, 119, 180, 0.20)",
                        line=dict(width=0),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )

        # Mean curve
        mask_mean = np.isfinite(x) & np.isfinite(y)
        for a, b in _true_runs(mask_mean):
            if b - a < 1:
                continue
            fig.add_trace(
                go.Scatter(
                    x=x[a:b],
                    y=y[a:b],
                    mode="lines+markers",
                    line=dict(color="rgb(31, 119, 180)", width=2),
                    marker=dict(size=6),
                    showlegend=False,
                )
            )

        fig.update_layout(
            template="simple_white",
            margin=dict(l=45, r=10, t=25, b=35),
            height=240,
        )
        fig.update_xaxes(title_text=x_label)
        fig.update_yaxes(title_text="Hz", rangemode="tozero")
        return fig

    unit_rows = []
    for unit_id in unit_ids:
        ss = _shape_str(speed_shape_by_unit, unit_id)
        ps = _shape_str(progress_shape_by_unit, unit_id)
        ls = _shape_str(location_shape_by_unit, unit_id)

        row_title_parts = [f"unit {unit_id}"]
        if ss is not None:
            row_title_parts.append(f"speed={ss}")
        if ps is not None:
            row_title_parts.append(f"progress={ps}")
        if ls is not None:
            row_title_parts.append(f"location={ls}")
        row_title = " | ".join(row_title_parts)

        speed_fig = _make_single_panel(
            x=speed_centers,
            y=_get_unit_curve(speed_tuning, unit_id),
            lo=_get_unit_curve(speed_lower, unit_id),
            hi=_get_unit_curve(speed_upper, unit_id),
            x_label=speed_x_label,
        )
        progress_fig = _make_single_panel(
            x=progress_centers,
            y=_get_unit_curve(progress_tuning, unit_id),
            lo=_get_unit_curve(progress_lower, unit_id),
            hi=_get_unit_curve(progress_upper, unit_id),
            x_label=progress_x_label,
        )
        location_fig = _make_single_panel(
            x=location_centers,
            y=_get_unit_curve(location_tuning, unit_id),
            lo=_get_unit_curve(location_lower, unit_id),
            hi=_get_unit_curve(location_upper, unit_id),
            x_label=location_x_label,
        )

        row_view = Box(
            direction="horizontal",
            show_titles=True,
            items=[
                LayoutItem(PlotlyFigure(fig=speed_fig, height=260), title="Speed", stretch=1),
                LayoutItem(PlotlyFigure(fig=progress_fig, height=260), title="Progress", stretch=1),
                LayoutItem(PlotlyFigure(fig=location_fig, height=260), title="Location", stretch=1),
            ],
        )

        unit_rows.append(LayoutItem(row_view, title=row_title, min_size=280))

    layout = Box(
        direction="vertical",
        scrollbar=True,
        show_titles=True,
        items=unit_rows,
    )

    # The sortingview URL generator uses FIGURL_BASE_URL env var implicitly if you set it,
    # but doesn't accept base_url directly. Keep the parameter for API compatibility.
    if base_url is not None:
        raise ValueError(
            "base_url is not supported for the sortingview-based layout; "
            "set FIGURL_BASE_URL env var instead, or pass base_url=None."
        )

    # hide_app_bar isn't supported here either (sortingview view_url controls the app)
    if hide_app_bar:
        raise ValueError("hide_app_bar is not supported for sortingview-based figures")

    # Plotly figures commonly contain float64 ndarrays in their `.to_dict()`;
    # sortingview/figurl serialization rejects float64 unless allow_float64=True.
    return layout.url(label=label, allow_float64=True)




def plot_speed_vs_progress(
    trialized_position,
    mask,
    n_progress_bins,
    progress_col,
    trial_number_col="trial_number",
    speed_col="speed",
    linewidth: float = 2.0,
    *,
    window_size: Optional[float] = None,
    step_size: Optional[float] = None,
    window_centers: Optional[np.ndarray] = None,
    min_samples_per_trial_window: int = 1,
    recompute_progress: bool = False,
    ax: plt.axes = None
):
    """
    Plot average speed as a function of trial progress.

    Two binning modes are supported:

    - Discrete bins (default): progress is discretized into `n_progress_bins` equal-width bins.
      Per-bin mean speed is computed per trial, then averaged across trials with SEM.

    - Sliding window ("continuous") binning: when `window_size` and `step_size` are provided,
      per-window mean speed is computed per trial using a moving window in progress space.
      Windows are centered at `window_centers` (if provided); otherwise, if `step_size` is
      provided, centers are `np.arange(0, 1, step_size)` (with the endpoint included when
      possible). If neither `window_centers` nor `step_size` are provided, window centers are
      automatically set from `n_progress_bins` as `(np.arange(n_progress_bins) + 0.5) / n_progress_bins`.
      The plot shows mean ± SEM across trials.

    Parameters
    ----------
    trialized_position : pd.DataFrame
        Must contain `trial_number_col`, `progress_col`, and `speed_col`.
    mask : array-like of bool
        Boolean mask aligned to `trialized_position` rows, selecting samples to include.
    n_progress_bins : int
        Number of discrete progress bins when using the default (non-sliding) mode.
    progress_col : str
        Column name for progress (expected in [0, 1]).
    trial_number_col : str
        Column name for trial identity.
    speed_col : str
        Column name for speed.
    window_size : float, optional
        Sliding-window width in progress units (0-1). Enables sliding window mode when used
        together with `step_size`.
    step_size : float, optional
        Step between successive window centers in progress units (0-1). Enables sliding
        window mode when used together with `window_size`.
    window_centers : np.ndarray, optional
        Explicit window centers in progress units (0-1). If provided, `step_size` and
        `n_progress_bins` are ignored for center placement.
    min_samples_per_trial_window : int
        Minimum number of samples from a trial required to contribute to a window; otherwise
        that trial/window entry is treated as NaN.
    recompute_progress : bool
        If True, re-normalize `progress_col` within each trial after applying `mask`.
        This is useful when `mask` removes early/late parts of a trial, causing progress to
        no longer span [0, 1]. For each trial, the masked progress values are mapped to
        [0, 1] using (p - p_min) / (p_max - p_min). Trials with no progress range
        (p_max == p_min) are dropped.
    """
    trialized_position_masked = trialized_position.loc[mask, [trial_number_col, progress_col, speed_col]].copy()

    trialized_position_masked = trialized_position_masked.replace([np.inf, -np.inf], np.nan).dropna()

    if recompute_progress:
        per_trial_min = trialized_position_masked.groupby(trial_number_col)[progress_col].transform("min")
        per_trial_max = trialized_position_masked.groupby(trial_number_col)[progress_col].transform("max")
        denom = per_trial_max - per_trial_min
        trialized_position_masked[progress_col] = (trialized_position_masked[progress_col] - per_trial_min) / denom
        # Drop trials/windows with undefined normalization (e.g., only one unique progress value).
        trialized_position_masked = trialized_position_masked.replace([np.inf, -np.inf], np.nan).dropna()

    trialized_position_masked = trialized_position_masked[
        (trialized_position_masked[progress_col] >= 0) & (trialized_position_masked[progress_col] <= 1)
    ]

    use_sliding_window = (window_size is not None) or (step_size is not None) or (window_centers is not None)

    if use_sliding_window:
        if window_size is None:
            raise ValueError("window_size must be provided for sliding-window binning")
        if window_size <= 0:
            raise ValueError("window_size must be > 0")
        if min_samples_per_trial_window < 1:
            raise ValueError("min_samples_per_trial_window must be >= 1")

        custom_centers = window_centers is not None
        centers_from_n_bins = False
        effective_step_size: Optional[float] = None
        if custom_centers:
            window_centers = np.asarray(window_centers, dtype=float)
        else:
            if step_size is None:
                if n_progress_bins is None:
                    raise ValueError("n_progress_bins must be provided when step_size is not specified")
                if int(n_progress_bins) < 1:
                    raise ValueError("n_progress_bins must be >= 1")
                n_centers = int(n_progress_bins)
                centers_from_n_bins = True
                window_centers = (np.arange(n_centers, dtype=float) + 0.5) / float(n_centers)
                effective_step_size = 1.0 / float(n_centers)
            else:
                if step_size <= 0:
                    raise ValueError("step_size must be > 0")
                effective_step_size = float(step_size)
                # Include endpoint if it lands close due to floating point; clip to [0, 1].
                window_centers = np.arange(0.0, 1.0 + step_size * 0.5, step_size, dtype=float)

        window_centers = np.clip(window_centers, 0.0, 1.0)
        # Guard against duplicates created by clipping/float rounding.
        window_centers = np.unique(window_centers)

        half_window = window_size / 2.0
        window_lows = np.clip(window_centers - half_window, 0.0, 1.0)
        window_highs = np.clip(window_centers + half_window, 0.0, 1.0)

        def _trial_window_means(progress_values: np.ndarray, speed_values: np.ndarray) -> np.ndarray:
            order = np.argsort(progress_values)
            p = progress_values[order]
            s = speed_values[order]
            prefix = np.concatenate([[0.0], np.cumsum(s, dtype=float)])

            left = np.searchsorted(p, window_lows, side="left")
            right = np.searchsorted(p, window_highs, side="right")
            counts = right - left

            sums = prefix[right] - prefix[left]
            means = np.full(window_centers.shape, np.nan, dtype=float)
            valid = counts >= min_samples_per_trial_window
            means[valid] = sums[valid] / counts[valid]
            return means

        trial_groups = trialized_position_masked.groupby(trial_number_col, sort=False)
        per_trial = []
        for _, grp in trial_groups:
            per_trial.append(
                _trial_window_means(
                    grp[progress_col].to_numpy(dtype=float),
                    grp[speed_col].to_numpy(dtype=float),
                )
            )

        if len(per_trial) == 0:
            mean_speed_across_trials = np.full(window_centers.shape, np.nan, dtype=float)
            sem_speed_across_trials = np.full(window_centers.shape, np.nan, dtype=float)
            n_trials_per_bin = np.zeros(window_centers.shape, dtype=int)
        else:
            per_trial_arr = np.asarray(per_trial, dtype=float)  # (n_trials, n_windows)
            mean_speed_across_trials = np.nanmean(per_trial_arr, axis=0)
            n_trials_per_bin = np.sum(~np.isnan(per_trial_arr), axis=0).astype(int)

            std_speed_across_trials = np.nanstd(per_trial_arr, axis=0, ddof=1)
            sem_speed_across_trials = std_speed_across_trials / np.sqrt(np.maximum(n_trials_per_bin, 1))

        x_values = window_centers
        x_label = (
            f"{progress_col} (sliding window; window={window_size:g}, "
            + (
                "custom centers"
                if custom_centers
                else (f"n={len(window_centers)}" if centers_from_n_bins else f"step={effective_step_size:g}")
            )
            + ")"
        )
    else:
        progress_bin_edges = np.linspace(0, 1, n_progress_bins + 1)
        progress_bin_centers = (progress_bin_edges[:-1] + progress_bin_edges[1:]) / 2

        trialized_position_masked["progress_bin"] = pd.cut(
            trialized_position_masked[progress_col],
            bins=progress_bin_edges,
            labels=False,
            include_lowest=True,
        ).astype(int)

        speed_by_trial_and_bin = (
            trialized_position_masked
            .groupby([trial_number_col, "progress_bin"])[speed_col]
            .mean()
            .unstack("progress_bin")
            .reindex(columns=np.arange(n_progress_bins))
        )

        mean_speed_across_trials = speed_by_trial_and_bin.mean(axis=0, skipna=True).to_numpy()

        n_trials_per_bin = speed_by_trial_and_bin.notna().sum(axis=0).to_numpy()
        std_speed_across_trials = speed_by_trial_and_bin.std(axis=0, skipna=True, ddof=1).to_numpy()
        sem_speed_across_trials = std_speed_across_trials / np.sqrt(np.maximum(n_trials_per_bin, 1))

        x_values = progress_bin_centers
        x_label = f"{progress_col} (binned)"

    # --- plot ---
    # fig, ax = plt.subplots(figsize=(14, 8), layout="tight")

    ax.plot(
        x_values,
        mean_speed_across_trials,
        # color="k",
        linewidth=linewidth,
        label="mean across trials",
        marker="o",
    )
    ax.fill_between(
        x_values,
        mean_speed_across_trials - sem_speed_across_trials,
        mean_speed_across_trials + sem_speed_across_trials,
        # color="k",
        alpha=0.2,
        label="±1 SEM across trials",
    )

    # ax.set_xlabel(x_label)
    # ax.set_ylabel("speed (cm/s)")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False)
    plt.show()
