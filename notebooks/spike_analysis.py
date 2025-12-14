import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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


def get_unit_firing_rate(spikes: list, unit: int):
    return len(spikes[unit])/((spikes[unit][-1]) - (spikes[unit][0]))


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





def plot_rest_vs_run_fr(mean_fr_df: pd.DataFrame, alpha: float = 0.05):
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
            )

        plt.xticks([0, 1], ["Rest", "Run"])
        plt.ylabel("Firing rate (Hz)")
        plt.title("Per-unit change in firing rate")
        plt.show()

    elif plot_type == "2":
        plt.figure(figsize=(24, 12), layout="tight")

        plt.errorbar(
            mean_fr_df["rest firing rate"],
            mean_fr_df["run firing rate"],
            xerr=mean_fr_df["rest sem"],
            yerr=mean_fr_df["run sem"],
            fmt="o",
            alpha=0.7,
            capsize=3,
        )

        max_val = max(
            mean_fr_df["rest firing rate"].max(),
            mean_fr_df["run firing rate"].max(),
        )
        plt.plot([0, max_val], [0, max_val], "k--", label="y = x")

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




def plot_speed_tuning_grid(speed_tuning: dict,
                           position_df: pd.DataFrame,
                           spikes_list: list,
                           mask: np.ndarray = None,
                           n_units: int = -1,
                           label = None):
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
        ax.plot(speed_bin_centers, rate, marker='o', linewidth=1)
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






def compute_speed_tuning_CI(spikes_list: list,
                            trialwise_position_df: pd.DataFrame,
                            n_bins: int = 8):
    
    
    def _spike_counts_per_speed_bin(spike_times, timestamps, speed, speed_bins):
        speed_at_spikes = spikes_to_speed(spike_times, timestamps, speed)
        spike_counts, _ = np.histogram(speed_at_spikes, bins=speed_bins)
        return spike_counts    
        
    
    timestamps = trialwise_position_df.index.to_list()
    dt = np.median(np.diff(timestamps))
    
    speed = trialwise_position_df.speed.to_numpy()
    max_speed = np.nanmax(speed)  
    speed_bins = np.linspace(0, max_speed, n_bins + 1)
    
    occupancy_counts, _ = np.histogram(speed, bins=speed_bins)
    occupancy_time = occupancy_counts * dt
    
    
    #for one unit
    spike_times = spikes_list[unit]
    spike_counts = _spike_counts_per_speed_bin(spike_times, timestamps, speed, speed_bins)

    valid = occupancy_time > MIN_OCCUPANCY

    rate = np.full_like(occupancy_time, np.nan, dtype=float)
    se_rate = np.full_like(occupancy_time, np.nan, dtype=float)

    rate[valid] = spike_counts[valid] / occupancy_time[valid]
    se_rate[valid] = np.sqrt(spike_counts[valid]) / occupancy_time[valid]

    z = 1.96  # for ~95% CI
    lower = rate - z * se_rate
    upper = rate + z * se_rate

    lower = np.clip(lower, 0, None)
    
    return rate, lower, upper




def get_si_recording_and_sorting(sorted_group_key: dict, ):

    spikes, units = SortedSpikesGroup.fetch_spike_data(
        key= sorted_group_key,
        return_unit_ids=True,
    )

    n_units = len(units)
    unit_ids = np.arange(n_units) 

    unit_table = pd.DataFrame(units)
    unit_table["group_unit_id"] = unit_ids


    units_by_merge_group = (
        unit_table
        .groupby("spikesorting_merge_id")["group_unit_id"]
        .apply(list)
        .to_dict()
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


    recording = sc.aggregate_channels(recordings)

    renamed_unit_ids = np.concatenate(
        [s.get_unit_ids() for s in sortings_filtered]
    )
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
    suptitle: str = ""
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

                ax.plot(speed_bin_centers, fr_curve, marker="o", linewidth=1)
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
                ax.plot(t, template[:, peak_ch], color="C0")
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





def plot_inbound_outbound_fr(fr_df: pd.DataFrame):
    """
    Plot per-unit inbound vs outbound firing rates with Poisson SE error bars:
      - left: per-unit paired lines with vertical error bars
      - right: scatter inbound vs outbound with x/y error bars.
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
        ax.plot([0, 1], [fr_in, fr_out], "-o", color="gray", alpha=0.4)
        ax.errorbar(0, fr_in, yerr=se_in, fmt="none", ecolor="k", alpha=0.4, capsize=3)
        ax.errorbar(1, fr_out, yerr=se_out, fmt="none", ecolor="k", alpha=0.4, capsize=3)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Inbound", "Outbound"])
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Per-unit inbound vs outbound firing rates")

    # right: scatter with error bars
    ax = axes[1]
    ax.errorbar(inbound, outbound, xerr=inbound_se, yerr=outbound_se,
                fmt="o", alpha=0.7, ecolor="k", capsize=3)
    max_val = np.nanmax([inbound.max(), outbound.max()]) if inbound.size > 0 else 1.0
    ax.plot([0, max_val], [0, max_val], "k--", label="y = x")
    ax.set_xlabel("Inbound FR (Hz)")
    ax.set_ylabel("Outbound FR (Hz)")
    ax.set_title("Inbound vs outbound FR (with SE)")
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





from scipy import stats


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
                           label = None):
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
        ax.plot(position_bin_centers, rate, marker='o', linewidth=1)
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




def compute_tuning(position_df: pd.DataFrame,
                         column: str,
                         spikes_list: list,
                         n_bins: int = 6,
                         mask: np.ndarray = None):
    """
    position_df: full tuner_trials_merged_df (NOT pre-masked)
    mask: boolean array on position_df.index (e.g. zone=='run' & trial_type=='inbound')
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
    max_tuner = np.nanmax(tuner_masked)
    tuner_bins = np.linspace(0, max_tuner, n_bins + 1)
    tuner_bin_centers = (tuner_bins[:-1] + tuner_bins[1:]) / 2

    occupancy_counts, _ = np.histogram(tuner_masked, bins=tuner_bins)
    occupancy_time = occupancy_counts * dt

    tuner_tuning = {}

    for unit, spike_times in enumerate(spikes_list):
        tuner_at_spikes = spikes_to_tuner(spike_times, timestamps, tuner, mask=mask)
        spike_counts, _ = np.histogram(tuner_at_spikes, bins=tuner_bins)

        with np.errstate(divide='ignore', invalid='ignore'):
            firing_rate = spike_counts / occupancy_time
            firing_rate[occupancy_time == 0] = np.nan

        tuner_tuning[unit] = firing_rate

    return tuner_tuning, tuner_bin_centers





def plot_tuning_grid(tuner_tuning: dict,
                           column: str,
                           position_df: pd.DataFrame,
                           spikes_list: list,
                           mask: np.ndarray = None,
                           n_units: int = -1,
                           label = None):
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

    max_tuner = np.nanmax(tuner_masked)
    tuner_bins = np.linspace(0, max_tuner, n_bins + 1)
    tuner_bin_centers = (tuner_bins[:-1] + tuner_bins[1:]) / 2

    occupancy_counts, _ = np.histogram(tuner_masked, bins=tuner_bins)
    occupancy_time = occupancy_counts * dt  # seconds in each tuner bin

    # --- helper: spike counts per tuner bin, using the same mask ---
    def _spike_counts_per_tuner_bin(spike_times):
        tuner_at_spikes = spikes_to_tuner(spike_times, timestamps, tuner, mask=mask)
        spike_counts, _ = np.histogram(tuner_at_spikes, bins=tuner_bins)
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

        ax = axes[i]
        ax.plot(tuner_bin_centers, rate, marker='o', linewidth=1)
        ax.fill_between(tuner_bin_centers, lower, upper,
                        color='C0', alpha=0.3, label='95% CI')

        ax.set_title(str(unit), fontsize=8)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.tick_params(axis='both', labelsize=6)

    # hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'{column} tuning curves {label}', fontsize=14)
    fig.text(0.5, 0.04, f'{column}', ha='center')
    fig.text(0.04, 0.5, 'Firing rate (Hz)', va='center', rotation='vertical')
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.ylim(-10, 20)
    plt.show()