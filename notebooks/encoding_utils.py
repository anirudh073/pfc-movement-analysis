"""
Utility functions for GLM encoding analysis (encoding.ipynb).

Sections:
  0. Configuration     : CONFIG, build_model_registry
  1. Data preparation  : interp_col, load_and_prepare_data
  2. GLM fitting       : fit_glm_all_units, fit_single_unit, add_spike_history
  3. Drop-one analysis : make_drop_one_specs, build_reduced_formula,
                         fit_drop_one, run_drop_one_suite, compute_drop_one_lrt
  4. Diagnostics       : compute_residuals, plot_residuals,
                         compute_ks_rescaled, plot_ks,
                         _simulate_autoregressive, plot_predicted_isi,
                         plot_diagnostics
  5. Visualization     : set_plot_state, plot_place_field, plot_place_field_grid
"""

import ast
import os, re, warnings
from functools import partial
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, wilcoxon, spearmanr
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from patsy import build_design_matrices
import itertools
from collections import Counter 
sns.set_context("talk")


# 0. Configuration 

CONFIG = dict(
    bin_size      = 0.002, #ms
    speed_min     = 5,
    speed_max     = 120,
    speed_df      = 4, #degrees of freedom
    pos_df        = 8,
    branch_pos_df = 4,
    tp_df         = 6, #trial progress
    base_dir      = "/media/labuser/NA_1_2025/spyglass/wilbur",
    wtrack_name   = "Wtrack_center0_wilbur20210512",
    trialized_position = "/media/labuser/NA_1_2025/spyglass/wilbur/analysis/position/trialized_position_center0.csv"
)


BRANCH_SPECS = {
    "top": {
        "segments": (1, 3),
        "origin_segment": 1,
        "direction": -1.0,
    },
    "middle": {
        "segments": (0,),
        "origin_segment": 0,
        "direction": 1.0,
    },
    "bottom": {
        "segments": (2, 4),
        "origin_segment": 2,
        "direction": 1.0,
    },
}


def build_model_registry(cfg=None):
    """Return dict of {name: {formula, dataset}} for all encoding models.

    ``dataset`` is "common" (all trials) or "outbound" (outbound only).
    Callers pair models with the appropriate (cov_df, spike_counts) by key.
    """
    cfg = cfg or CONFIG
    spd = f"bs(speed_scaled, df={cfg['speed_df']})"
    pos = f"bs(pos_scaled, df={cfg['pos_df']})"
    branch_pos = (
        f"C(branch_id) + bs(branch_pos_scaled, df={cfg['branch_pos_df']}):C(branch_id)"
    )
    tp  = f"cr(trial_progress, df={cfg['tp_df']}, constraints='center')"

    return {
        "null":                    {"formula": "spike_count ~ 1",                             "dataset": "common"},
        "null_outbound":           {"formula": "spike_count ~ 1",                             "dataset": "outbound"},
        "trial_type":              {"formula": "spike_count ~ trial_type",                    "dataset": "common"},
        "choice":                  {"formula": "spike_count ~ choice",                        "dataset": "outbound"},
        "speed_spline":            {"formula": f"spike_count ~ {spd}",                        "dataset": "common"},
        "pos_spline":              {"formula": f"spike_count ~ {pos}",                        "dataset": "common"},
        "branch_pos_spline":       {"formula": f"spike_count ~ {branch_pos}",                 "dataset": "common"},
        "trial_progress_spline":   {"formula": f"spike_count ~ {tp}",                         "dataset": "common"},
        "full_model":              {"formula": f"spike_count ~ trial_type + {branch_pos} + {spd}",   "dataset": "common"},
        "temporal_model":          {"formula": f"spike_count ~ trial_type + {tp} + {spd}",    "dataset": "common"},
        "choice_full_model":       {"formula": f"spike_count ~ choice + {pos} + {spd}",       "dataset": "outbound"},
        "choice_temporal_model":   {"formula": f"spike_count ~ choice + {tp} + {spd}",        "dataset": "outbound"},
    }


def _result_converged(res):
    """Return a robust convergence flag across IRLS and optimizer-based fits."""
    converged = getattr(res, "converged", None)
    if converged is not None:
        return bool(converged)
    mle_retvals = getattr(res, "mle_retvals", None)
    if isinstance(mle_retvals, dict) and "converged" in mle_retvals:
        return bool(mle_retvals["converged"])
    return True


def _result_is_finite(res):
    """Return True only for numerically valid fitted results."""
    params = np.asarray(getattr(res, "params", []), dtype=float)
    llf = getattr(res, "llf", np.nan)
    deviance = getattr(res, "deviance", np.nan)
    return (
        np.isfinite(params).all()
        and np.isfinite(llf)
        and np.isfinite(deviance)
    )


def _fit_glm_with_fallback(formula, data, family=None):
    """Fit a GLM strictly with IRLS and reject numerically bad results."""
    family = family or sm.families.Poisson()
    model = smf.glm(formula, data=data, family=family)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = model.fit(disp=False)

    if not _result_converged(res):
        raise RuntimeError("IRLS did not converge")
    if not _result_is_finite(res):
        raise RuntimeError("IRLS returned a non-finite result")

    res._codex_fit_method = "irls"
    return res


def model_csv_path(registry_key, base_dir=None, cfg=None):
    """Return the default CSV path for a registry key.

    Existing convention: ``speed_spline`` → ``speed_spline_model_all.csv``,
    but ``full_model`` → ``full_model_all.csv`` (no double "model").
    """
    cfg = cfg or CONFIG
    base_dir = base_dir or cfg["base_dir"]
    _OVERRIDES = {"null_outbound": "null_model_out_all"}
    if registry_key in _OVERRIDES:
        stem = _OVERRIDES[registry_key]
    elif registry_key.endswith("_model"):
        stem = f"{registry_key}_all"
    else:
        stem = f"{registry_key}_model_all"
    return f"{base_dir}/analysis/{stem}.csv"


def resolve_model_name(model_name, fit_history=False, cfg=None):
    """Resolve model key/stem to history or non-history naming convention.

    Supports both registry keys (e.g. ``full_model``) and csv stems
    (e.g. ``full_model_all``). If ``fit_history`` is True and *model_name*
    is not already history-tagged, this returns the corresponding history
    name/stem when known.
    """
    if model_name is None or not fit_history:
        return model_name
    if "_history" in str(model_name):
        return model_name

    cfg = cfg or CONFIG
    registry = build_model_registry(cfg)

    # Build map for both registry keys and csv stems.
    mapping = {}
    for key in registry.keys():
        hist_key = f"{key}_history"
        base_stem = os.path.splitext(os.path.basename(model_csv_path(key, cfg=cfg)))[0]
        hist_stem = os.path.splitext(os.path.basename(model_csv_path(hist_key, cfg=cfg)))[0]
        mapping[key] = hist_key
        mapping[base_stem] = hist_stem

    if model_name in mapping:
        return mapping[model_name]

    # Conservative fallback for unknown names.
    return f"{model_name}_history"


# 1. Data preparation 

def interp_col(col_values, times, bin_centers):
    if pd.api.types.is_numeric_dtype(col_values):
        vals = col_values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() < 2:
            return np.full(len(bin_centers), np.nan)
        # Exclude NaN anchor points — np.interp propagates NaN from any bracketing point
        return np.interp(bin_centers, times[valid], vals[valid], left=np.nan, right=np.nan)
    else:
        # ffill: use the last timestamp <= bin_center (right-1), not the next one
        idx = (np.searchsorted(times, bin_centers, side='right') - 1).clip(0, len(times) - 1)
        return col_values.iloc[idx].values


def _segment_position_bounds(position_df):
    """Return per-segment linear-position bounds from the source position table."""
    seg_bounds = (
        position_df.groupby("track_segment_id")["linear_position"]
        .agg(["min", "max"])
        .to_dict("index")
    )
    return {int(seg): {"min": vals["min"], "max": vals["max"]} for seg, vals in seg_bounds.items()}


def _add_branch_position_columns(cov_df, seg_bounds):
    """Add branch-aware labels and outward-from-junction branch coordinates."""
    cov_df = cov_df.copy()
    segment_ids = cov_df["track_segment_id"].astype("Int64")
    seg_min = segment_ids.map({seg: bounds["min"] for seg, bounds in seg_bounds.items()}).astype(float)
    seg_max = segment_ids.map({seg: bounds["max"] for seg, bounds in seg_bounds.items()}).astype(float)

    # ``linear_position`` is interpolated independently of ``track_segment_id``.
    # Around graph transitions this can produce impossible row combinations
    # (e.g. a segment label from one arm with a position from another).  For the
    # branch-aware basis, enforce segment-consistent support before computing the
    # outward-from-junction coordinate.
    linear_pos_clipped = cov_df["linear_position"].clip(lower=seg_min, upper=seg_max)

    branch_id = pd.Series(pd.NA, index=cov_df.index, dtype="object")
    branch_pos_cm = pd.Series(np.nan, index=cov_df.index, dtype=float)

    for branch_name, spec in BRANCH_SPECS.items():
        branch_mask = segment_ids.isin(spec["segments"])
        if not branch_mask.any():
            continue

        origin_bounds = seg_bounds[spec["origin_segment"]]
        if spec["direction"] > 0:
            origin_pos = origin_bounds["min"]
            branch_pos_cm.loc[branch_mask] = (
                linear_pos_clipped.loc[branch_mask] - origin_pos
            )
        else:
            origin_pos = origin_bounds["max"]
            branch_pos_cm.loc[branch_mask] = (
                origin_pos - linear_pos_clipped.loc[branch_mask]
            )
        branch_id.loc[branch_mask] = branch_name

    if branch_id.isna().any():
        missing_segments = sorted(segment_ids[branch_id.isna()].dropna().unique().tolist())
        raise ValueError(
            f"Could not assign branch_id for track_segment_id values: {missing_segments}"
        )

    cov_df["branch_id"] = pd.Categorical(
        branch_id, categories=list(BRANCH_SPECS.keys()), ordered=True
    )
    cov_df["branch_pos_cm"] = branch_pos_cm
    cov_df["linear_position_branchsafe"] = linear_pos_clipped

    branch_max = cov_df.groupby("branch_id", observed=True)["branch_pos_cm"].transform("max")
    cov_df["branch_pos_scaled"] = np.where(branch_max > 0, branch_pos_cm / branch_max, 0.0)
    return cov_df


def load_and_prepare_data(cfg=None):
    """Load position + spikes, bin, filter, scale.

    Returns a dict with keys:
        cov_df_common, cov_df_out_common,
        spike_counts_common, spike_counts_out_common,
        unit_ids, bin_centers,
        scaling_params  — dict with pos_min, pos_max, speed_min, speed_max
    """
    cfg = cfg or CONFIG
    base_dir = cfg["base_dir"]
    bin_size = cfg["bin_size"]

    trialized_position = pd.read_csv(cfg["trialized_position"], index_col="time")
    seg_bounds = _segment_position_bounds(trialized_position)
    data = np.load(f"{base_dir}/analysis/final_spikes/mfpc_spikes.npz",
                   allow_pickle=True)
    mpfc_spikes = [data[f"arr_{i}"] for i in range(len(data.files))]

    bin_edges = np.arange(
        trialized_position.index.min(),
        trialized_position.index.max() + bin_size,
        bin_size,
    )
    bin_centers = bin_edges[:-1] + bin_size / 2
    spike_counts = np.array(
        [np.histogram(s, bins=bin_edges)[0] for s in mpfc_spikes]
    )

    # interpolate covariates to 2ms bins
    skip_cols = {"video_frame_ind"}
    ffill_cols = {
        "track_segment_id", "trial_number", "trial_start", "trial_end",
        "trial_duration (s)", "trial_type", "left/right", "epoch",
    }
    cols_to_interp = [c for c in trialized_position.columns
                      if c not in skip_cols]
    times = trialized_position.index.astype(float).values

    def _interp(col_name):
        col_values = trialized_position[col_name]
        if col_name in ffill_cols:
            # forward-fill for categorical columns stored as int
            idx = (np.searchsorted(times, bin_centers, side='right') - 1).clip(0, len(times) - 1)
            return col_values.iloc[idx].values
        return interp_col(col_values, times, bin_centers)

    interpolated = {col: _interp(col) for col in cols_to_interp}
    interp_pos = pd.DataFrame(interpolated, columns=cols_to_interp)
    interp_pos.insert(0, "time_bin_center", bin_centers)

    # NaN out bins that fall outside any epoch's time range
    epoch_ranges = trialized_position.groupby("epoch").apply(
        lambda g: (g.index.min(), g.index.max()))
    in_epoch = np.zeros(len(bin_centers), dtype=bool)
    for t_start, t_end in epoch_ranges.values:
        in_epoch |= (bin_centers >= t_start) & (bin_centers <= t_end)
    if not in_epoch.all():
        interp_pos.loc[~in_epoch] = np.nan

    # base mask: run zone, outbound + inbound
    base_mask = (
        (interp_pos["zone"] == "run")
        & (interp_pos["trial_type"].isin(["outbound", "inbound"]))
    )
    cov_df = interp_pos[base_mask].rename(columns={"left/right": "choice"})
    spike_counts_masked = spike_counts[:, base_mask]
    unit_ids = np.arange(len(spike_counts_masked))

    # common mask: speed + position filters
    common_mask = (
        cov_df["speed"].notna()
        & (cov_df["speed"] > cfg["speed_min"])
        & (cov_df["speed"] < cfg["speed_max"])
        & cov_df["linear_position"].notna()
    )
    cov_df_common = cov_df[common_mask].copy()
    spike_counts_common = spike_counts_masked[:, common_mask]
    cov_df_common = _add_branch_position_columns(cov_df_common, seg_bounds)

    speed_min_val = cov_df_common["speed"].min()
    speed_max_val = cov_df_common["speed"].max()
    pos_min_val   = cov_df_common["linear_position"].min()
    pos_max_val   = cov_df_common["linear_position"].max()

    cov_df_common["speed_scaled"] = (
        (cov_df_common["speed"] - speed_min_val) / (speed_max_val - speed_min_val)
    )
    cov_df_common["pos_scaled"] = (
        (cov_df_common["linear_position"] - pos_min_val) / (pos_max_val - pos_min_val)
    )

    # outbound-only subset
    outbound_common_mask = common_mask & (cov_df["trial_type"] == "outbound")
    cov_df_out_common = cov_df[outbound_common_mask].copy()
    spike_counts_out_common = spike_counts_masked[:, outbound_common_mask]
    cov_df_out_common["speed_scaled"] = (
        (cov_df_out_common["speed"] - speed_min_val) / (speed_max_val - speed_min_val)
    )
    cov_df_out_common["pos_scaled"] = (
        (cov_df_out_common["linear_position"] - pos_min_val) / (pos_max_val - pos_min_val)
    )
    cov_df_out_common = _add_branch_position_columns(cov_df_out_common, seg_bounds)

    # correct trials: outbound + inbound, no zone or speed filter
    correct_mask = interp_pos["trial_type"].isin(["outbound", "inbound"])
    cov_df_correct = interp_pos[correct_mask].rename(columns={"left/right": "choice"})
    spike_counts_correct = spike_counts[:, correct_mask]

    return dict(
        cov_df_correct       = cov_df_correct,
        spike_counts_correct = spike_counts_correct,
        cov_df_common        = cov_df_common,
        cov_df_out_common    = cov_df_out_common,
        spike_counts_common  = spike_counts_common,
        spike_counts_out_common = spike_counts_out_common,
        unit_ids             = unit_ids,
        bin_centers          = bin_centers,
        scaling_params       = dict(
            pos_min  = pos_min_val,
            pos_max  = pos_max_val,
            speed_min = speed_min_val,
            speed_max = speed_max_val,
        ),
    )


def bin_and_average(values, bin_by, n_bins=50):
    """Bin ``values`` by ``bin_by`` and return (bin_centers, bin_means)."""
    edges = np.linspace(np.nanmin(bin_by), np.nanmax(bin_by), n_bins + 1)
    idx = np.clip(np.digitize(bin_by, edges) - 1, 0, n_bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2
    means = np.array([values[idx == b].mean() if (idx == b).any() else np.nan
                       for b in range(n_bins)])
    return centers, means


def load_model_outputs(model_name, base_dir=None, cfg=None, fit_history=False):
    """Load CSV diagnostics + NPZ residual profiles for a fitted model.

    Accepts either registry keys (e.g. ``"full_model"``) or csv stems
    (e.g. ``"full_model_all"``), with or without history resolution.

    Returns (diag_df, profiles_dict).
    """
    cfg = cfg or CONFIG
    base_dir = base_dir or cfg["base_dir"]
    resolved = resolve_model_name(model_name, fit_history=fit_history, cfg=cfg)
    resolved_stem = os.path.splitext(os.path.basename(str(resolved)))[0]

    # Primary diagnostics stem follows model_csv_path(...) naming (e.g. *_model_all).
    if resolved_stem.endswith("_all"):
        primary_stem = resolved_stem
    else:
        primary_stem = os.path.splitext(
            os.path.basename(model_csv_path(resolved_stem, base_dir=base_dir, cfg=cfg))
        )[0]

    # Backward-compatible fallback for older artifacts named without *_all.
    stems = [primary_stem]
    if resolved_stem not in stems:
        stems.append(resolved_stem)

    attempted = []
    for stem in stems:
        diag_path = f"{base_dir}/analysis/diagnostics_{stem}.csv"
        npz_path = f"{base_dir}/analysis/residual_profiles_{stem}.npz"
        attempted.append((diag_path, npz_path))
        if os.path.exists(diag_path) and os.path.exists(npz_path):
            diag = pd.read_csv(diag_path, index_col=0)
            raw = np.load(npz_path, allow_pickle=True)
            profiles = {k: raw[k] for k in raw.files}
            return diag, profiles

    attempts_text = "\n".join(
        [f"  diagnostics: {d}\n  profiles:    {p}" for d, p in attempted]
    )
    raise FileNotFoundError(
        f"Could not find diagnostics/profile outputs for model '{model_name}' "
        f"(resolved '{resolved_stem}'). Tried:\n{attempts_text}"
    )


# 2. GLM fitting

def make_cv_trial_folds(trial_ids,
                        n_folds=5,
                        seed=42):
    """Return a list of train/test trial id splits for k-fold CV.

    Parameters
    ----------
    trial_ids : array-like
    n_folds : int
    seed : int

    Returns
    -------
    list of dicts: [{"train": [...], "test": [...]}]
    """
    trial_ids = np.array(trial_ids)
    rng = np.random.default_rng(seed)
    trial_ids_shuffled = trial_ids.copy()
    rng.shuffle(trial_ids_shuffled)
    split = np.array_split(trial_ids_shuffled, n_folds)
    results = []
    for i in range(n_folds):
        test = split[i].flatten().tolist()
        train = np.concatenate([chunk for idx, chunk in enumerate(split) if idx != i]).tolist()
        results.append({"train": train, "test": test})
    return results


def fit_glm_cv(df: pd.DataFrame,
               formula: str,
               fold: dict,
               spike_counts_masked,
               per_unit_transform,
               fit_history):
    pass


def _fit_unit_task(i, uid, cov_df, spike_counts_row, formula, per_unit_transform,
                   deep_diagnostics, d_cov_cols, d_cat_cols,
                   bin_edges, bin_centers, cat_labels, diagnostics_n_bins,
                   eval_cov_df=None, eval_spike_counts_row=None):
    """Fit one unit; returns (fit_row, diag_row, cont_profiles, cat_profiles, valid_id).

    Module-level so joblib can pickle it for subprocess dispatch.
    Called by fit_glm_all_units when n_jobs != 1.
    """
    df = cov_df.copy()
    df["spike_count"] = spike_counts_row
    if per_unit_transform is not None:
        df = per_unit_transform(df, spike_counts_row, i)
    try:
        res = _fit_glm_with_fallback(formula, df, family=sm.families.Poisson())
        fit_row = dict(
            unit=uid, aic=res.aic, llf=res.llf, deviance=res.deviance,
            n_params=len(res.params), n_obs=int(res.nobs),
            converged=_result_converged(res),
            coef=res.params.to_dict(), bse=res.bse.to_dict(),
            deviance_null=res.null_deviance, df_model=res.df_model,
        )
        try:
            if eval_cov_df is not None:
                eval_df = eval_cov_df.copy()
                eval_df["spike_count"] = eval_spike_counts_row
                if per_unit_transform is not None:
                    eval_df = per_unit_transform(eval_df, eval_spike_counts_row, i)
                test_X = build_design_matrices([res.model.data.design_info], eval_df)[0]
                y_test = np.asarray(eval_df["spike_count"])
                mu_test = res.predict(exog=np.asarray(test_X))
                fit_row["llf_eval"] = res.model.family.loglike(y_test, mu_test)
                fit_row["n_spikes_eval"] = int(y_test.sum())
            else:
                fit_row["llf_eval"] = np.nan
                fit_row["n_spikes_eval"] = np.nan
        except Exception as e:
            warnings.warn(str(e))
            fit_row["llf_eval"] = np.nan
            fit_row["n_spikes_eval"] = np.nan

        if not deep_diagnostics:
            return fit_row, None, None, None, None

        predicted = np.asarray(res.predict())
        observed = np.asarray(res.model.endog)
        raw = observed - predicted
        n_spikes = int((observed > 0).sum())
        cumresid = np.cumsum(raw)
        drift_auc = np.abs(cumresid).mean() / max(n_spikes, 1)
        z_unsorted = _compute_z_unsorted(predicted, observed)
        if len(z_unsorted) >= 4:
            z_sorted = np.sort(z_unsorted)
            n = len(z_sorted)
            ecdf = np.arange(1, n + 1) / n
            ks_D = float(np.max(np.abs(ecdf - z_sorted)))
            z_autocorr, _ = spearmanr(z_unsorted[:-1], z_unsorted[1:])
        else:
            ks_D, z_autocorr = np.nan, np.nan

        diag_row = dict(unit=uid, converged=True, ks_D=ks_D,
                        ks_z_autocorr=float(z_autocorr), drift_auc=drift_auc)
        ss_total = float(np.sum(raw ** 2))
        cov_df_fit = df.drop(columns=["spike_count"]).reset_index(drop=True)
        if len(cov_df_fit) != len(observed):
            cov_df_fit = cov_df_fit.iloc[-len(observed):].reset_index(drop=True)

        for col in d_cov_cols:
            if col not in cov_df_fit.columns:
                diag_row[f"resid_eta2_{col}"] = np.nan
                continue
            valid_mask = ~np.isnan(cov_df_fit[col].values)
            if valid_mask.sum() > 10 and ss_total > 0:
                cov_vals = cov_df_fit[col].values[valid_mask]
                res_vals = raw[valid_mask]
                bins = np.linspace(cov_vals.min(), cov_vals.max(), 51)
                bin_idx = np.clip(np.digitize(cov_vals, bins) - 1, 0, 49)
                ss_between = sum(
                    (res_vals[bin_idx == b].mean() ** 2) * (bin_idx == b).sum()
                    for b in range(50) if (bin_idx == b).any()
                )
                diag_row[f"resid_eta2_{col}"] = float(ss_between / ss_total)
            else:
                diag_row[f"resid_eta2_{col}"] = np.nan

        cont_profiles = {}
        for col in bin_edges:
            if col not in cov_df_fit.columns:
                cont_profiles[col] = None
                continue
            valid_mask = ~np.isnan(cov_df_fit[col].values)
            cov_vals = cov_df_fit[col].values[valid_mask]
            res_vals = raw[valid_mask]
            idx = np.clip(np.digitize(cov_vals, bin_edges[col]) - 1, 0, diagnostics_n_bins - 1)
            cont_profiles[col] = np.array([
                res_vals[idx == b].mean() if (idx == b).any() else np.nan
                for b in range(diagnostics_n_bins)
            ])

        cat_profiles = {}
        for col in cat_labels:
            if col not in cov_df_fit.columns:
                cat_profiles[col] = None
                continue
            cat_vals = cov_df_fit[col].values
            cat_profiles[col] = np.array([
                raw[cat_vals == c].mean() if (cat_vals == c).any() else np.nan
                for c in cat_labels[col]
            ])

        return fit_row, diag_row, cont_profiles, cat_profiles, uid

    except Exception as e:
        fit_row = dict(
            unit=uid, aic=np.nan, llf=np.nan, deviance=np.nan,
            n_params=np.nan, n_obs=np.nan, converged=False,
            coef=None, bse=None, deviance_null=np.nan,
            df_model=np.nan, error=str(e), llf_eval=np.nan,
            n_spikes_eval=np.nan,
        )
        if not deep_diagnostics:
            return fit_row, None, None, None, None
        diag_row = dict(unit=uid, converged=False,
                        ks_D=np.nan, ks_z_autocorr=np.nan, drift_auc=np.nan)
        for col in d_cov_cols:
            diag_row[f"resid_eta2_{col}"] = np.nan
        return fit_row, diag_row, None, None, None


def fit_glm_all_units(formula: str,
                      cov_df: pd.DataFrame,
                      spike_counts_masked: np.array,
                      unit_ids: np.array,
                      bin_size = 0.002,
                      per_unit_transform = None,
                      save_path = None,
                      model_name = None,
                      fit_history = False,
                      deep_diagnostics = False,
                      diagnostics_base_dir = None,
                      diagnostics_model_name = None,
                      diagnostics_covariate_cols = None,
                      diagnostics_categorical_cols = None,
                      diagnostics_n_bins = 50,
                      return_diagnostics = False,
                      refit = False,
                      n_jobs = -1,
                      eval_cov_df = None,
                      eval_spike_counts_masked = None):
    """Fit a Poisson GLM for each unit and collect summary statistics.

    Parameters
    ----------
    n_jobs : int
        Number of parallel worker processes for joblib.Parallel.
        1 (default) runs the original sequential loop unchanged.
        -1 uses all available CPU cores via _fit_unit_task workers.
    per_unit_transform : callable, optional
        ``fn(df, spike_counts_1d, unit_index) -> df``.  Called after adding
        ``spike_count`` to *df*.  Use this to inject unit-specific covariates
        (e.g. spike history).  The returned DataFrame may be shorter than the
        input (rows with NaN history bins are typically dropped).
    save_path : str, optional
        CSV path for caching.  When the file exists and *refit* is False,
        read from disk instead of fitting.  Results are saved here after fitting.
    model_name : str, optional
        Registry key (e.g. ``"speed_spline"``).  If *save_path* is not given,
        the default CSV path is derived via ``model_csv_path(model_name)``.
    fit_history : bool
        If True, resolve *model_name* to the history-tagged cache name when
        deriving/reading the default CSV path (e.g. ``full_model`` →
        ``full_model_history``).
    deep_diagnostics : bool
        If True, compute scalar diagnostics and residual profiles in the same
        fitting pass and save them to diagnostics/residual_profiles files.
    diagnostics_base_dir : str, optional
        Base directory for diagnostics outputs. Defaults to CONFIG["base_dir"].
    diagnostics_model_name : str, optional
        Stem used for diagnostics filenames. Defaults to fit CSV stem.
    diagnostics_covariate_cols : list[str], optional
        Continuous covariate columns for residual η² and profile panels.
    diagnostics_categorical_cols : list[str], optional
        Categorical columns for residual profile panels.
    diagnostics_n_bins : int
        Number of bins for continuous residual profiles.
    return_diagnostics : bool
        If True and *deep_diagnostics* is enabled, return
        ``(fit_df, diag_df, profiles_dict)``.
    refit : bool
        If True, always fit even when the cached CSV exists on disk.
    """
    resolved_model_name = resolve_model_name(model_name, fit_history=fit_history, cfg=CONFIG)
    if save_path is None and resolved_model_name is not None:
        save_path = model_csv_path(resolved_model_name)

    diag_df = None
    profiles = None
    diag_csv_path = None
    diag_npz_path = None
    if deep_diagnostics:
        dbase = diagnostics_base_dir or CONFIG["base_dir"]
        if diagnostics_model_name is None:
            if save_path is not None:
                diag_stem = os.path.splitext(os.path.basename(save_path))[0]
            else:
                diag_stem = resolved_model_name if resolved_model_name is not None else "model"
        else:
            diag_stem = diagnostics_model_name
        diag_csv_path = f"{dbase}/analysis/diagnostics_{diag_stem}.csv"
        diag_npz_path = f"{dbase}/analysis/residual_profiles_{diag_stem}.npz"

    # Cache short-circuit:
    # - params-only mode: existing fit csv is enough
    # - deep diagnostics mode: require all artifacts, otherwise refit once
    if save_path and not refit and os.path.exists(save_path):
        if not deep_diagnostics or (diag_csv_path and diag_npz_path and os.path.exists(diag_csv_path) and os.path.exists(diag_npz_path)):
            print(f"Loading cached fit from {save_path}")
            cached = pd.read_csv(save_path, index_col=0)
            if resolved_model_name is not None and "model" not in cached.columns:
                cached["model"] = resolved_model_name
            if deep_diagnostics and return_diagnostics:
                diag_df = pd.read_csv(diag_csv_path, index_col=0)
                raw_npz = np.load(diag_npz_path, allow_pickle=True)
                profiles = {k: raw_npz[k] for k in raw_npz.files}
                return cached, diag_df, profiles
            return cached
        print("Fit cache exists but diagnostics cache missing; refitting once to compute diagnostics.")

    # Pre-compute profile binning metadata if deep diagnostics requested.
    d_cov_cols = diagnostics_covariate_cols or []
    d_cat_cols = diagnostics_categorical_cols or []
    bin_edges = {}
    bin_centers = {}
    cat_labels = {}
    if deep_diagnostics:
        for col in d_cov_cols:
            if col not in cov_df.columns:
                continue
            valid = cov_df[col].dropna().values
            if len(valid) == 0:
                continue
            edges = np.linspace(valid.min(), valid.max(), diagnostics_n_bins + 1)
            bin_edges[col] = edges
            bin_centers[col] = (edges[:-1] + edges[1:]) / 2
        for col in d_cat_cols:
            if col in cov_df.columns:
                vals = cov_df[col].dropna().unique()
                cat_labels[col] = sorted(vals)

    diag_rows = []
    profile_rows = {col: [] for col in bin_edges}
    cat_profile_rows = {col: [] for col in cat_labels}
    valid_prof_ids = []

    rows = []
    desc = resolved_model_name or "fitting"
    if n_jobs == 1:
        # Original sequential loop — unchanged behaviour.
        for i, uid in tqdm(enumerate(unit_ids), total=len(unit_ids), desc=desc, unit="unit"):
            df = cov_df.copy()
            df["spike_count"] = spike_counts_masked[i]  # pre-masked counts
            if per_unit_transform is not None:
                df = per_unit_transform(df, spike_counts_masked[i], i)
            try:
                res = _fit_glm_with_fallback(formula, df, family=sm.families.Poisson())
                rows.append(dict(
                    unit=uid,
                    aic=res.aic,
                    llf=res.llf,
                    deviance=res.deviance,
                    n_params=len(res.params),
                    n_obs=int(res.nobs),
                    converged=_result_converged(res),
                    coef=res.params.to_dict(),
                    bse=res.bse.to_dict(),
                    deviance_null = res.null_deviance,
                    df_model = res.df_model
                ))

                if deep_diagnostics:
                    predicted = np.asarray(res.predict())
                    observed = np.asarray(res.model.endog)
                    raw = observed - predicted
                    n_spikes = int((observed > 0).sum())

                    # Cumulative residual drift.
                    cumresid = np.cumsum(raw)
                    drift_auc = np.abs(cumresid).mean() / max(n_spikes, 1)

                    # KS D and z autocorrelation.
                    z_unsorted = _compute_z_unsorted(predicted, observed)
                    if len(z_unsorted) >= 4:
                        z_sorted = np.sort(z_unsorted)
                        n = len(z_sorted)
                        ecdf = np.arange(1, n + 1) / n
                        ks_D = float(np.max(np.abs(ecdf - z_sorted)))
                        z_autocorr, _ = spearmanr(z_unsorted[:-1], z_unsorted[1:])
                    else:
                        ks_D, z_autocorr = np.nan, np.nan

                    row = dict(
                        unit=uid, converged=True, ks_D=ks_D,
                        ks_z_autocorr=float(z_autocorr), drift_auc=drift_auc
                    )
                    ss_total = float(np.sum(raw ** 2))

                    # Use aligned covariates corresponding to fitted rows.
                    cov_df_fit = df.drop(columns=["spike_count"]).reset_index(drop=True)
                    if len(cov_df_fit) != len(observed):
                        cov_df_fit = cov_df_fit.iloc[-len(observed):].reset_index(drop=True)

                    # Residual eta2 per continuous covariate.
                    for col in d_cov_cols:
                        if col not in cov_df_fit.columns:
                            row[f"resid_eta2_{col}"] = np.nan
                            continue
                        valid_mask = ~np.isnan(cov_df_fit[col].values)
                        if valid_mask.sum() > 10 and ss_total > 0:
                            cov_vals = cov_df_fit[col].values[valid_mask]
                            res_vals = raw[valid_mask]
                            bins = np.linspace(cov_vals.min(), cov_vals.max(), 51)
                            bin_idx = np.clip(np.digitize(cov_vals, bins) - 1, 0, 49)
                            ss_between = sum(
                                (res_vals[bin_idx == b].mean() ** 2) * (bin_idx == b).sum()
                                for b in range(50) if (bin_idx == b).any()
                            )
                            row[f"resid_eta2_{col}"] = float(ss_between / ss_total)
                        else:
                            row[f"resid_eta2_{col}"] = np.nan

                    diag_rows.append(row)

                    # Continuous residual profiles.
                    for col in bin_edges:
                        if col not in cov_df_fit.columns:
                            continue
                        valid_mask = ~np.isnan(cov_df_fit[col].values)
                        cov_vals = cov_df_fit[col].values[valid_mask]
                        res_vals = raw[valid_mask]
                        idx = np.clip(np.digitize(cov_vals, bin_edges[col]) - 1, 0, diagnostics_n_bins - 1)
                        means = np.array([
                            res_vals[idx == b].mean() if (idx == b).any() else np.nan
                            for b in range(diagnostics_n_bins)
                        ])
                        profile_rows[col].append(means)

                    # Categorical residual profiles.
                    for col in cat_labels:
                        if col not in cov_df_fit.columns:
                            continue
                        cat_vals = cov_df_fit[col].values
                        means = np.array([
                            raw[cat_vals == c].mean() if (cat_vals == c).any() else np.nan
                            for c in cat_labels[col]
                        ])
                        cat_profile_rows[col].append(means)

                    valid_prof_ids.append(uid)

            except Exception as e:
                rows.append(dict(
                    unit=uid, aic=np.nan, llf=np.nan, deviance=np.nan,
                    n_params=np.nan, n_obs=np.nan, converged=False,
                    coef=None, bse=None, deviance_null=np.nan,
                    df_model=np.nan, error=str(e)
                ))
                if deep_diagnostics:
                    row = dict(unit=uid, converged=False,
                               ks_D=np.nan, ks_z_autocorr=np.nan, drift_auc=np.nan)
                    for col in d_cov_cols:
                        row[f"resid_eta2_{col}"] = np.nan
                    diag_rows.append(row)

    else:
        # Parallel path: dispatch one worker process per unit via joblib.
        # _fit_unit_task is module-level so it can be pickled by loky.
        # return_as='generator_unordered' lets tqdm update as each worker finishes,
        # then we sort by input order to keep results aligned with unit_ids.
        raw_gen = Parallel(n_jobs=n_jobs, prefer='processes', return_as='generator_unordered')(
            delayed(_fit_unit_task)(
                i, uid, cov_df, spike_counts_masked[i], formula, per_unit_transform,
                deep_diagnostics, d_cov_cols, d_cat_cols,
                bin_edges, bin_centers, cat_labels, diagnostics_n_bins,
                eval_cov_df=eval_cov_df,
                eval_spike_counts_row=eval_spike_counts_masked[i] if eval_spike_counts_masked is not None else None,
            )
            for i, uid in enumerate(unit_ids)
        )
        parallel_results = list(tqdm(raw_gen, total=len(unit_ids), desc=desc, unit="unit"))
        # restore original order (generator_unordered returns as workers finish)
        parallel_results.sort(key=lambda t: list(unit_ids).index(t[0]["unit"]))
        for fit_row, diag_row, cont_profiles, cat_profiles, valid_id in parallel_results:
            rows.append(fit_row)
            if deep_diagnostics:
                diag_rows.append(diag_row)
                if valid_id is not None:
                    for col in bin_edges:
                        if cont_profiles and cont_profiles.get(col) is not None:
                            profile_rows[col].append(cont_profiles[col])
                    for col in cat_labels:
                        if cat_profiles and cat_profiles.get(col) is not None:
                            cat_profile_rows[col].append(cat_profiles[col])
                    valid_prof_ids.append(valid_id)

    result = pd.DataFrame(rows)
    if resolved_model_name is not None:
        result["model"] = resolved_model_name
    if save_path:
        result.to_csv(save_path)
        print(f"Saved {len(result)} units : {save_path}")

    if deep_diagnostics:
        diag_df = pd.DataFrame(diag_rows)
        profiles = {"unit_ids": np.array(valid_prof_ids)}
        for col in bin_edges:
            profiles[f"{col}_profiles"] = np.array(profile_rows[col])
            profiles[f"{col}_centers"] = bin_centers[col]
        for col in cat_labels:
            profiles[f"{col}_cat_profiles"] = np.array(cat_profile_rows[col])
            profiles[f"{col}_cat_labels"] = np.array(cat_labels[col], dtype=object)

        if diag_csv_path is not None and diag_npz_path is not None:
            diag_df.to_csv(diag_csv_path)
            print(f"Saved → {diag_csv_path}")
            np.savez(diag_npz_path, **profiles)
            print(f"Saved → {diag_npz_path}")

        if return_diagnostics:
            return result, diag_df, profiles

    return result


def fit_single_unit(formula, cov_df, spike_counts, unit_idx,
                    family=None, per_unit_transform=None):
    """Fit one unit interactively and return the GLMResults object."""
    if family is None:
        family = sm.families.Poisson()
    df = cov_df.copy()
    df["spike_count"] = spike_counts[unit_idx]
    if per_unit_transform is not None:
        df = per_unit_transform(df, spike_counts[unit_idx], unit_idx)
    return _fit_glm_with_fallback(formula, df, family=family)


def add_spike_history(df, spike_counts_unit, unit_idx,
                      windows_ms=((0, 2), (2, 10), (10, 20), (20, 50)),
                      bin_size=0.002):
    """Add windowed spike-history covariates for one unit.

    For each (lo_ms, hi_ms) window, counts spikes in the interval
    (t - hi, t - lo] and stores the result as ``hist_{lo}_{hi}ms``.
    Rows where any history window is undefined (edge bins) are dropped.

    Designed as a ``per_unit_transform`` callback for ``fit_glm_all_units``.
    """
    for lo_ms, hi_ms in windows_ms:
        lo_bins = int(lo_ms / (bin_size * 1000))
        hi_bins = int(hi_ms / (bin_size * 1000))
        cs = np.concatenate([[0], np.cumsum(spike_counts_unit)])
        col = np.full(len(df), np.nan)
        valid = np.arange(hi_bins, len(df))
        col[valid] = cs[valid - lo_bins] - cs[valid - hi_bins]
        df[f"hist_{lo_ms}_{hi_ms}ms"] = col
    return df.dropna()


# 3. Drop-one analysis

def build_reduced_formula(spec: dict,
                         drop_term: str):
    included_terms = [term for term in spec["terms"] if term != drop_term]
    return spec["formula_lhs"] + "~" + " + ".join(included_terms)


def fit_drop_one(model_name: str,
                 spec: dict,
                 drop_term: str,
                 base_dir: str,
                 unit_ids: list,
                 n_jobs: int = -1):

    formula = build_reduced_formula(spec, drop_term)
    drop_term_safe = re.sub(r'[^\w]', '_', drop_term).strip('_')

    res = fit_glm_all_units(
        formula, spec["cov_df"], spec["spike_counts"], unit_ids,
        per_unit_transform=spec.get("per_unit_transform"),
        n_jobs=n_jobs,
    )

    res["model"] = model_name
    res["dropped_term"] = drop_term

    res.to_csv(f"{base_dir}/analysis/{model_name}_drop_{drop_term_safe}.csv")

    n_converged = res["converged"].sum()
    print(f"{model_name} drop={drop_term} | {len(spec['cov_df']):,} bins × {len(unit_ids)} units | {n_converged}/{len(unit_ids)} converged → saved")

    return res


def run_drop_one_suite(drop_one_specs: dict,
                       model_name: str,
                       base_dir: str,
                       unit_ids: list,
                       check_if_exists: bool = True,
                       n_jobs: int = -1):

    spec = drop_one_specs[model_name]
    print(f"\nRunning drop-one suite: {model_name}({len(spec['terms'])} fits)")

    result_list = []
    for drop_term in spec["terms"]:
        drop_term_safe = re.sub(r'[^\w]', '_', drop_term).strip('_')
        out_path = f"{base_dir}/analysis/{model_name}_drop_{drop_term_safe}.csv"

        if check_if_exists and os.path.exists(out_path):
            print(f"    Skipping {drop_term} - file exists")
            result_list.append(pd.read_csv(out_path, index_col=0))
            continue

        result = fit_drop_one(model_name=model_name,
                     spec=spec,
                     drop_term=drop_term,
                     base_dir=base_dir,
                     unit_ids=unit_ids,
                     n_jobs=n_jobs)
        result_list.append(result)

    return result_list


def _parse_formula_terms(formula):
    """Split a patsy formula's RHS into individual additive terms."""
    rhs = formula.split("~", 1)[1].strip()
    # Split on '+' that is NOT inside parentheses
    terms = []
    depth = 0
    current = []
    for ch in rhs:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '+' and depth == 0:
            terms.append("".join(current).strip())
            current = []
            continue
        current.append(ch)
    terms.append("".join(current).strip())
    return [t for t in terms if t]


def _infer_term_df(term, levels = None):
    """Extract degrees of freedom from a spline term string.

    Returns the df value for spline terms (bs/cr), or 1 for plain
    categorical/continuous terms.
    
    term: str, the term to scan
    levels: dict, maps column name to number of unique levels in data, e.g. {"branch_id": 3, "trial_type": 2, "choice": 2}
    """
    levels = levels or {}
    if " + " in term:
        return sum(_infer_term_df(t.strip(), levels=levels) for t in term.split(" + ")) # for compound terms
    has_interaction = ":" in term
    has_categorical = "C(" in term
    has_spline_df = "df" in term
    
    if has_interaction and has_categorical and has_spline_df:
        m = re.search(r'C\((\w+)\)', term)
        varname = m.group(1)
        
        m = re.search(r'df\s*=\s*(\d+)', term)
        spline_df = int(m.group(1))
        
        return spline_df*(levels.get(varname, 2) -1) # number of levels in varname - 1
    
    elif has_categorical and not has_spline_df:
        m = re.search(r'C\((\w+)\)', term)
        varname = m.group(1)
        return (levels.get(varname, 2) -1)
    
    elif has_spline_df:
        m = re.search(r'df\s*=\s*(\d+)', term)
        if m:
            return int(m.group(1))
        else:
            raise ValueError(f"term '{term}' contains 'df' but no 'df=N' pattern found")

    else:
        warnings.warn(f"term '{term}' defaulted to df=1")
        return 1


def make_drop_one_specs(datasets, model_names, base_dir=None,
                        registry=None, cfg=None,
                        fit_history=False, history_windows_ms=None,
                        include_history_in_drop_one=False,
                        term_groups = None,
                        levels = {"branch_id": 3,
                                  "trial_type": 2,
                                  "choice": 2}):
    """Build drop-one specs from the model registry.

    Parameters
    ----------
    datasets : dict
        ``{"common": (cov_df, spike_counts), "outbound": (cov_df, spike_counts)}``
    model_names : list of str
        Keys into *registry* for which to build drop-one specs.
    base_dir : str, optional
    registry : dict, optional — from ``build_model_registry()``.
    cfg : dict, optional — falls back to ``CONFIG``.
    fit_history: bool, whether history terms were fit
    history_windows_ms: list of (start_ms, end_ms), history windows in ms
    include_history_in_drop_one: bool, whether to include history terms in drop-one suite (default False)
    term_groups: list of lists    , group terms to be dropped together
    """
    print(f"Using variable levels {levels}: make sure this is correct")
    
    cfg = cfg or CONFIG
    base_dir = base_dir or cfg["base_dir"]
    if registry is None:
        registry = build_model_registry(cfg)

    history_transform = make_history_transform(
        windows_ms=history_windows_ms, bin_size=cfg["bin_size"]
    ) if fit_history else None

    def _run_name(name):
        return f"{name}_history" if fit_history else name

    specs = {}
    for name in model_names:
        entry = registry[name]
        run_name = _run_name(name)
        formula = entry["formula"]
        if fit_history:
            formula = make_history_formula(formula, windows_ms=history_windows_ms)
        ds_key  = entry["dataset"]
        terms   = _parse_formula_terms(formula)
        if fit_history and not include_history_in_drop_one:
            hist_names = set(history_term_names(history_windows_ms))
            terms = [t for t in terms if t not in hist_names]

        for group in (term_groups or []):
            if not set(group).issubset(terms):
                raise ValueError(f"term_group {group} not found in terms for model '{name}'")

            grouped_terms = " + ".join(group)
            terms = [term for term in terms if term not in group]
            terms.append(grouped_terms)
                
        cov_df, spike_counts = datasets[ds_key]
        model_stem = os.path.splitext(os.path.basename(model_csv_path(run_name, base_dir=base_dir, cfg=cfg)))[0]
        null_run_name = _run_name("null" if ds_key == "common" else "null_outbound")
        specs[model_stem] = {
            "formula_lhs": formula.split("~")[0].strip(),
            "terms": terms,
            "delta_df": {t: _infer_term_df(t, levels = levels) for t in terms},
            "cov_df": cov_df,
            "spike_counts": spike_counts,
            "null_csv": model_csv_path(null_run_name, base_dir=base_dir, cfg=cfg),
            "per_unit_transform": history_transform,
            "fit_history": bool(fit_history),
        }

    return specs


def compute_drop_one_lrt(model_name, base_dir, drop_one_specs):
    """
    Load full model + all drop-one CSVs for model_name.
    For each dropped term, compute per-unit LRT (full vs reduced), delta AIC,
    and partial McFadden pseudo-R² = lrt_stat / deviance_null.
    Only includes units where both full and reduced models converged.
    Returns a long-form DataFrame with one row per (unit, dropped_term).
    """
    spec = drop_one_specs[model_name]

    full = pd.read_csv(f"{base_dir}/analysis/{model_name}.csv", index_col=0)
    full = full.set_index("unit")

    rows = []
    for term in spec["terms"]:
        drop_term_safe = re.sub(r'[^\w]', '_', term).strip('_')
        reduced = pd.read_csv(
            f"{base_dir}/analysis/{model_name}_drop_{drop_term_safe}.csv", index_col=0
        )
        reduced = reduced.set_index("unit")

        lrt_df = spec["delta_df"][term]

        n_skipped = 0
        for uid in full.index:
            if not full.loc[uid, "converged"] or not reduced.loc[uid, "converged"]:
                n_skipped += 1
                continue

            llf_full    = full.loc[uid, "llf"]
            llf_reduced = reduced.loc[uid, "llf"]
            aic_full    = full.loc[uid, "aic"]
            aic_reduced = reduced.loc[uid, "aic"]

            lrt_stat     = 2 * (llf_full - llf_reduced)
            lrt_pval     = 1 - chi2.cdf(lrt_stat, lrt_df) if lrt_df > 0 else np.nan
            deviance_null = full.loc[uid, "deviance_null"]
            deviance_full = full.loc[uid, "deviance"]
            partial_r2   = lrt_stat / deviance_null if deviance_null > 0 else np.nan
            if deviance_null - deviance_full <= 0:
                fdl = np.nan
            else:
                fdl = lrt_stat/(deviance_null - deviance_full)

            rows.append(dict(
                unit         = uid,
                model        = model_name,
                dropped_term = term,
                lrt_stat     = lrt_stat,
                lrt_df       = lrt_df,
                lrt_pval     = lrt_pval,
                significant  = bool(lrt_pval < 0.05) if not np.isnan(lrt_pval) else False,
                delta_aic    = aic_full - aic_reduced,  # negative = full model better; kept for non-nested comparisons
                partial_r2   = partial_r2,               # N-invariant: lrt_stat / deviance_null,
                fraction_of_explained_deviance_lost = fdl                                
            ))

        if n_skipped:
            print(f"  [{model_name} drop={term}] skipped {n_skipped} units (non-converged)")

    return pd.DataFrame(rows)

def compute_single_var_vs_null(single_var_model_keys, base_dir,
                               full_model_key="full_model",
                               null_model_key="null",
                               use_history=False,
                               cfg = CONFIG,):

    def _key(k):
        return f"{k}_history" if use_history else k

    base_dir = cfg["base_dir"]
    null = pd.read_csv(model_csv_path(_key(null_model_key), base_dir = base_dir, cfg = cfg), index_col = 0)
    full = pd.read_csv(model_csv_path(_key(full_model_key), base_dir = base_dir, cfg = cfg), index_col=0)

    full = full.set_index("unit")
    null = null.set_index("unit")
    
    rows = []
    registry = build_model_registry(cfg = cfg)
    
    
    for key in single_var_model_keys:
        single_model = pd.read_csv(model_csv_path(_key(key), base_dir=base_dir, cfg = cfg), index_col=0)
        single_model = single_model.set_index("unit")
        formula = registry[key]["formula"]
        term = formula.split("~", 1)[1].strip()
        
        
        
        n_skipped = 0
        
        for uid in full.index:
            if not full.loc[uid, "converged"] or not single_model.loc[uid, "converged"] or not null.loc[uid, "converged"]:
                n_skipped+=1
                continue
            
            llf_single = single_model.loc[uid, "llf"]
            llf_null = null.loc[uid, "llf"]
            lrt_stat = 2*(llf_single - llf_null)
            lrt_df = single_model.loc[uid, "n_params"] - 1 # n_params_single - n_params_null
            
            lrt_pval = 1- chi2.cdf(lrt_stat, lrt_df)
            
            deviance_single = single_model.loc[uid, "deviance"]
            deviance_null = null.loc[uid, "deviance_null"]
            deviance_full = full.loc[uid, "deviance"]
            
            if (deviance_null - deviance_full) <= 0:
                fdc = np.nan
            else:
                fdc = (deviance_null - deviance_single)/(deviance_null-deviance_full) #fraction of explained deviance captured
            
            rows.append(dict(
                unit = uid,
                model = key,
                term = term,
                lrt_stat = lrt_stat,
                lrt_df = lrt_df,
                lrt_pval = lrt_pval,
                significant = bool(lrt_pval<0.05) if not np.isnan(lrt_pval) else False,
                fraction_of_explained_deviance_captured = fdc))
        
        if n_skipped:
            print(f"{key} skipped {n_skipped} units (not converged)")
             
    return pd.DataFrame(rows)       
        
        
def compute_term_redundancy(drop_one_results, add_one_results):
    #expects drop_one_results fitted to a SINGLE full model
    merged = pd.merge(drop_one_results, add_one_results, left_on=["unit", "dropped_term"], right_on= ["unit", "term"], how = "inner")
    merged["redundancy"] = merged["fraction_of_explained_deviance_captured"] - merged["fraction_of_explained_deviance_lost"]
    merged = merged.drop(columns = ["dropped_term", "lrt_stat_x", "lrt_stat_y", "lrt_pval_x", "lrt_pval_y", "lrt_df_x", "lrt_df_y", "significant_x", "significant_y", "delta_aic", "partial_r2"])
    return merged
    

    



def apply_fdr_correction(drop_one_results,
                         alpha = 0.05):
    df = drop_one_results.copy().reset_index(drop=True)
    df["significant_fdr"] = False

    for (model, term), group in df.groupby(["model", "dropped_term"]):
        valid = group["lrt_pval"].notna()
        pvals = group.loc[valid, "lrt_pval"].values
        
        if len(pvals) == 0:
            continue
        
        reject = multipletests(pvals, method = 'fdr_bh', alpha = alpha)[0]

        valid_idx = group.index[valid.values]
        df.loc[valid_idx, "significant_fdr"] = reject
        
    return df
    

def pairwise_term_comparison(drop_one_results: pd.DataFrame,
                             model_name: str,
                             term_labels = None,
                             alpha = 0.05):
    df = drop_one_results[drop_one_results["model"]==model_name]
    df_wide = df.pivot(index = "unit", columns = "dropped_term", values = "partial_r2")
    
    a_list = []
    b_list = []
    pvals_list = []
    for A, B in itertools.combinations(df_wide.columns, 2):
        columnA = df_wide[A]
        columnB = df_wide[B]
        valid = pd.concat([columnA, columnB], axis =1, join = "inner").dropna().index
        stat, pvals = wilcoxon(columnA.loc[valid], columnB.loc[valid], zero_method = "wilcox", alternative = "two-sided", method = "auto")
        a_list.append(A)
        b_list.append(B)
        pvals_list.append(pvals)
    
    pvals_dict = {"term 1": a_list, "term 2": b_list, "pval": pvals_list}
    pvals_df = pd.DataFrame(pvals_dict)
    
    reject, corrected_pvals, _, _ = multipletests(pvals_dict["pval"], method = "fdr_bh", alpha = alpha)
    
    pvals_df["pval"] = corrected_pvals
    terms = list(df_wide.columns)
    result = pd.DataFrame(np.nan, index = terms, columns = terms)
    for _, row in pvals_df.iterrows():
        result.loc[row["term 1"], row["term 2"]] = row["pval"]
        result.loc[row["term 2"], row["term 1"]] = row["pval"]
        
    return result


TERM_LABELS = {
    "trial_type": "trial_type",
    "choice":     "choice",
    "bs(pos_scaled, df = 8)":  "position",
    "C(branch_id) + bs(branch_pos_scaled, df = 6):C(branch_id)": "branch_position",
    "bs(speed_scaled, df = 4)": "speed",
}

EXCLUDE_TERMS = {
    "cr(trial_progress, df = 6, constraints = 'center')",
}


def plot_drop_one_summary(drop_one_results, pairwise_pvals=None,
                          heatmap_model="full_model_all", term_labels=None,
                          r2_thresholds=(0.01, 0.05, 0.10)):
    """3-panel drop-one summary.

    Panel A : % units where partial_r2 > threshold (N-invariant unique contribution)
    Panel B : partial_r2 violin per term (effect size distribution)
    Panel C : pairwise q-value heatmap with significance annotations

    r2_thresholds : tuple of partial R² thresholds for bar groups (default: 0.01, 0.05, 0.10).
    pairwise_pvals : output of pairwise_term_comparison() — optional, skips Panel C if None.
    """
    if term_labels is None:
        term_labels = TERM_LABELS

    df = drop_one_results[~drop_one_results["dropped_term"].isin(EXCLUDE_TERMS)].copy()
    df["term_label"] = df["dropped_term"].map(term_labels).fillna(df["dropped_term"])

    term_order = (df.groupby("term_label")["partial_r2"]
                    .median().sort_values(ascending=False).index.tolist())

    ncols = 3 if pairwise_pvals is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

 # A % units above partial R² threshold
    ax = axes[0]
    agg_order = (df.groupby("term_label")["partial_r2"]
                   .apply(lambda x: (x > r2_thresholds[0]).mean())
                   .sort_values(ascending=False).index.tolist())
    x = np.arange(len(agg_order))
    colors = ["steelblue", "tomato", "forestgreen"]
    w_total = 0.7
    w = w_total / len(r2_thresholds)
    for k, thresh in enumerate(r2_thresholds):
        pct = [
            (df[df["term_label"] == t]["partial_r2"] > thresh).mean() * 100
            for t in agg_order
        ]
        offset = (k - (len(r2_thresholds) - 1) / 2) * w
        ax.bar(x + offset, pct, w, color=colors[k % len(colors)],
               alpha=0.85, label=f"partial R² > {thresh:.2f}")
    ax.set_xticks(x); ax.set_xticklabels(agg_order, rotation=25, ha="right")
    ax.set_ylabel("% units"); ax.set_title("Unique contribution (partial R²)")
    ax.legend(fontsize=8); sns.despine(ax=ax)

 # B effect size violin
    ax = axes[1]
    sns.violinplot(data=df, x="partial_r2", y="term_label", order=term_order,
                   inner="box", cut=0, color="steelblue", ax=ax)
    ax.axvline(0, color="red", lw=1, ls="--", label="0")
    ax.set_xlabel("Partial R² (lrt_stat / null deviance)")
    ax.set_ylabel(""); ax.set_title("Effect size (partial R²)")
    ax.legend(fontsize=8); sns.despine(ax=ax)

 # C pairwise heatmap
    if pairwise_pvals is not None:
        ax = axes[2]
        pw  = pairwise_pvals.rename(index=term_labels, columns=term_labels)
        mat = pw.values.astype(float)
        im  = ax.imshow(mat, cmap="RdYlGn_r", vmin=0, vmax=0.1,
                        aspect="auto", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="FDR q-value", shrink=0.8)
        ax.set_xticks(range(len(pw.columns))); ax.set_xticklabels(pw.columns, rotation=30, ha="right")
        ax.set_yticks(range(len(pw.index)));   ax.set_yticklabels(pw.index)
        ax.set_title("Pairwise term comparison (Wilcoxon, FDR)")

        for i, row_vals in enumerate(mat):
            for j, val in enumerate(row_vals):
                if np.isnan(val):
                    continue
                symbol = "***" if val < 0.001 else "**" if val < 0.01 else "*" if val < 0.05 else ""
                ax.text(j, i, symbol, ha="center", va="center", fontsize=11)

        sns.despine(ax=ax)

    plt.suptitle("Drop-one analysis summary", fontsize=13)
    plt.tight_layout()






# 4. Diagnostics — residuals & KS test

def _align_to_model(results, spike_counts=None, cov_df=None):
    """Align spike_counts and cov_df to the rows the model was actually fitted on.

    When a per_unit_transform (e.g. add_spike_history) drops leading rows,
    the model sees fewer bins than the original arrays.  This helper detects
    the mismatch and tail-trims the inputs to match.

    Returns (observed, cov_df) — either or both may be None if not requested.
    """
    n_fit = int(results.nobs)
    observed = np.asarray(results.model.endog)

    if cov_df is not None:
        if len(cov_df) != n_fit:
            cov_df = cov_df.iloc[-n_fit:].reset_index(drop=True)
        else:
            cov_df = cov_df.copy().reset_index(drop=True)

    return observed, cov_df


def compute_residuals(results, spike_counts=None, cov_df=None):
    """
    Compute per-bin raw and cumulative residuals for a fitted Poisson GLM.

    raw_residual_i      = observed_i - predicted_i   (spike counts per bin)
    cumulative_residual = cumsum(raw_residual)        (integrated misfit)

    Returns a DataFrame with covariate columns (if cov_df given) plus:
      raw_residual        : observed minus E[N|X] per bin
      cumulative_residual : running cumulative sum of raw_residual over time

    Parameters
    ----------
    results      : fitted statsmodels GLM result object (res.predict() must work)
    spike_counts : ignored (kept for backward compatibility); observed counts
                   are read from results.model.endog so alignment is automatic.
    cov_df       : covariate DataFrame. If longer than the fitted model (e.g.
                   because add_spike_history dropped leading rows), it is
                   automatically tail-trimmed to match.
    """
    predicted = np.asarray(results.predict())
    observed, cov_df = _align_to_model(results, cov_df=cov_df)
    raw = observed - predicted

    if cov_df is not None:
        df = cov_df
    else:
        df = pd.DataFrame(index=range(len(predicted)))
    df["raw_residual"]        = raw
    df["cumulative_residual"] = np.cumsum(raw)
    return df


def _load_trialized_position():
    """Load trialized_position CSV once and cache at module level."""
    path = CONFIG["trialized_position"]
    if not hasattr(_load_trialized_position, "_cache") or _load_trialized_position._path != path:
        _load_trialized_position._cache = pd.read_csv(path, index_col="time",
                                                       usecols=["time", "epoch"])
        _load_trialized_position._path = path
    return _load_trialized_position._cache


DEFAULT_PANELS = {
    "linear_position": "continuous",
    "speed":           "continuous",
    "trial_progress":  "continuous",
    "trial_type":      "categorical",
    "choice":          "categorical",
}

DEFAULT_HISTORY_WINDOWS_MS = ((0, 2), (2, 10))


def history_term_names(windows_ms=None):
    """Return history covariate column names for the provided windows."""
    windows_ms = windows_ms or DEFAULT_HISTORY_WINDOWS_MS
    return [f"hist_{lo}_{hi}ms" for lo, hi in windows_ms]


def make_history_formula(formula, windows_ms=None):
    """Return formula augmented with history terms (if not already present)."""
    windows_ms = windows_ms or DEFAULT_HISTORY_WINDOWS_MS
    terms = history_term_names(windows_ms)
    lhs, rhs = formula.split("~", 1)
    rhs = rhs.strip()
    extras = []
    for term in terms:
        pat = rf"(?<![A-Za-z0-9_]){re.escape(term)}(?![A-Za-z0-9_])"
        if re.search(pat, rhs) is None:
            extras.append(term)
    if extras:
        rhs = rhs + " + " + " + ".join(extras)
    return f"{lhs.strip()} ~ {rhs}"


def make_history_transform(windows_ms=None, bin_size=None):
    """Return per-unit transform callback that injects spike-history covariates."""
    windows_ms = windows_ms or DEFAULT_HISTORY_WINDOWS_MS
    bin_size = CONFIG["bin_size"] if bin_size is None else bin_size
    return partial(add_spike_history, windows_ms=windows_ms, bin_size=bin_size)


def plot_residuals(residuals_df, n_bins=50, title="",
                   show_cumulative=False, panels=None):
    """
    Diagnostic plots of model residuals.

    Panel 0 (optional) — cumulative residual vs time:
        Running sum of (observed - predicted). A trend indicates temporal drift
        not captured by any covariate.

    Continuous panels — CUSUM + binned mean residual vs covariate:
        Systematic shape indicates unmodelled nonlinearity or a missing predictor.

    Categorical panels — mean residual per category (± SEM):
        A non-zero mean for one category indicates the model under/over-predicts
        that condition and the categorical predictor should be added.

    Parameters
    ----------
    residuals_df    : output of compute_residuals()
    n_bins          : number of bins for continuous covariate plots
    title           : overall figure suptitle
    show_cumulative : bool — include the cumulative residual vs time panel
    panels          : dict {col: "continuous"|"categorical"}, optional.
        Specifies which columns to plot and how. Defaults to DEFAULT_PANELS.
        Columns absent from residuals_df or entirely NaN are silently skipped.
    """
    if panels is None:
        panels = DEFAULT_PANELS

    # Only keep panels whose column is present and has non-NaN data
    active = {col: kind for col, kind in panels.items()
              if col in residuals_df.columns and residuals_df[col].notna().any()}

    panel_size = 7
    n_panels = len(active) + (1 if show_cumulative else 0)
    ncols = int(np.ceil(np.sqrt(n_panels)))
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_size * ncols, panel_size * nrows))
    axes = np.atleast_1d(axes).flatten()
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    ax_idx = 0

 # Optional: cumulative residual vs time (first epoch only)
    if show_cumulative:
        ax = axes[ax_idx]; ax_idx += 1

        tp = _load_trialized_position()
        ep = tp["epoch"].dropna()
        first_epoch_val = ep.iloc[0]
        first_epoch_rows = tp.index[tp["epoch"] == first_epoch_val]
        t_start, t_end = first_epoch_rows.min(), first_epoch_rows.max()

        bin_times = residuals_df["time_bin_center"].values
        epoch1_mask = (bin_times >= t_start) & (bin_times <= t_end)
        epoch1_raw  = residuals_df.loc[epoch1_mask, "raw_residual"].values
        epoch1_times_min = (bin_times[epoch1_mask] - t_start) / 60

        cumresid = np.cumsum(epoch1_raw)
        stride   = max(1, len(cumresid) // 5000)
        ax.plot(epoch1_times_min[::stride], cumresid[::stride],
                lw=0.6, color="steelblue", rasterized=True)
        ax.axhline(0, color="red", lw=1, ls="--")

        ax.set_xlabel("Time in epoch (min)")
        ax.set_ylabel("Cumulative residual (spikes)")
        ax.set_title(f"Cumulative residual — epoch {int(first_epoch_val)}")
        sns.despine(ax=ax)

    CUSUM_COLOR = "tomato"
    RESID_COLOR = "steelblue"

    for col, kind in active.items():
        ax = axes[ax_idx]; ax_idx += 1
        valid    = residuals_df[col].notna()
        res_vals = residuals_df.loc[valid, "raw_residual"].values

        if kind == "continuous":
            cov_vals = residuals_df.loc[valid, col].values

            # Binned mean residual
            centers, mean_resid = bin_and_average(res_vals, cov_vals, n_bins)

            # CUSUM sorted by covariate value
            sort_order = np.argsort(cov_vals)
            cusum      = np.cumsum(res_vals[sort_order])
            x_cusum    = cov_vals[sort_order]

            ax.plot(x_cusum, cusum, lw=0.8, color=CUSUM_COLOR, alpha=0.7)
            ax.axhline(0, color=CUSUM_COLOR, lw=0.8, ls="--")
            ax2 = ax.twinx()
            bar_w = (centers[-1] - centers[0]) / max(len(centers) - 1, 1) * 0.9
            ax2.bar(centers, mean_resid, width=bar_w,
                    color=RESID_COLOR, alpha=0.4)
            ax2.axhline(0, color=RESID_COLOR, lw=0.8, ls="--")

            ax.set_ylabel("CUSUM (spikes)", color=CUSUM_COLOR)
            ax.tick_params(axis="y", colors=CUSUM_COLOR)
            ax.spines["left"].set_color(CUSUM_COLOR)
            ax2.set_ylabel("Mean residual (spikes/bin)", color=RESID_COLOR)
            ax2.tick_params(axis="y", colors=RESID_COLOR)
            ax2.spines["right"].set_color(RESID_COLOR)

        elif kind == "categorical":
            cats      = sorted(residuals_df.loc[valid, col].unique())
            cat_vals  = residuals_df.loc[valid, col].values
            means     = np.array([res_vals[cat_vals == c].mean() for c in cats])
            sems      = np.array([res_vals[cat_vals == c].std() /
                                  np.sqrt((cat_vals == c).sum()) for c in cats])

            ax.bar(cats, means, yerr=sems, color=RESID_COLOR, alpha=0.6,
                   error_kw=dict(lw=1.2, capsize=4, capthick=1.2))
            ax.axhline(0, color=RESID_COLOR, lw=0.8, ls="--")
            ax.set_ylabel("Mean residual (spikes/bin)", color=RESID_COLOR)
            ax.tick_params(axis="y", colors=RESID_COLOR)
            ax.spines["left"].set_color(RESID_COLOR)

        ax.set_xlabel(col)
        ax.set_title(f"Residual vs {col}")
        sns.despine(ax=ax)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()


def _compute_z_unsorted(predicted, spike_counts):
    """
    Core ISI rescaling: returns z values in temporal (ISI) order, not sorted.
    Used by compute_ks_rescaled (needs sorted) and compute_diagnostics_all_units
    (needs temporal order for autocorrelation).
    """
    counts    = np.asarray(spike_counts, dtype=int)
    spike_idx = np.where(counts > 0)[0]
    if len(spike_idx) < 2:
        return np.array([])
    z_vals = []
    for i in range(1, len(spike_idx)):
        start = spike_idx[i - 1] + 1
        end   = spike_idx[i]     + 1
        u_i   = predicted[start:end].sum()
        z_vals.append(1.0 - np.exp(-u_i))
    return np.array(z_vals)


def compute_ks_rescaled(results, spike_counts=None):
    """
    Apply the Time Rescaling Theorem to obtain Uniform(0,1) test statistics.

    For a Poisson GLM with predicted counts λ_j·Δt per bin, the integrated
    intensity across the ISI from spike i-1 to spike i is:

        u_i = Σ_{j = idx[i-1]+1}^{idx[i]} predicted_j

    Under a correct model, u_i ~ Exponential(1). Transforming:

        z_i = 1 - exp(-u_i)  →  z_i ~ Uniform(0, 1)

    The KS plot (empirical CDF of z_i vs diagonal) diagnoses specific failures:
      - Bows above diagonal : model underestimates rate at short ISIs (bursting)
      - Bows below diagonal : model overestimates rate (suppression not modelled)
      - S-shape             : correct mean rate, wrong temporal structure
      - Autocorrelated z_i  : unmodelled spike history / refractoriness

    Returns z_vals sorted ascending — pass directly to plot_ks().

    Parameters
    ----------
    results      : fitted statsmodels GLM result
    spike_counts : ignored (kept for backward compatibility); observed counts
                   are read from results.model.endog so alignment is automatic.
    """
    predicted = np.asarray(results.predict())
    observed, _ = _align_to_model(results)
    return np.sort(_compute_z_unsorted(predicted, observed))


def plot_ks(z_vals, alpha=0.05, ax=None, title = None):
    """
    KS plot: empirical CDF of rescaled ISIs vs Uniform(0,1) reference diagonal.

    95% confidence bands use the Kolmogorov–Smirnov distribution:
        ε = sqrt(−log(α/2) / (2n))

    Parameters
    ----------
    z_vals : sorted array from compute_ks_rescaled()
    alpha  : confidence level for bands (default 0.05 → 95% CI)
    ax     : matplotlib Axes; creates a new figure if None
    title  : plot title
    """
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 5))

    n = len(z_vals)
    if n == 0:
        ax.text(0.5, 0.5, "No ISIs", ha="center", va="center", transform=ax.transAxes)
        return

    ecdf    = np.arange(1, n + 1) / n
    epsilon = np.sqrt(-np.log(alpha / 2) / (2 * n))

    ax.fill_between(
        z_vals,
        np.clip(ecdf - epsilon, 0, 1),
        np.clip(ecdf + epsilon, 0, 1),
        alpha=0.2, color="steelblue", label=f"{int((1 - alpha) * 100)}% CI"
    )
    ax.plot(z_vals, ecdf, color="steelblue", lw=1.5, label="Empirical CDF")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Uniform(0,1)")
    ax.set_xlabel("z  (rescaled ISI)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(title)
    ax.legend(fontsize=8)
    sns.despine(ax=ax)

    if standalone:
        plt.tight_layout()


def _simulate_autoregressive(results, windows_ms=None, bin_size=0.002, n_sim=20, rng=None,
                             return_runs=False):
    """Simulate spike trains forward from a history GLM's own predictions.

    At each bin, non-history covariate drive comes from the fitted model design
    matrix (results.model.exog), while history covariates are recomputed from
    simulated (not observed) spikes. This keeps simulation compatible with
    null, single-variable, history-only, and full models.

    Uses a running cumulative sum so each window sum is O(1) per step,
    mirroring the cumsum trick in add_spike_history.

    History windows are inferred from coefficient names matching:
      hist_<lo_ms>_<hi_ms>ms
    The windows_ms argument is retained only for backward compatibility.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n_bins = int(results.nobs)
    exog = np.asarray(results.model.exog, dtype=float)
    exog_names = list(results.model.exog_names)

    # Align coefficient vector to model exog columns (works for null/history/full).
    params_raw = results.params
    if isinstance(params_raw, pd.Series):
        coef_series = params_raw.reindex(exog_names).fillna(0.0)
    else:
        coef_series = pd.Series(np.asarray(params_raw, dtype=float), index=exog_names)
    coefs = coef_series.to_numpy(dtype=float)

    # History terms are dynamic and must be recomputed from simulated spikes.
    # Non-history terms (position, speed, trial type, splines, intercept, etc.)
    # are taken directly from each fitted exog row.
    bsms = bin_size * 1000.0
    hist_mask = np.zeros(len(exog_names), dtype=bool)
    hist_terms = []
    hist_pattern = re.compile(r"^hist_(\d+(?:\.\d+)?)_(\d+(?:\.\d+)?)ms$")
    for i, (name, coef) in enumerate(zip(exog_names, coefs)):
        m = hist_pattern.match(name)
        if m is None:
            continue
        lo_ms = float(m.group(1))
        hi_ms = float(m.group(2))
        lo_b = int(round(lo_ms / bsms))
        hi_b = int(round(hi_ms / bsms))
        hist_mask[i] = True
        hist_terms.append((lo_b, hi_b, float(coef)))

    max_lag = max((hi for _, hi, _ in hist_terms), default=0)
    coefs_nonhist = coefs.copy()
    coefs_nonhist[hist_mask] = 0.0
    lp_nonhist = exog @ coefs_nonhist

    sim_isis_all = []
    run_data = []
    for _ in range(n_sim):
        cs     = np.zeros(n_bins + 1)  # cs[t] = total simulated spikes in [0, t)
        spikes = np.zeros(n_bins)

        for t in range(max_lag, n_bins):
            if t > 0:
                cs[t] = cs[t - 1] + spikes[t - 1]
            lp = lp_nonhist[t]
            for lo_b, hi_b, coef in hist_terms:
                end = t if lo_b == 0 else t - lo_b
                lp += coef * (cs[end] - cs[t - hi_b])
            spikes[t] = rng.poisson(np.exp(min(lp, 5.0)))

        idx = np.where(spikes > 0)[0]
        run_isis = np.array([])
        if len(idx) > 1:
            run_isis = np.diff(idx) * bin_size * 1000
            sim_isis_all.extend(run_isis)

        if return_runs:
            run_data.append((spikes, run_isis))

    if return_runs:
        return run_data
    return np.array(sim_isis_all)


def plot_predicted_isi(results, windows_ms=None, bin_size=None, max_isi_ms=200,
                       n_bins=40, n_sim=20, ax=None, title=None,
                       occ_floor=0.05, occ_factor=6.0,
                       rate_floor_hz=30.0, rate_factor=6.0,
                       max_count=15):
    """
    Plot observed vs predicted ISI distribution for a fitted Poisson GLM.

    Observed ISIs are derived from the model's endog (spike counts per bin).
    Predicted ISIs are estimated by conditional autoregressive simulation:
    non-history terms use the fitted design-matrix drive at each bin, while
    history covariates are recomputed from the simulated spike train so that
    post-spike dynamics (refractoriness, facilitation) propagate correctly.

    Parameters
    ----------
    results    : fitted statsmodels GLM result (Poisson)
    windows_ms : optional legacy arg (kept for backward compatibility).
                 History columns are inferred from fitted coefficient names
                 matching hist_<lo>_<hi>ms.
    bin_size   : bin width in seconds (default: CONFIG["bin_size"])
    max_isi_ms : upper limit of the histogram range in ms
    n_bins     : number of histogram bins
    n_sim      : number of simulated spike trains pooled for the prediction
    ax         : matplotlib Axes; creates a new figure if None
    title      : plot title
    occ_floor  : minimum occupied-bin threshold for rejecting unstable runs
    occ_factor : reject if sim occupied-bin fraction > occ_factor * observed
    rate_floor_hz : minimum firing-rate threshold (Hz) for rejection
    rate_factor   : reject if simulated mean rate > rate_factor * observed
    max_count  : reject if max simulated count in any bin exceeds this value
    """
    bin_size = bin_size or CONFIG["bin_size"]
    rng = np.random.default_rng(0)

    observed = np.asarray(results.model.endog, dtype=int)
    spike_idx = np.where(observed > 0)[0]
    observed_isis_ms = np.diff(spike_idx) * bin_size * 1000

    obs_occ = float((observed > 0).mean())
    obs_rate_hz = float(observed.mean() / bin_size)
    occ_cut = max(occ_floor, occ_factor * obs_occ)
    rate_cut_hz = max(rate_floor_hz, rate_factor * obs_rate_hz)

    run_data = _simulate_autoregressive(
        results, windows_ms, bin_size, n_sim, rng, return_runs=True
    )
    accepted_isis = []
    n_rejected = 0
    for spikes, run_isis in run_data:
        sim_occ = float((spikes > 0).mean())
        sim_rate_hz = float(spikes.mean() / bin_size)
        sim_max = float(spikes.max()) if len(spikes) else 0.0

        explode = (
            (sim_occ > occ_cut)
            or (sim_rate_hz > rate_cut_hz)
            or (sim_max > max_count)
        )
        if explode:
            n_rejected += 1
            continue
        if run_isis.size:
            accepted_isis.append(run_isis)

    sim_isis_ms = np.concatenate(accepted_isis) if accepted_isis else np.array([])
    n_kept = n_sim - n_rejected

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 4))

    bins = np.linspace(0, max_isi_ms, n_bins + 1)
    obs_clipped = observed_isis_ms[observed_isis_ms <= max_isi_ms]
    sim_clipped = sim_isis_ms[sim_isis_ms <= max_isi_ms]

    ax.hist(obs_clipped, bins=bins, density=True, alpha=0.6,
            color="steelblue", label="Observed")
    pred_label = f"Predicted (kept {n_kept}/{n_sim})"
    if sim_clipped.size:
        ax.hist(sim_clipped, bins=bins, density=True, histtype="step",
                color="tomato", lw=2, label=pred_label)
    else:
        ax.plot([], [], color="tomato", lw=2, label=pred_label)
        ax.text(0.5, 0.85, "No accepted simulated ISIs",
                ha="center", va="center", transform=ax.transAxes,
                color="tomato", fontsize=9)

    ax.set_xlabel("ISI (ms)")
    ax.set_ylabel("Density")
    ax.set_title(title or "ISI distribution: observed vs predicted")
    ax.legend(fontsize=9)
    sns.despine(ax=ax)

    if standalone:
        plt.tight_layout()


def plot_diagnostics_batch(unit_list, formula, cov_df, spike_counts_masked, unit_ids,
                           panels=None, n_bins=50, alpha=0.05, show_cumulative=False):
    """
    Fit GLM and run full diagnostics for a chosen list of units (max 5).

    Produces one row per unit. Columns (left to right):
      [optional: cumulative residual vs time] [residual panels...] [KS plot]

    Continuous panels show binned mean residual vs covariate.
    Categorical panels show mean residual ± SEM per category.

    Parameters
    ----------
    unit_list           : list of unit IDs to inspect (truncated to 5 if longer)
    formula             : GLM formula string used for fitting
    cov_df              : covariate DataFrame (n_bins rows, no spike_count column)
    spike_counts_masked : 2-D array (n_units × n_bins), observed spike counts
    unit_ids            : 1-D array of unit IDs, same order as spike_counts_masked rows
    panels              : dict {col: "continuous"|"categorical"}, optional.
        Defaults to DEFAULT_PANELS. Columns absent from cov_df are skipped.
    n_bins              : number of bins for continuous covariate residual panels
    alpha               : CI level for KS confidence band (default 0.05)
    show_cumulative     : bool — include cumulative residual vs time panel (default False)
    """
    if panels is None:
        panels = DEFAULT_PANELS

    active = {col: kind for col, kind in panels.items()
              if col in cov_df.columns and cov_df[col].notna().any()}

    if len(unit_list) > 5:
        print(f"unit_list has {len(unit_list)} entries — truncating to first 5")
        unit_list = unit_list[:5]

    unit_ids_arr = np.asarray(unit_ids)
    n_rows = len(unit_list)
    n_cols = (1 if show_cumulative else 0) + len(active) + 1  # panels + KS

    col_titles = (["Cumul. resid. vs time"] if show_cumulative else []) + \
                 [f"Resid. vs {c}" for c in active] + ["KS"]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4.5 * n_rows),
        constrained_layout=True,
    )
    axes = np.array(axes).reshape(n_rows, n_cols)

    RESID_COLOR = "steelblue"

    for row_idx, uid in enumerate(unit_list):
        matches = np.where(unit_ids_arr == uid)[0]
        if len(matches) == 0:
            for ax in axes[row_idx]:
                ax.text(0.5, 0.5, f"unit {uid}\nnot found",
                        ha="center", va="center", transform=ax.transAxes, fontsize=9)
            continue
        i = matches[0]

 # fit GLM
        df = cov_df.copy()
        df["spike_count"] = spike_counts_masked[i]
        try:
            res = _fit_glm_with_fallback(formula, df, family=sm.families.Poisson())
            if not _result_converged(res):
                raise RuntimeError("did not converge")
        except Exception as e:
            for ax in axes[row_idx]:
                ax.text(0.5, 0.5, f"unit {uid}\nfit failed\n{e}",
                        ha="center", va="center", transform=ax.transAxes, fontsize=8)
            continue

        predicted = np.asarray(res.predict())
        observed, cov_df_fit = _align_to_model(res, cov_df=cov_df)
        raw = observed - predicted

        col_idx = 0

 # Optional: cumulative residual vs time
        if show_cumulative:
            ax = axes[row_idx, col_idx]; col_idx += 1
            cumresid = np.cumsum(raw)
            stride   = max(1, len(cumresid) // 5000)
            ax.plot(np.arange(0, len(cumresid), stride), cumresid[::stride],
                    lw=0.6, color=RESID_COLOR, rasterized=True)
            ax.axhline(0, color="red", lw=1, ls="--")
            ax.set_xlabel("Time", fontsize=8)
            ax.set_ylabel(f"unit {uid}\ncumul. resid.", fontsize=8)
            ax.tick_params(labelsize=7)
            sns.despine(ax=ax)

 # Covariate panels
        cont_ax_indices = []
        for col, kind in active.items():
            ax = axes[row_idx, col_idx]; col_idx += 1
            valid    = cov_df_fit[col].notna().values
            res_vals = raw[valid]

            if kind == "continuous":
                cov_vals = cov_df_fit[col].values[valid]
                centers, mean_resid = bin_and_average(res_vals, cov_vals, n_bins)
                ax.plot(centers, mean_resid, lw=1.2, color=RESID_COLOR)
                ax.axhline(0, color=RESID_COLOR, lw=0.8, ls="--")
                ax.set_ylabel("Mean resid.\n(spikes/bin)", fontsize=8, color=RESID_COLOR)
                cont_ax_indices.append(col_idx - 1)

            elif kind == "categorical":
                cat_vals = cov_df_fit[col].values[valid]
                cats     = sorted(set(cat_vals[~pd.isnull(cat_vals)]))
                means    = np.array([res_vals[cat_vals == c].mean() for c in cats])
                sems     = np.array([res_vals[cat_vals == c].std() /
                                     np.sqrt((cat_vals == c).sum()) for c in cats])
                ax.bar(cats, means, yerr=sems, color=RESID_COLOR, alpha=0.6,
                       error_kw=dict(lw=1.2, capsize=4, capthick=1.2))
                ax.axhline(0, color=RESID_COLOR, lw=0.8, ls="--")
                ax.set_ylabel("Mean resid.\n(spikes/bin)", fontsize=8, color=RESID_COLOR)

            ax.set_xlabel(col, fontsize=8)
            ax.tick_params(labelsize=7)
            ax.tick_params(axis="y", colors=RESID_COLOR)
            ax.spines["left"].set_color(RESID_COLOR)
            sns.despine(ax=ax)

        # Equalise y-limits across continuous panels
        if len(cont_ax_indices) > 1:
            cont_axes = [axes[row_idx, j] for j in cont_ax_indices]
            max_abs   = max(max(abs(y) for y in ax.get_ylim()) for ax in cont_axes)
            for ax in cont_axes:
                ax.set_ylim(-max_abs, max_abs)

 # KS panel
        ax = axes[row_idx, col_idx]
        z  = compute_ks_rescaled(res)
        n  = len(z)
        if n > 0:
            ecdf    = np.arange(1, n + 1) / n
            epsilon = np.sqrt(-np.log(alpha / 2) / (2 * n))
            ax.fill_between(z, np.clip(ecdf - epsilon, 0, 1),
                            np.clip(ecdf + epsilon, 0, 1),
                            alpha=0.2, color=RESID_COLOR)
            ax.plot(z, ecdf, color=RESID_COLOR, lw=1.5)
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            D      = np.max(np.abs(ecdf - z))
            passed = D <= epsilon
            ax.text(0.05, 0.90, f"D={D:.3f} {'pass' if passed else 'FAIL'}",
                    transform=ax.transAxes, fontsize=8,
                    color="green" if passed else "red", fontweight="bold")
        ax.set_xlabel("z (rescaled ISI)", fontsize=8)
        ax.set_ylabel("Empirical CDF", fontsize=8)
        ax.tick_params(labelsize=7)
        sns.despine(ax=ax)

        # Unit label on leftmost panel
        axes[row_idx, 0].set_ylabel(f"unit {uid}\n" +
                                    axes[row_idx, 0].get_ylabel(), fontsize=8)

    # Column titles on first row only
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=9, fontweight="bold")

    fig.suptitle("GLM diagnostics — selected units", fontsize=12)


def plot_diagnostics(results, spike_counts=None, cov_df=None,
                     n_bins=50, unit_label="", alpha=0.05,
                     panels=None, show_cumulative=False):
    """
    Full diagnostic dashboard for one unit's fitted Poisson GLM.

    Calls compute_residuals → plot_residuals (covariate panels),
    then compute_ks_rescaled → plot_ks in a separate figure.

    Parameters
    ----------
    results         : fitted statsmodels GLM result
    spike_counts    : ignored (kept for backward compatibility); observed counts
                      are read from the fitted model automatically.
    cov_df          : covariate DataFrame. Auto-trimmed to match model rows if
                      longer (e.g. when add_spike_history dropped leading rows).
    n_bins          : bins for continuous covariate residual plots
    unit_label      : string identifier shown in plot titles
    alpha           : CI level for KS bands (default 0.05)
    panels          : dict {col: "continuous"|"categorical"}, optional.
        Forwarded to plot_residuals. Defaults to DEFAULT_PANELS.
    show_cumulative : bool — include the cumulative residual vs time panel.
    """
    residuals_df = compute_residuals(results, cov_df=cov_df)
    plot_residuals(residuals_df, n_bins=n_bins,
                   title=f"Residuals — {unit_label}",
                   panels=panels, show_cumulative=show_cumulative)

    z_vals = compute_ks_rescaled(results)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_ks(z_vals, alpha=alpha, ax=ax, title=f"KS — {unit_label}",)
    plt.tight_layout()


def compute_model_diagnostics(formula, cov_df, spike_counts_masked, unit_ids,
                               covariate_cols, base_dir, model_name,
                               n_bins=50, unit_subset=None, categorical_cols=None,
                               per_unit_transform=None):
    """
    Fit GLM once per unit and compute both scalar diagnostics and residual profiles.

    Saves two files (skipped when unit_subset is given):
      {base_dir}/analysis/diagnostics_{model_name}.csv       — scalar diagnostics
      {base_dir}/analysis/residual_profiles_{model_name}.npz — binned profiles

    If both files already exist, loads and returns them without refitting.

    Scalars per unit
    ----------------
    ks_D             : KS D = max|ECDF(z) − z|. Overall spike-timing misfit.
    ks_z_autocorr    : Spearman r(z_i, z_{i+1}). Serial ISI structure.
    drift_auc        : mean(|cumresid|) / n_spikes. Temporal non-stationarity.
    resid_eta2_{col} : η² = Σ(n_b × mean_b²) / Σ(raw²). Covariate misfit fraction.

    Returns
    -------
    diag_df  : DataFrame of scalar diagnostics (one row per unit)
    profiles : dict with "{col}_profiles" (n_units × n_bins), "{col}_centers", "unit_ids"
    """
    csv_path = f"{base_dir}/analysis/diagnostics_{model_name}.csv"
    npz_path = f"{base_dir}/analysis/residual_profiles_{model_name}.npz"

    if unit_subset is None and os.path.exists(csv_path) and os.path.exists(npz_path):
        print(f"Loading existing: {csv_path}")
        diag_df = pd.read_csv(csv_path, index_col=0)
        raw_npz = np.load(npz_path, allow_pickle=True)
        profiles = {k: raw_npz[k] for k in raw_npz.files}
        return diag_df, profiles

    unit_ids_arr = np.asarray(unit_ids)
    if unit_subset is not None:
        uid_set  = set(unit_subset)
        indices  = [i for i, uid in enumerate(unit_ids_arr) if uid in uid_set]
    else:
        indices  = list(range(len(unit_ids_arr)))

 # pre-compute bin edges for profiles
    bin_edges   = {}
    bin_centers = {}
    for col in covariate_cols:
        valid = cov_df[col].dropna().values
        edges = np.linspace(valid.min(), valid.max(), n_bins + 1)
        bin_edges[col]   = edges
        bin_centers[col] = (edges[:-1] + edges[1:]) / 2

    categorical_cols = categorical_cols or []
    # pre-compute category labels per categorical col
    cat_labels = {}
    for col in categorical_cols:
        if col in cov_df.columns:
            vals = cov_df[col].dropna().unique()
            cat_labels[col] = sorted(vals)

    diag_rows      = []
    profile_rows   = {col: [] for col in covariate_cols}
    cat_profile_rows = {col: [] for col in categorical_cols if col in cov_df.columns}
    valid_prof_ids = []

    for count, i in enumerate(indices):
        uid = unit_ids_arr[i]
        df  = cov_df.copy()
        df["spike_count"] = spike_counts_masked[i]
        if per_unit_transform is not None:
            df = per_unit_transform(df, spike_counts_masked[i], i)

        try:
            res = _fit_glm_with_fallback(formula, df, family=sm.families.Poisson())
            if not _result_converged(res):
                raise RuntimeError("not converged")
        except Exception:
            row = dict(unit=uid, converged=False,
                       ks_D=np.nan, ks_z_autocorr=np.nan, drift_auc=np.nan)
            for col in covariate_cols:
                row[f"resid_eta2_{col}"] = np.nan
            diag_rows.append(row)
            continue

        predicted = np.asarray(res.predict())
        observed, cov_df_fit = _align_to_model(res, cov_df=cov_df)
        raw       = observed - predicted
        n_spikes  = int((observed > 0).sum())

 # drift AUC
        cumresid  = np.cumsum(raw)
        drift_auc = np.abs(cumresid).mean() / max(n_spikes, 1)

 # KS D + z autocorrelation
        z_unsorted = _compute_z_unsorted(predicted, observed)
        if len(z_unsorted) >= 4:
            z_sorted      = np.sort(z_unsorted)
            n             = len(z_sorted)
            ecdf          = np.arange(1, n + 1) / n
            ks_D          = float(np.max(np.abs(ecdf - z_sorted)))
            z_autocorr, _ = spearmanr(z_unsorted[:-1], z_unsorted[1:])
        else:
            ks_D, z_autocorr = np.nan, np.nan

 # residual η² per covariate
        row      = dict(unit=uid, converged=True,
                        ks_D=ks_D, ks_z_autocorr=float(z_autocorr), drift_auc=drift_auc)
        ss_total = float(np.sum(raw ** 2))
        for col in covariate_cols:
            valid_mask = ~np.isnan(cov_df_fit[col].values)
            if valid_mask.sum() > 10 and ss_total > 0:
                cov_vals = cov_df_fit[col].values[valid_mask]
                res_vals = raw[valid_mask]
                bins     = np.linspace(cov_vals.min(), cov_vals.max(), 51)
                bin_idx  = np.clip(np.digitize(cov_vals, bins) - 1, 0, 49)
                ss_between = sum(
                    (res_vals[bin_idx == b].mean() ** 2) * (bin_idx == b).sum()
                    for b in range(50) if (bin_idx == b).any()
                )
                row[f"resid_eta2_{col}"] = float(ss_between / ss_total)
            else:
                row[f"resid_eta2_{col}"] = np.nan
        diag_rows.append(row)

 # residual profiles
        for col in covariate_cols:
            valid_mask = ~np.isnan(cov_df_fit[col].values)
            cov_vals   = cov_df_fit[col].values[valid_mask]
            res_vals   = raw[valid_mask]
            idx        = np.clip(np.digitize(cov_vals, bin_edges[col]) - 1, 0, n_bins - 1)
            means      = np.array([
                res_vals[idx == b].mean() if (idx == b).any() else np.nan
                for b in range(n_bins)
            ])
            profile_rows[col].append(means)
 # categorical profiles
        for col in categorical_cols:
            if col not in cov_df_fit.columns:
                continue
            cat_vals = cov_df_fit[col].values
            means = np.array([
                raw[cat_vals == c].mean() if (cat_vals == c).any() else np.nan
                for c in cat_labels[col]
            ])
            cat_profile_rows[col].append(means)

        valid_prof_ids.append(uid)

        if (count + 1) % 50 == 0:
            print(f"  {count + 1}/{len(indices)} units done")

    diag_df  = pd.DataFrame(diag_rows)
    profiles = {"unit_ids": np.array(valid_prof_ids)}
    for col in covariate_cols:
        profiles[f"{col}_profiles"] = np.array(profile_rows[col])
        profiles[f"{col}_centers"]  = bin_centers[col]
    for col in categorical_cols:
        if col in cat_profile_rows:
            profiles[f"{col}_cat_profiles"] = np.array(cat_profile_rows[col])
            profiles[f"{col}_cat_labels"]   = np.array(cat_labels[col], dtype=object)

    if unit_subset is None:
        diag_df.to_csv(csv_path)
        print(f"Saved → {csv_path}")
        np.savez(npz_path, **profiles)
        print(f"Saved → {npz_path}")

    return diag_df, profiles


def compute_diagnostics_all_units(formula, cov_df, spike_counts_masked, unit_ids,
                                   covariate_cols, base_dir, model_name,
                                   unit_subset=None, per_unit_transform=None):
    """Thin wrapper — calls compute_model_diagnostics and returns only diag_df."""
    diag_df, _ = compute_model_diagnostics(
        formula, cov_df, spike_counts_masked, unit_ids,
        covariate_cols, base_dir, model_name,
        unit_subset=unit_subset,
        per_unit_transform=per_unit_transform,
    )
    return diag_df


def _build_long_df(diag_dfs):
    """Normalise diag_dfs to dict, filter converged, add model column."""
    if isinstance(diag_dfs, pd.DataFrame):
        diag_dfs = {"model": diag_dfs}
    frames = []
    for label, df in diag_dfs.items():
        tmp = df[df["converged"] == True].copy()
        tmp["model"] = label
        frames.append(tmp)
    return pd.concat(frames, ignore_index=True), list(diag_dfs.keys())


def _infer_covariate_cols(long_df):
    return [c.replace("resid_eta2_", "")
            for c in long_df.columns if c.startswith("resid_eta2_")]


def plot_diagnostics_population(diag_dfs, covariate_cols=None, alpha=0.05):
    """
    Two figures summarising population-level GLM diagnostics.

    Figure 1 — timing & stationarity (panels A–C):
      A : KS D distribution — overall spike-timing misfit
      B : drift_auc — temporal non-stationarity
      C : ks_z_autocorr — ISI serial correlation (spike history)

    Figure 2 — covariate residual structure (panel D):
      D  : η² violin per covariate × model
      D2 : scatter η²_cov1 vs η²_cov2 per unit (2-covariate case only)

    Parameters
    ----------
    diag_dfs       : DataFrame or dict {label: DataFrame}
    covariate_cols : covariate names; inferred from column names if None
    alpha          : KS CI threshold for reference line on Panel A
    """
    long_df, model_order = _build_long_df(diag_dfs)
    n_models = len(model_order)
    if covariate_cols is None:
        covariate_cols = _infer_covariate_cols(long_df)
    palette = sns.color_palette("tab10", n_models)

 # Figure 1: timing & stationarity
    fig1, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    ax = axes[0]
    sns.violinplot(data=long_df, x="model", y="ks_D", order=model_order,
                   palette=palette, inner="box", cut=0, ax=ax)
    median_n = long_df["ks_D"].count() // n_models
    ref_eps  = np.sqrt(-np.log(alpha / 2) / (2 * max(median_n, 1)))
    ax.axhline(ref_eps, color="red", lw=1, ls="--", label=f"ε (n≈{median_n})")
    ax.set_ylabel("KS D statistic"); ax.set_title("A  KS timing misfit")
    ax.legend(fontsize=8); sns.despine(ax=ax)

    ax = axes[1]
    sns.violinplot(data=long_df, x="model", y="drift_auc", order=model_order,
                   palette=palette, inner="box", cut=0, ax=ax)
    ax.axhline(0, color="red", lw=1, ls="--")
    ax.set_ylabel("drift AUC (norm.)"); ax.set_title("B  Temporal non-stationarity")
    sns.despine(ax=ax)

    ax = axes[2]
    sns.violinplot(data=long_df, x="model", y="ks_z_autocorr", order=model_order,
                   palette=palette, inner="box", cut=0, ax=ax)
    ax.axhline(0, color="red", lw=1, ls="--", label="0 (no history)")
    ax.set_ylabel("Spearman r(z_i, z_{i+1})")
    ax.set_title("C  ISI autocorrelation (spike history)")
    ax.legend(fontsize=8); sns.despine(ax=ax)

    fig1.suptitle("Population diagnostics — timing & stationarity", fontsize=13)

 # Figure 2: covariate residual structure
    eta2_cols = [f"resid_eta2_{c}" for c in covariate_cols]
    melt_df   = long_df.melt(id_vars=["unit", "model"], value_vars=eta2_cols,
                              var_name="covariate", value_name="eta2")
    melt_df["covariate"] = melt_df["covariate"].str.replace("resid_eta2_", "",
                                                             regex=False)

    fig2, ax2 = plt.subplots(figsize=(6, 5), constrained_layout=True)
    sns.violinplot(data=melt_df, x="covariate", y="eta2", hue="model",
                   hue_order=model_order, palette=palette,
                   inner="box", cut=0, ax=ax2)
    ax2.axhline(0, color="red", lw=1, ls="--", label="η² = 0")
    ax2.set_ylabel("η²  (residual variance explained by covariate)")
    ax2.set_title("D  Covariate residual structure (η²)")
    ax2.legend(fontsize=7, loc="upper right")
    sns.despine(ax=ax2)
    fig2.suptitle("Population diagnostics — covariate structure", fontsize=13)

    # scatter: η²_cov1 vs η²_cov2, only when exactly 2 covariates
    if len(covariate_cols) == 2:
        fig3, ax3 = plt.subplots(figsize=(5, 5))
        c0, c1 = eta2_cols
        for label, color in zip(model_order, palette):
            sub = long_df[long_df["model"] == label]
            ax3.scatter(sub[c0], sub[c1], s=15, alpha=0.6,
                        color=color, label=label)
        ax3.axhline(0, color="grey", lw=0.8, ls="--")
        ax3.axvline(0, color="grey", lw=0.8, ls="--")
        ax3.set_xlabel(f"η²(residual, {covariate_cols[0]})")
        ax3.set_ylabel(f"η²(residual, {covariate_cols[1]})")
        ax3.set_title("D (scatter)  Co-occurrence of residual η²\n"
                      "Diagonal = both failures in same unit → missing interaction")
        ax3.legend(fontsize=8)
        sns.despine(ax=ax3)
        fig3.tight_layout()


def compute_residual_profiles(formula, cov_df, spike_counts_masked, unit_ids,
                               covariate_cols, base_dir, model_name, n_bins=50):
    """Thin wrapper — calls compute_model_diagnostics and returns only profiles."""
    _, profiles = compute_model_diagnostics(
        formula, cov_df, spike_counts_masked, unit_ids,
        covariate_cols, base_dir, model_name, n_bins=n_bins,
    )
    return profiles


def plot_residual_heterogeneity(profiles, variable, row_normalise=True):
    """
    Population-level residual heterogeneity heatmap for a single covariate.

    Parameters
    ----------
    profiles      : dict from compute_model_diagnostics.
    variable      : str, covariate name (e.g. "linear_position", "speed",
                    "trial_type"). Continuous variables use ``{variable}_profiles``
                    and ``{variable}_centers``; categorical variables use
                    ``{variable}_cat_profiles`` and ``{variable}_cat_labels``.
    row_normalise : bool (default True). If True, each row is divided by its
                    max absolute value so all neurons share the same ±1 scale.
    """
    # detect variable type
    if f"{variable}_cat_profiles" in profiles:
        kind = "categorical"
    elif f"{variable}_profiles" in profiles:
        kind = "continuous"
    else:
        raise ValueError(f"No profiles found for variable '{variable}'. "
                         f"Available keys: {list(profiles.keys())}")

    is_position = (kind == "continuous" and variable == "linear_position")

    if is_position:
        fig, (ax, ax_track) = plt.subplots(
            2, 1, figsize=(10, 9),
            gridspec_kw={"height_ratios": [6, 1], "hspace": 0.08},
        )
    else:
        fig, ax = plt.subplots(figsize=(8, 8))

    if kind == "continuous":
        mat = np.array(profiles[f"{variable}_profiles"])
        centers = np.array(profiles[f"{variable}_centers"])

        if row_normalise:
            row_max = np.nanmax(np.abs(mat), axis=1, keepdims=True)
            row_max[row_max == 0] = np.nan
            mat_plot = mat / row_max
            vmin, vmax = -1, 1
        else:
            mat_plot = mat
            absmax = np.nanmax(np.abs(mat))
            vmin, vmax = -absmax, absmax

        sort_order = np.argsort(np.nanargmax(np.abs(mat_plot), axis=1))

        if is_position:
            # remap columns onto the gapped, duplicated-stem axis
            axis_info = _get_linearized_branch_axis()
            seg_bounds = axis_info["seg_bounds"]

            # rebuild the section list with segment ids and directions
            # (mirrors the definition inside _get_linearized_branch_axis)
            seg_len = {seg: sb["max"] - sb["min"]
                       for seg, sb in seg_bounds.items()}
            section_defs = [
                dict(seg=3, forward=False),
                dict(seg=1, forward=False),
                dict(seg=0, forward=True),   # stem_left
                dict(seg=0, forward=False),  # stem_right
                dict(seg=2, forward=True),
                dict(seg=4, forward=True),
            ]

            # collect columns per track section, with gap separators
            section_blocks = []
            gap_positions = axis_info["gap_centers"]
            gap_idx = 0
            for i_sec, (sec_def, ts) in enumerate(
                    zip(section_defs, axis_info["track_sections"])):
                # insert NaN gap before this section if needed
                if gap_idx < len(gap_positions):
                    gc = gap_positions[gap_idx]
                    if ts["x0"] > gc:
                        section_blocks.append(("gap", gc))
                        gap_idx += 1

                seg = sec_def["seg"]
                raw_min = seg_bounds[seg]["min"]
                raw_max = seg_bounds[seg]["max"]
                x0, x1 = ts["x0"], ts["x1"]
                mask = (centers >= raw_min) & (centers <= raw_max)
                idx = np.where(mask)[0]
                if len(idx) == 0:
                    continue
                seg_centers = centers[idx]
                frac = np.where(
                    raw_max > raw_min,
                    (seg_centers - raw_min) / (raw_max - raw_min),
                    0.0,
                )
                if sec_def["forward"]:
                    gx = x0 + frac * (x1 - x0)
                else:
                    gx = x1 - frac * (x1 - x0)
                order = np.argsort(gx)
                section_blocks.append(("data", gx[order], mat_plot[:, idx[order]]))
            # trailing gap
            if gap_idx < len(gap_positions):
                section_blocks.append(("gap", gap_positions[gap_idx]))

            # build final matrix with NaN columns at gaps
            n_units = mat_plot.shape[0]
            gap_width = 10.0
            all_cols = []
            all_x = []
            for block in section_blocks:
                if block[0] == "gap":
                    gc = block[1]
                    all_x.append(gc - gap_width / 2)
                    all_cols.append(np.full(n_units, np.nan))
                    all_x.append(gc + gap_width / 2)
                    all_cols.append(np.full(n_units, np.nan))
                else:
                    _, gx, cols = block
                    for j in range(len(gx)):
                        all_x.append(gx[j])
                        all_cols.append(cols[:, j])

            mat_gapped = np.column_stack(all_cols)
            gapped_x = np.array(all_x)

            # pcolormesh needs cell edges, not centers
            n_cols = len(gapped_x)
            x_edges = np.empty(n_cols + 1)
            x_edges[1:-1] = 0.5 * (gapped_x[:-1] + gapped_x[1:])
            x_edges[0] = gapped_x[0] - (x_edges[1] - gapped_x[0])
            x_edges[-1] = gapped_x[-1] + (gapped_x[-1] - x_edges[-2])
            y_edges = np.arange(n_units + 1)

            im = ax.pcolormesh(
                x_edges, y_edges,
                mat_gapped[sort_order],
                cmap="RdBu_r", vmin=vmin, vmax=vmax,
                shading="flat",
            )
            # gap and branch annotations
            for x_gap in axis_info.get("gap_centers", []):
                ax.axvline(x_gap, color="k", lw=1.5, ls="--", zorder=5, alpha=0.8)
            x_dup = axis_info.get("stem_duplicate_center")
            if x_dup is not None:
                ax.axvline(x_dup, color="k", lw=1.75, ls="-", zorder=5, alpha=1)

            label_map = {"top": "upper", "middle": "stem", "bottom": "lower"}
            n_units = mat.shape[0]
            for branch_name, label in label_map.items():
                for span in axis_info.get("branch_spans", {}).get(branch_name, []):
                    x_center = 0.5 * (span["xmin"] + span["xmax"])
                    ax.text(
                        x_center, -0.5, label,
                        ha="center", va="bottom", fontsize=10,
                        fontweight="semibold", color="0.25",
                        clip_on=False,
                    )
            ax.xaxis.set_visible(False)
            sns.despine(ax=ax, bottom=True)

            _draw_repeated_stem_track(ax_track, axis_info)
            ax_track.set_xlim(ax.get_xlim())
            ax_track.set_xlabel("Linear position (cm)")
        else:
            im = ax.imshow(
                mat_plot[sort_order], aspect="auto", cmap="RdBu_r",
                vmin=vmin, vmax=vmax,
                extent=[centers[0], centers[-1], mat.shape[0], 0],
                interpolation="nearest",
            )
            ax.set_xlabel(variable)
            sns.despine(ax=ax)

    elif kind == "categorical":
        mat = np.array(profiles[f"{variable}_cat_profiles"])
        labels = list(profiles[f"{variable}_cat_labels"])
        if row_normalise:
            row_max = np.nanmax(np.abs(mat), axis=1, keepdims=True)
            row_max[row_max == 0] = np.nan
            mat_plot = mat / row_max
            vmin, vmax = -1, 1
        else:
            mat_plot = mat
            absmax = np.nanmax(np.abs(mat))
            vmin, vmax = -absmax, absmax
        sort_order = np.argsort(np.nanargmax(np.abs(mat_plot), axis=1))
        im = ax.imshow(
            mat_plot[sort_order], aspect="auto", cmap="RdBu_r",
            vmin=vmin, vmax=vmax,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_xlabel(variable)
        sns.despine(ax=ax)

    cbar_label = "Normalised residual (a.u.)" if row_normalise else "Residual (spikes/bin)"
    title_suffix = "(row-normalised)" if row_normalise else "(raw)"
    cbar_axes = [ax, ax_track] if is_position else ax
    plt.colorbar(im, ax=cbar_axes, label=cbar_label, shrink=0.8)
    ax.set_ylabel("Unit (sorted by peak)")
    ax.set_title(f"Residual profiles — {variable}\n{title_suffix}")

    plt.tight_layout()
    return fig


def plot_residual_profiles(profiles_dicts, covariate_cols=None):
    """
    Population-level mean residual profile per covariate, with model comparison.

    One panel per covariate. For each model, shows the mean residual averaged
    across all units as a function of covariate value, with an uncertainty band.
    A flat line at zero = model captures that variable perfectly across the population.
    Systematic peaks reveal where on the track / at what speed the model fails.

    Parameters
    ----------
    profiles_dicts : dict {model_label: profiles_dict} or single profiles_dict
                     (profiles_dict is the output of compute_residual_profiles)
    covariate_cols : list of covariate names; inferred from keys if None
    """
    if not isinstance(profiles_dicts, dict) or "unit_ids" in profiles_dicts:
        profiles_dicts = {"model": profiles_dicts}

    first = next(iter(profiles_dicts.values()))
    detected = {}
    for key in first:
        if key.endswith("_cat_profiles"):
            detected[key.replace("_cat_profiles", "")] = "categorical"
        elif key.endswith("_profiles"):
            detected[key.replace("_profiles", "")] = "continuous"

    if covariate_cols is None:
        covariate_cols = list(detected.keys())
    panel_kind = {col: detected.get(col, "continuous") for col in covariate_cols}

    model_order = list(profiles_dicts.keys())
    palette     = sns.color_palette("tab10", len(model_order))
    n_covs      = len(covariate_cols)

    fig, axes = plt.subplots(1, n_covs, figsize=(6 * n_covs, 5),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for k, col in enumerate(covariate_cols):
        ax = axes[k]
        ax.axhline(0, color="red", lw=1, ls="--", zorder=0)

        for label, color in zip(model_order, palette):
            prof  = profiles_dicts[label]
            kind = panel_kind[col]
            if kind == "categorical":
                prof_key = f"{col}_cat_profiles"
                lab_key = f"{col}_cat_labels"
                if prof_key not in prof or lab_key not in prof:
                    continue
                mat = np.asarray(prof[prof_key])
                labels = list(prof[lab_key])
                if mat.shape[0] == 0:
                    continue
                mean_prof = np.nanmean(mat, axis=0)
                sem_prof = np.nanstd(mat, axis=0) / np.sqrt(mat.shape[0])
                x = np.arange(len(labels))
                ax.plot(x, mean_prof, color=color, lw=1.5, label=label)
                ax.fill_between(x,
                                mean_prof - sem_prof,
                                mean_prof + sem_prof,
                                color=color, alpha=0.25)
                ax.set_xticks(x)
                ax.set_xticklabels(labels)
            else:
                prof_key = f"{col}_profiles"
                cent_key = f"{col}_centers"
                if prof_key not in prof or cent_key not in prof:
                    continue
                mat = np.asarray(prof[prof_key])   # (n_units, n_bins)
                cents = np.asarray(prof[cent_key]) # (n_bins,)
                if mat.shape[0] == 0:
                    continue
                mean_prof = np.nanmean(mat, axis=0)
                sem_prof  = np.nanstd(mat, axis=0) / np.sqrt(mat.shape[0])
                ax.plot(cents, mean_prof, color=color, lw=1.5, label=label)
                ax.fill_between(cents,
                                mean_prof - sem_prof,
                                mean_prof + sem_prof,
                                color=color, alpha=0.25)

        ax.set_xlabel(col)
        ax.set_ylabel("Mean residual (spikes/bin)")
        ax.set_title(f"Population residual profile\n{col}")
        ax.legend(fontsize=8)
        sns.despine(ax=ax)

    fig.suptitle("Population mean residual profiles", fontsize=13)


def plot_model_improvement(diag_dfs, reference, scalars=None):
    """
    Distribution of per-unit improvement (Δ) relative to a reference model.

    For each non-reference model and each scalar, computes:
        Δ = scalar_target − scalar_reference  (per unit, matched on unit ID)

    Negative Δ = improvement over reference (lower is better for D, η², drift).
    Positive Δ for ks_z_autocorr has no natural direction — shown for completeness.

    One panel per scalar, one violin per target model. Zero line = no change.

    Parameters
    ----------
    diag_dfs   : dict {model_label: DataFrame} — must include reference key
    reference  : str, key of the reference model (e.g. "null_model_all")
    scalars    : list of column names to compare; defaults to all diagnostic scalars
    """
    ref_df = diag_dfs[reference][diag_dfs[reference]["converged"] == True].set_index("unit")

    if scalars is None:
        scalars = (["ks_D", "drift_auc", "ks_z_autocorr"]
                   + [c for c in ref_df.columns if c.startswith("resid_eta2_")])
    scalars = [s for s in scalars if s in ref_df.columns]

    target_labels = [k for k in diag_dfs if k != reference]
    palette       = sns.color_palette("tab10", len(target_labels))

    rows = []
    for label in target_labels:
        tgt = diag_dfs[label][diag_dfs[label]["converged"] == True].set_index("unit")
        shared = ref_df.index.intersection(tgt.index)
        for s in scalars:
            if s not in tgt.columns:
                continue
            delta = tgt.loc[shared, s] - ref_df.loc[shared, s]
            for uid, val in delta.items():
                rows.append(dict(unit=uid, model=label, scalar=s, delta=val))

    delta_df = pd.DataFrame(rows)
    n_scalars = len(scalars)

    fig, axes = plt.subplots(1, n_scalars,
                             figsize=(4.5 * n_scalars, 5),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for k, s in enumerate(scalars):
        ax   = axes[k]
        sub  = delta_df[delta_df["scalar"] == s]
        sns.violinplot(data=sub, x="model", y="delta", order=target_labels,
                       palette=palette, inner="box", cut=0, ax=ax)
        ax.axhline(0, color="red", lw=1, ls="--", label="no change")
        ax.set_ylabel(f"Δ {s}")
        ax.set_title(s)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(fontsize=7)
        sns.despine(ax=ax)

    fig.suptitle(f"Model improvement vs reference: {reference}", fontsize=13)


def plot_residual_rms(profiles_dicts, covariate_cols=None):
    """
    Distribution of per-unit RMS of binned residual profiles, one panel per covariate.

    For each neuron and covariate, computes:
        RMS = sqrt(mean(binned_profile²))   over bins

    This captures how much systematic structure remains in the residuals without
    positive/negative cancellation, and is insensitive to number of bins.
    A model that accounts for a variable will produce flatter profiles → lower RMS.

    Parameters
    ----------
    profiles_dicts : dict {model_label: profiles_dict}
                     profiles_dict is the output of compute_residual_profiles
    covariate_cols : list of covariate names; inferred from first model if None
    """
    if not isinstance(profiles_dicts, dict) or "unit_ids" in profiles_dicts:
        profiles_dicts = {"model": profiles_dicts}

    if covariate_cols is None:
        first = next(iter(profiles_dicts.values()))
        covariate_cols = [k.replace("_profiles", "")
                          for k in first if k.endswith("_profiles")]

    model_order = list(profiles_dicts.keys())
    palette     = sns.color_palette("tab10", len(model_order))
    n_covs      = len(covariate_cols)

    fig, axes = plt.subplots(1, n_covs, figsize=(5 * n_covs, 5),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for k, col in enumerate(covariate_cols):
        ax = axes[k]
        rows = []
        for label, prof in profiles_dicts.items():
            key = f"{col}_profiles"
            if key not in prof:
                continue
            mat = np.array(prof[key])           # (n_units, n_bins)
            rms = np.sqrt(np.nanmean(mat ** 2, axis=1))  # (n_units,)
            for v in rms:
                rows.append({"model": label, "rms": v})

        if not rows:
            continue

        sub = pd.DataFrame(rows)
        sns.violinplot(data=sub, x="model", y="rms", order=model_order,
                       palette=palette, inner="box", cut=0, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("RMS binned residual (spikes/bin)")
        ax.set_title(col)
        ax.tick_params(axis="x", rotation=20)
        sns.despine(ax=ax)

    fig.suptitle("Per-unit RMS of binned residual profiles", fontsize=13)
    return fig


def plot_residual_rms_pair_profiles(prof_a, prof_b,
                                    label_a="model A", label_b="model B",
                                    covariate_cols=None):
    """
    Pairwise scatter of per-unit residual-profile RMS for two models.

    For each shared unit and covariate, computes:
        RMS = sqrt(mean(profile^2)) over profile bins

    Parameters
    ----------
    prof_a, prof_b : dict
        Profile dicts as returned by ``compute_model_diagnostics`` /
        ``load_model_outputs`` (must include ``unit_ids``).
    label_a, label_b : str
        Axis labels for model A/B.
    covariate_cols : list[str], optional
        Covariates to plot (continuous and/or categorical). If None, defaults
        to ``linear_position, speed, trial_type`` (only those three), filtered
        to whichever are present in both profile dicts.

    Returns
    -------
    fig, axes, shared_unit_ids
    """
    def _continuous_covariates(prof):
        return {
            k.replace("_profiles", "")
            for k in prof.keys()
            if k.endswith("_profiles") and not k.endswith("_cat_profiles")
        }

    def _categorical_covariates(prof):
        return {
            k.replace("_cat_profiles", "")
            for k in prof.keys()
            if k.endswith("_cat_profiles")
        }

    shared_cont = _continuous_covariates(prof_a) & _continuous_covariates(prof_b)
    shared_cat  = _categorical_covariates(prof_a) & _categorical_covariates(prof_b)
    shared_any  = shared_cont | shared_cat

    if covariate_cols is None:
        preferred_order = ["linear_position", "speed", "trial_type"]
        covariate_cols = [c for c in preferred_order if c in shared_any]
    else:
        covariate_cols = [c for c in covariate_cols if c in shared_any]

    if len(covariate_cols) == 0:
        raise ValueError("No shared residual-profile covariates found.")

    ids_a = np.asarray(prof_a["unit_ids"])
    ids_b = np.asarray(prof_b["unit_ids"])
    shared, idx_a, idx_b = np.intersect1d(ids_a, ids_b, return_indices=True)
    if len(shared) == 0:
        raise ValueError("No shared unit_ids between profile dicts.")

    n_covs = len(covariate_cols)
    fig, axes = plt.subplots(1, n_covs, figsize=(4.5 * n_covs, 4.5),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)

    for ax, col in zip(axes, covariate_cols):
        if col in shared_cont:
            key = f"{col}_profiles"
        else:
            key = f"{col}_cat_profiles"

        mat_a = np.asarray(prof_a[key], dtype=float)[idx_a]
        mat_b = np.asarray(prof_b[key], dtype=float)[idx_b]
        rms_a = np.sqrt(np.nanmean(mat_a ** 2, axis=1))
        rms_b = np.sqrt(np.nanmean(mat_b ** 2, axis=1))

        finite_max = np.nanmax(np.r_[rms_a, rms_b])
        lim = 1.0 if not np.isfinite(finite_max) else max(1e-9, finite_max) * 1.1

        ax.scatter(rms_a, rms_b, s=12, alpha=0.5, color="steelblue")
        ax.plot([0, lim], [0, lim], color="red", lw=1, ls="--", label="no change")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel(f"{label_a} RMS (spikes/bin)")
        ax.set_ylabel(f"{label_b} RMS (spikes/bin)")
        ax.set_title(col)
        ax.legend(fontsize=8)
        sns.despine(ax=ax)

    fig.suptitle(
        f"Per-neuron RMS of binned residual profiles: {label_a} vs {label_b} "
        f"(n={len(shared)} units)",
        fontsize=12,
    )
    return fig, axes, shared


def plot_residual_rms_pair_models(model_a, model_b,
                                  label_a=None, label_b=None,
                                  base_dir=None, cfg=None,
                                  fit_history_a=False, fit_history_b=False,
                                  covariate_cols=None):
    """
    Convenience wrapper: load profile outputs by model name and make pairwise RMS scatter.

    Parameters
    ----------
    model_a, model_b : str
        Model keys/stems accepted by ``load_model_outputs``.
    label_a, label_b : str, optional
        Display labels for axes. Defaults to resolved model names.
    base_dir, cfg : optional
        Forwarded to ``load_model_outputs``.
    fit_history_a, fit_history_b : bool
        If True, resolve each model to its ``*_history`` counterpart.
    covariate_cols : list[str], optional
        Forwarded to ``plot_residual_rms_pair_profiles``.

    Returns
    -------
    fig, axes, shared_unit_ids
    """
    cfg = cfg or CONFIG
    resolved_a = resolve_model_name(model_a, fit_history=fit_history_a, cfg=cfg)
    resolved_b = resolve_model_name(model_b, fit_history=fit_history_b, cfg=cfg)

    _, prof_a = load_model_outputs(
        model_a, base_dir=base_dir, cfg=cfg, fit_history=fit_history_a
    )
    _, prof_b = load_model_outputs(
        model_b, base_dir=base_dir, cfg=cfg, fit_history=fit_history_b
    )

    label_a = resolved_a if label_a is None else label_a
    label_b = resolved_b if label_b is None else label_b
    return plot_residual_rms_pair_profiles(
        prof_a=prof_a,
        prof_b=prof_b,
        label_a=label_a,
        label_b=label_b,
        covariate_cols=covariate_cols,
    )


# 5. Visualization 

_plot_state = {
    "rate_matrix":  None,
    "pos_pred_vals": None,
    "pos_run":       None,
    "peak_pos_cm":   None,
}


def set_plot_state(rate_matrix, pos_pred_vals, pos_run, peak_pos_cm):
    _plot_state["rate_matrix"]   = rate_matrix
    _plot_state["pos_pred_vals"] = pos_pred_vals
    _plot_state["pos_run"]       = pos_run
    _plot_state["peak_pos_cm"]   = peak_pos_cm


def plot_place_field(uid, ax_curve=None, ax_track=None, graph=None):
    """Rate curve + track heatmap for one unit. Pass axes to embed in a grid."""
    rate_matrix   = _plot_state["rate_matrix"]
    pos_pred_vals = _plot_state["pos_pred_vals"]
    pos_run       = _plot_state["pos_run"]
    peak_pos_cm   = _plot_state["peak_pos_cm"]

    rate = rate_matrix[uid]
    standalone = ax_curve is None
    if standalone:
        fig, (ax_curve, ax_track) = plt.subplots(1, 2, figsize=(13, 4))

 # rate curve
    ax_curve.plot(pos_pred_vals, rate, color="steelblue", lw=1.5)
    ax_curve.axvline(peak_pos_cm[uid], color="red", lw=1, ls="--",
                     label=f"peak @ {peak_pos_cm[uid]:.0f} cm")
    ax_curve.set_xlabel("linear position (cm)")
    ax_curve.set_ylabel("firing rate (Hz)")
    ax_curve.set_title(f"unit {uid}")
    ax_curve.legend(fontsize=8)
    sns.despine(ax=ax_curve)

 # track heatmap
    track_rate = np.interp(pos_run["linear_position"], pos_pred_vals, rate)
    if graph is None:
        import spyglass.linearization.v1 as sgpl
        graph = sgpl.TrackGraph & {"track_graph_name": CONFIG["wtrack_name"]}
    graph.plot_track_graph(ax=ax_track, draw_edge_labels=False)
    for ln in ax_track.lines:
        ln.set_color("lightgrey")
    sc = ax_track.scatter(pos_run["projected_x_position"], pos_run["projected_y_position"],
                          c=track_rate, cmap="hot_r", s=3, zorder=3,
                          vmin=rate.min(), vmax=rate.max())
    plt.colorbar(sc, ax=ax_track, label="Hz", shrink=0.8)
    ax_track.set_title(f"unit {uid} — peak {peak_pos_cm[uid]:.0f} cm")
    ax_track.set_xlabel("x (cm)"); ax_track.set_ylabel("y (cm)")

    if standalone:
        plt.tight_layout()


def plot_tuning_comparison(results, syn_df, scaled_col, actual_col, cov_df,
                           min_val=None, max_val=None, xlabel="Variable",
                           bin_size=0.002, n_bins=50, ax=None):
    """Overlay GLM predicted vs empirical occupancy-normalized tuning curve."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(7, 4))

    # Predicted curve
    pred_hz = results.predict(syn_df) / bin_size
    if min_val is not None and max_val is not None:
        x_pred = syn_df[scaled_col].values * (max_val - min_val) + min_val
    else:
        x_pred = syn_df[scaled_col].values

    # Empirical occupancy-normalized rate
    valid = cov_df[actual_col].notna()
    actual_vals = cov_df.loc[valid, actual_col].values
    spike_vals  = cov_df.loc[valid, "spike_count"].values

    occ,  bins = np.histogram(actual_vals, bins=n_bins)
    spks, _    = np.histogram(actual_vals, bins=bins, weights=spike_vals)
    with np.errstate(invalid="ignore", divide="ignore"):
        empirical_rate = np.where(occ > 0, spks / (occ * bin_size), np.nan)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    ax.plot(bin_centers, empirical_rate, color="grey", alpha=0.7, lw=1.2, label="Empirical")
    ax.plot(x_pred, pred_hz, color="steelblue", lw=2, label="GLM predicted")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Firing rate (Hz)")
    ax.legend(fontsize=8)
    sns.despine(ax=ax)

    if standalone:
        plt.tight_layout()


def _get_linearized_branch_axis(cfg=None, points_per_segment=200):
    """Return plotting metadata for a split-junction, duplicated-stem display."""
    cfg = cfg or CONFIG
    position_df = pd.read_csv(cfg["trialized_position"], usecols=["track_segment_id", "linear_position"])
    position_df = position_df.dropna(subset=["track_segment_id", "linear_position"]).copy()
    position_df["track_segment_id"] = position_df["track_segment_id"].astype(int)

    seg_bounds = _segment_position_bounds(position_df)
    seg_len = {seg: seg_bounds[seg]["max"] - seg_bounds[seg]["min"] for seg in seg_bounds}
    top_max = seg_len[1] + seg_len[3]
    stem_max = seg_len[0]
    bottom_max = seg_len[2] + seg_len[4]
    gap_size = 10.0

    sections = [
        dict(seg=3, edge_label="3", node_left="3", node_right="2",
             branch_id="top", display_section="upper",
             branch_pos_start=top_max, branch_pos_end=seg_len[1]),
        dict(seg=1, edge_label="1", node_left="2", node_right="1",
             branch_id="top", display_section="upper",
             branch_pos_start=seg_len[1], branch_pos_end=0.0),
        dict(seg=None, gap=True),
        dict(seg=0, edge_label="0", node_left="1", node_right="0",
             branch_id="middle", display_section="stem_left",
             branch_pos_start=0.0, branch_pos_end=stem_max),
        dict(seg=0, edge_label="0", node_left="0", node_right="1",
             branch_id="middle", display_section="stem_right",
             branch_pos_start=stem_max, branch_pos_end=0.0),
        dict(seg=None, gap=True),
        dict(seg=2, edge_label="2", node_left="1", node_right="4",
             branch_id="bottom", display_section="lower",
             branch_pos_start=0.0, branch_pos_end=seg_len[2]),
        dict(seg=4, edge_label="4", node_left="4", node_right="5",
             branch_id="bottom", display_section="lower",
             branch_pos_start=seg_len[2], branch_pos_end=bottom_max),
    ]

    current_x = 0.0
    gap_centers = []
    branch_spans = {"top": [], "middle": [], "bottom": []}
    track_sections = []
    rows = []
    for section in sections:
        if section.get("gap"):
            gap_centers.append(current_x + gap_size / 2.0)
            rows.append(pd.DataFrame({
                "track_segment_id": [np.nan],
                "linear_position": [np.nan],
                "linear_position_nogap": [np.nan],
                "branch_id": [pd.NA],
                "display_section": [pd.NA],
                "branch_pos_cm": [np.nan],
                "branch_pos_scaled": [np.nan],
            }))
            current_x += gap_size
            continue

        seg = section["seg"]
        seg_length = seg_len[seg]
        x = np.linspace(current_x, current_x + seg_length, points_per_segment)
        branch_pos = np.linspace(section["branch_pos_start"], section["branch_pos_end"], points_per_segment)
        branch_max = {"top": top_max, "middle": stem_max, "bottom": bottom_max}[section["branch_id"]]
        rows.append(pd.DataFrame({
            "track_segment_id": seg,
            "linear_position": x,
            "linear_position_nogap": x,
            "branch_id": section["branch_id"],
            "display_section": section["display_section"],
            "branch_pos_cm": branch_pos,
            "branch_pos_scaled": np.where(branch_max > 0, branch_pos / branch_max, 0.0),
        }))

        track_sections.append({
            "x0": current_x,
            "x1": current_x + seg_length,
            "edge_label": section["edge_label"],
            "node_left": section["node_left"],
            "node_right": section["node_right"],
        })
        current_x += seg_length

    branch_spans["top"].append({"xmin": track_sections[0]["x0"], "xmax": track_sections[1]["x1"]})
    branch_spans["middle"].append({"xmin": track_sections[2]["x0"], "xmax": track_sections[2]["x1"]})
    branch_spans["middle"].append({"xmin": track_sections[3]["x0"], "xmax": track_sections[3]["x1"]})
    branch_spans["bottom"].append({"xmin": track_sections[4]["x0"], "xmax": track_sections[5]["x1"]})

    grid = pd.concat(rows, ignore_index=True)
    grid["branch_id"] = pd.Categorical(
        grid["branch_id"], categories=list(BRANCH_SPECS.keys()), ordered=True
    )

    return {
        "seg_bounds": seg_bounds,
        "gap_centers": gap_centers,
        "stem_duplicate_center": 0.5 * (track_sections[2]["x1"] + track_sections[3]["x0"]),
        "branch_spans": branch_spans,
        "track_sections": track_sections,
        "grid": grid,
    }


def _draw_repeated_stem_track(ax, axis_info):
    """Draw the duplicated-stem reference track used by the branch plots."""
    y0 = 0.0
    node_positions = []
    for i, section in enumerate(axis_info["track_sections"]):
        ax.plot([section["x0"], section["x1"]], [y0, y0], color="k", lw=2.0, zorder=1)
        ax.text(
            0.5 * (section["x0"] + section["x1"]), y0,
            section["edge_label"], ha="center", va="center", fontsize=12, color="0.15"
        )
        node_positions.append((section["x0"], section["node_left"]))
        if i == len(axis_info["track_sections"]) - 1:
            node_positions.append((section["x1"], section["node_right"]))
        elif axis_info["track_sections"][i + 1]["x0"] != section["x1"]:
            node_positions.append((section["x1"], section["node_right"]))

    xs = [x for x, _ in node_positions]
    ax.scatter(xs, [y0] * len(xs), s=320, color="#1f77b4", zorder=3)
    for x, label in node_positions:
        ax.text(x, y0, str(label), ha="center", va="center", fontsize=11, color="black", zorder=4)

    ax.set_ylim(-0.35, 0.35)
    ax.set_yticks([])
    ax.set_ylabel("")
    sns.despine(ax=ax, left=True)


def plot_linearized_branch_comparison(results, cov_df, n_bins=120, points_per_segment=200,
                                      bin_size=None, ax=None, title=None,
                                      emp_bin_cm=None, smooth_sigma_bins=None):
    """Overlay empirical and branch-model predicted rates on the gapped linearized axis."""
    if bin_size is None:
        bin_size = CONFIG["bin_size"]

    standalone = ax is None
    ax_track = None
    if standalone:
        fig, (ax, ax_track) = plt.subplots(
            2, 1, figsize=(10, 5.2), sharex=True,
            gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08}
        )

    pred_grid, axis_info = _make_linearized_branch_curve_frame(
        results, cov_df, n_bins=n_bins, points_per_segment=points_per_segment,
        bin_size=bin_size, emp_bin_cm=emp_bin_cm,
        smooth_sigma_bins=smooth_sigma_bins,
    )

    for x_gap in axis_info.get("gap_centers", []):
        ax.axvline(x_gap, color="k", lw=1.5, ls="--", zorder=0, alpha = 0.8)
    x_dup = axis_info.get("stem_duplicate_center")
    if x_dup is not None:
        ax.axvline(x_dup, color="k", lw=1.75, ls="-", zorder=0, alpha=1)

    ax.plot(pred_grid["linear_position"], pred_grid["emp_rate_hz"],
            color="grey", alpha=0.8, lw=1.3, label="Empirical")
    ax.plot(pred_grid["linear_position"], pred_grid["pred_rate_hz"],
            color="steelblue", lw=2.0, label="GLM predicted")

    y_vals = np.concatenate([
        pred_grid["emp_rate_hz"].to_numpy(dtype=float),
        pred_grid["pred_rate_hz"].to_numpy(dtype=float),
    ])
    finite_y = y_vals[np.isfinite(y_vals)]
    if finite_y.size:
        y_max = float(finite_y.max())
        y_min = float(finite_y.min())
        y_pad = max(1.0, 0.08 * max(y_max - y_min, 1.0))
        ax.set_ylim(top=y_max + y_pad)
        label_y = y_max + 0.6 * y_pad
    else:
        _, current_top = ax.get_ylim()
        label_y = current_top

    ax.set_xlabel("")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(title or "Branch-aware position model on linearized track")
    ax.legend(fontsize=8)
    ax.xaxis.set_visible(False)
    sns.despine(ax=ax, bottom=True)

    label_map = {"top": "upper", "middle": "stem", "bottom": "lower"}
    for branch_name, label in label_map.items():
        spans = axis_info.get("branch_spans", {}).get(branch_name, [])
        for span in spans:
            x_center = 0.5 * (span["xmin"] + span["xmax"])
            ax.text(
                x_center, label_y, label,
                ha="center", va="bottom", fontsize=12, fontweight="semibold", color="0.25",
                bbox=dict(facecolor="white", edgecolor="none", pad=0.2, alpha=0.9)
            )

    if ax_track is not None:
        _draw_repeated_stem_track(ax_track, axis_info)
        ax_track.set_xlim(ax.get_xlim())
        ax_track.set_xlabel("Linear position (cm)")

    if standalone:
        plt.tight_layout()


def plot_categorical_comparison(results, cat_col, cov_df, bin_size=0.002, ax=None):
    """Grouped bar chart: GLM predicted vs empirical rate per category."""
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 4))

    cats = sorted(cov_df[cat_col].dropna().unique())
    syn_df = pd.DataFrame({cat_col: cats})
    pred_hz = results.predict(syn_df).values / bin_size

    grouped = cov_df.groupby(cat_col)
    emp_hz = (grouped["spike_count"].sum() / (grouped.size() * bin_size)).reindex(cats).values

    x = np.arange(len(cats))
    width = 0.35
    ax.bar(x - width / 2, emp_hz,  width, label="Empirical",     color="grey",      alpha=0.8)
    ax.bar(x + width / 2, pred_hz, width, label="GLM predicted", color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_xlabel(cat_col)
    ax.legend(fontsize=8)
    sns.despine(ax=ax)

    if standalone:
        plt.tight_layout()


def compute_marginal_effect(results, term_pattern, sweep_values,
                            sweep_col, cov_df, bin_size=None):
    """Compute marginal firing-rate curve for one term, averaging over all others.

    For each sweep point, the linear predictor is decomposed into
    ``η_term`` (columns matching *term_pattern*) and ``η_other`` (everything
    else, evaluated at each observed data point).  The marginal rate is
    ``mean_over_data[ exp(η_term + η_other) ] / bin_size``.

    Parameters
    ----------
    results : statsmodels GLMResults
        Fitted multi-variable model.
    term_pattern : str
        Substring matched against ``results.model.exog_names`` to identify the
        columns belonging to the term (e.g. ``"pos_scaled"``, ``"speed_scaled"``,
        ``"trial_type"``).
    sweep_values : array-like
        Values to sweep the term over.  For continuous terms these are the
        *scaled* values that enter the formula (e.g. 0–1 for ``pos_scaled``).
        For categorical terms, pass the category labels.
    sweep_col : str
        Column name in the design DataFrame that receives *sweep_values*
        (e.g. ``"pos_scaled"``).
    cov_df : DataFrame
        The covariate DataFrame used during fitting (must contain all columns
        referenced by the formula).  Used both to construct the sweep design
        matrix (via Patsy ``design_info``) and to evaluate ``η_other``.
    bin_size : float, optional
        Time-bin width in seconds.  Defaults to ``CONFIG["bin_size"]``.

    Returns
    -------
    marginal_hz : ndarray, shape ``(len(sweep_values),)``
        Marginal firing rate in Hz at each sweep point.
    """
    import patsy as _patsy

    if bin_size is None:
        bin_size = CONFIG["bin_size"]

    X_fit = np.asarray(results.model.exog)
    params = np.asarray(results.params)
    col_names = results.model.exog_names

    term_cols = [i for i, n in enumerate(col_names) if term_pattern in n]
    other_cols = [i for i in range(len(col_names)) if i not in term_cols]
    eta_other = X_fit[:, other_cols] @ params[other_cols]

    # Build sweep design matrix using the model's own design_info.
    # Use results.model.data.frame as the source so that unit-specific columns
    # injected by per_unit_transform (e.g. spike-history covariates) are present.
    design_info = results.model.data.orig_exog.design_info
    n_sweep = len(sweep_values)
    fit_frame = results.model.data.frame.reset_index(drop=True)
    if fit_frame.empty:
        raise ValueError("Cannot build marginal curve from an empty fitted design frame")

    sweep_df = fit_frame.iloc[np.zeros(n_sweep, dtype=int)].copy().reset_index(drop=True)
    sweep_df[sweep_col] = np.asarray(sweep_values)
    # Fill non-swept columns with neutral values so patsy can evaluate
    for col in fit_frame.columns:
        if col == sweep_col or col == "spike_count":
            continue
        if pd.api.types.is_numeric_dtype(fit_frame[col]):
            sweep_df[col] = fit_frame[col].median()
        else:
            mode = fit_frame[col].mode(dropna=True)
            if len(mode) > 0:
                sweep_df[col] = mode.iloc[0]
            else:
                non_na = fit_frame[col].dropna()
                sweep_df[col] = non_na.iloc[0] if len(non_na) else np.nan

    X_sweep = np.asarray(
        _patsy.build_design_matrices([design_info], sweep_df,
                                     return_type="dataframe")[0]
    )
    eta_term = X_sweep[:, term_cols] @ params[term_cols]

    # Average exp(η_term + η_other) over all observed data points
    marginal_hz = np.exp(eta_term[:, None] + eta_other[None, :]).mean(axis=1) / bin_size
    return marginal_hz


def compute_marginal_effect_multi(results, term_patterns, sweep_df, bin_size=None):
    """Compute a marginal rate curve when sweeping multiple related predictors."""
    import patsy as _patsy

    if bin_size is None:
        bin_size = CONFIG["bin_size"]

    sweep_df = pd.DataFrame(sweep_df).reset_index(drop=True)
    if sweep_df.empty:
        return np.array([])

    if isinstance(term_patterns, str):
        term_patterns = (term_patterns,)

    X_fit = np.asarray(results.model.exog, dtype=float)
    col_names = list(results.model.exog_names)
    params_raw = results.params
    if isinstance(params_raw, pd.Series):
        coef_series = params_raw.reindex(col_names).fillna(0.0)
    else:
        coef_series = pd.Series(np.asarray(params_raw, dtype=float), index=col_names)
    params = coef_series.to_numpy(dtype=float)

    term_cols = [
        i for i, name in enumerate(col_names)
        if any(pattern in name for pattern in term_patterns)
    ]
    if not term_cols:
        raise ValueError(
            f"Could not find any design columns matching patterns {tuple(term_patterns)!r}"
        )

    other_cols = [i for i in range(len(col_names)) if i not in term_cols]
    if other_cols:
        eta_other = X_fit[:, other_cols] @ params[other_cols]
    else:
        eta_other = np.zeros(X_fit.shape[0], dtype=float)

    design_info = results.model.data.orig_exog.design_info
    fit_frame = results.model.data.frame.reset_index(drop=True)
    if fit_frame.empty:
        raise ValueError("Cannot build marginal curve from an empty fitted design frame")

    template = fit_frame.iloc[np.zeros(len(sweep_df), dtype=int)].copy().reset_index(drop=True)
    for col in template.columns:
        if col == "spike_count":
            continue
        if col in sweep_df.columns:
            template[col] = sweep_df[col].values
            continue

        source = fit_frame[col]
        if pd.api.types.is_numeric_dtype(source):
            template[col] = source.median()
        else:
            mode = source.mode(dropna=True)
            if len(mode) > 0:
                template[col] = mode.iloc[0]
            else:
                non_na = source.dropna()
                template[col] = non_na.iloc[0] if len(non_na) else np.nan

    X_sweep = np.asarray(
        _patsy.build_design_matrices([design_info], template, return_type="dataframe")[0]
    )
    eta_term = X_sweep[:, term_cols] @ params[term_cols]
    marginal_hz = np.exp(eta_term[:, None] + eta_other[None, :]).mean(axis=1) / bin_size
    return marginal_hz


def empirical_tuning_curve(cov_df, actual_col, bin_size=None, n_bins=50,
                           categorical=False):
    """Compute occupancy-normalized empirical firing-rate curve.

    Parameters
    ----------
    cov_df : DataFrame
        Must contain ``actual_col`` and ``"spike_count"`` columns.
    actual_col : str
        Column with the behavioural variable (e.g. ``"linear_position"``).
    bin_size : float, optional
    n_bins : int
        Number of histogram bins (ignored for categorical).
    categorical : bool
        If True, group by unique values instead of histogram binning.

    Returns
    -------
    x_vals, rate_hz : ndarrays
        Bin centres (or category labels) and firing rate in Hz.
    """
    if bin_size is None:
        bin_size = CONFIG["bin_size"]

    if categorical:
        cats = sorted(cov_df[actual_col].dropna().unique())
        grouped = cov_df.groupby(actual_col)
        rate_hz = (grouped["spike_count"].sum() / (grouped.size() * bin_size)).reindex(cats).values
        return np.array(cats), rate_hz

    valid = cov_df[actual_col].notna()
    vals = cov_df.loc[valid, actual_col].values
    spk  = cov_df.loc[valid, "spike_count"].values
    occ, bins = np.histogram(vals, bins=n_bins)
    spks, _   = np.histogram(vals, bins=bins, weights=spk)
    with np.errstate(invalid="ignore", divide="ignore"):
        rate_hz = np.where(occ > 0, spks / (occ * bin_size), np.nan)
    centers = (bins[:-1] + bins[1:]) / 2
    return centers, rate_hz


def _path_coordinate(branch_id, branch_pos_cm, stem_max, outer_branch):
    """Map branch-local coordinates to a continuous path coordinate."""
    branch_id = np.asarray(branch_id)
    branch_pos_cm = np.asarray(branch_pos_cm, dtype=float)
    out = np.full(len(branch_pos_cm), np.nan, dtype=float)

    stem_mask = branch_id == "middle"
    outer_mask = branch_id == outer_branch
    out[stem_mask] = stem_max - branch_pos_cm[stem_mask]
    out[outer_mask] = stem_max + branch_pos_cm[outer_mask]
    return out


def _estimate_empirical_path_curve(cov_valid, stem_max, outer_branch, n_bins, bin_size,
                                   emp_bin_cm=None, smooth_sigma_bins=None):
    """Estimate empirical firing rate on one continuous stem+arm path."""
    path_mask = cov_valid["branch_id"].isin(["middle", outer_branch])
    path_df = cov_valid.loc[path_mask].copy()
    if path_df.empty:
        return np.array([]), np.array([])

    x = _path_coordinate(
        path_df["branch_id"].to_numpy(),
        path_df["branch_pos_cm"].to_numpy(dtype=float),
        stem_max=stem_max,
        outer_branch=outer_branch,
    )
    w = path_df["spike_count"].to_numpy(dtype=float)
    valid = np.isfinite(x)
    x = x[valid]
    w = w[valid]
    if len(x) == 0:
        return np.array([]), np.array([])

    x_min = 0.0
    x_max = max(stem_max, float(np.nanmax(x)))
    if emp_bin_cm is not None:
        edges = np.arange(x_min, x_max + emp_bin_cm, emp_bin_cm, dtype=float)
        if len(edges) < 2 or edges[-1] < x_max:
            edges = np.append(edges, x_max)
    else:
        edges = np.linspace(x_min, x_max, n_bins + 1)

    occ, _ = np.histogram(x, bins=edges)
    spks, _ = np.histogram(x, bins=edges, weights=w)
    occ = occ.astype(float)
    spks = spks.astype(float)
    if smooth_sigma_bins is not None and smooth_sigma_bins > 0:
        occ = gaussian_filter1d(occ, smooth_sigma_bins, mode="nearest")
        spks = gaussian_filter1d(spks, smooth_sigma_bins, mode="nearest")

    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(occ > 0, spks / (occ * bin_size), np.nan)
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, rate


def _fill_empirical_pathwise_curve(pred_grid, cov_valid, n_bins, bin_size,
                                   emp_bin_cm=None, smooth_sigma_bins=None):
    """Fill empirical rate using one curve on 0->3 and one on 0->5."""
    if emp_bin_cm is not None and emp_bin_cm <= 0:
        raise ValueError("emp_bin_cm must be positive")
    if smooth_sigma_bins is not None and smooth_sigma_bins < 0:
        raise ValueError("smooth_sigma_bins must be non-negative")

    stem_max = float(pred_grid.loc[pred_grid["branch_id"] == "middle", "branch_pos_cm"].max())
    top_max = float(pred_grid.loc[pred_grid["branch_id"] == "top", "branch_pos_cm"].max())
    path_specs = (
        ("top", ("upper", "stem_left")),
        ("bottom", ("stem_right", "lower")),
    )

    for outer_branch, display_sections in path_specs:
        centers, rate = _estimate_empirical_path_curve(
            cov_valid,
            stem_max=stem_max,
            outer_branch=outer_branch,
            n_bins=n_bins,
            bin_size=bin_size,
            emp_bin_cm=emp_bin_cm,
            smooth_sigma_bins=smooth_sigma_bins,
        )
        valid_rate = np.isfinite(rate)
        if valid_rate.sum() == 0:
            continue

        branch_mask = pred_grid["display_section"].isin(display_sections)
        branch_df = pred_grid.loc[branch_mask, ["branch_id", "branch_pos_cm", "display_section"]].copy()
        if branch_df.empty:
            continue

        branch_pos = np.full(len(branch_df), np.nan, dtype=float)
        if outer_branch == "top":
            upper_mask = branch_df["display_section"] == "upper"
            stem_mask = branch_df["display_section"] == "stem_left"
            branch_pos[upper_mask] = stem_max + branch_df.loc[upper_mask, "branch_pos_cm"].to_numpy(dtype=float)
            branch_pos[stem_mask] = stem_max - branch_df.loc[stem_mask, "branch_pos_cm"].to_numpy(dtype=float)
        else:
            stem_mask = branch_df["display_section"] == "stem_right"
            lower_mask = branch_df["display_section"] == "lower"
            branch_pos[stem_mask] = stem_max - branch_df.loc[stem_mask, "branch_pos_cm"].to_numpy(dtype=float)
            branch_pos[lower_mask] = stem_max + branch_df.loc[lower_mask, "branch_pos_cm"].to_numpy(dtype=float)

        if valid_rate.sum() >= 2:
            pred_grid.loc[branch_mask, "emp_rate_hz"] = np.interp(
                branch_pos, centers[valid_rate], rate[valid_rate]
            )
        else:
            pred_grid.loc[branch_mask, "emp_rate_hz"] = rate[valid_rate][0]


def _make_linearized_branch_curve_frame(results, cov_df, n_bins=120,
                                        points_per_segment=200, bin_size=None,
                                        emp_bin_cm=None, smooth_sigma_bins=None):
    """Return branch-model predicted and empirical rates on the gapped track axis."""
    if bin_size is None:
        bin_size = CONFIG["bin_size"]

    axis_info = _get_linearized_branch_axis(points_per_segment=points_per_segment)
    pred_grid = axis_info["grid"].copy()
    pred_valid = pred_grid["linear_position"].notna()

    sweep_df = pred_grid.loc[pred_valid, ["branch_id", "branch_pos_scaled"]].copy()
    pred_hz = compute_marginal_effect_multi(
        results,
        term_patterns=("branch_id", "branch_pos_scaled"),
        sweep_df=sweep_df,
        bin_size=bin_size,
    )
    pred_grid["pred_rate_hz"] = np.nan
    pred_grid.loc[pred_valid, "pred_rate_hz"] = np.asarray(pred_hz, dtype=float)

    cov_valid = cov_df.loc[
        cov_df["branch_id"].notna()
        & cov_df["branch_pos_cm"].notna()
        & cov_df["spike_count"].notna(),
        ["branch_id", "branch_pos_cm", "spike_count"],
    ].copy()

    pred_grid["emp_rate_hz"] = np.nan
    if not cov_valid.empty:
        _fill_empirical_pathwise_curve(
            pred_grid, cov_valid, n_bins=n_bins, bin_size=bin_size,
            emp_bin_cm=emp_bin_cm, smooth_sigma_bins=smooth_sigma_bins,
        )

    return pred_grid, axis_info


def _infer_comparison_spec(model_key=None, formula=None, results=None):
    """Infer the most natural predicted-vs-empirical comparison axis for a model."""
    if formula is None and results is not None:
        formula = getattr(results.model, "formula", None)

    text_parts = []
    if model_key is not None:
        text_parts.append(str(model_key))
    if formula is not None:
        text_parts.append(str(formula))
    if results is not None:
        text_parts.extend(str(name) for name in getattr(results.model, "exog_names", []))
    text = " ".join(text_parts)

    if ("branch_pos_scaled" in text) or ("branch_id" in text):
        return {
            "kind": "linearized_branch",
            "xlabel": "Linear position (cm)",
        }
    if "pos_scaled" in text:
        return {
            "kind": "continuous",
            "term_pattern": "pos_scaled",
            "sweep_col": "pos_scaled",
            "actual_col": "linear_position",
            "xlabel": "Position (cm)",
            "scaled_from_actual": True,
        }
    if "trial_progress" in text:
        return {
            "kind": "continuous",
            "term_pattern": "trial_progress",
            "sweep_col": "trial_progress",
            "actual_col": "trial_progress",
            "xlabel": "Trial progress",
            "scaled_from_actual": False,
        }
    if "speed_scaled" in text:
        return {
            "kind": "continuous",
            "term_pattern": "speed_scaled",
            "sweep_col": "speed_scaled",
            "actual_col": "speed",
            "xlabel": "Speed (cm/s)",
            "scaled_from_actual": True,
        }
    if re.search(r"(?<![A-Za-z0-9_])choice(?![A-Za-z0-9_])", text):
        return {
            "kind": "categorical",
            "term_pattern": "choice",
            "sweep_col": "choice",
            "actual_col": "choice",
            "xlabel": "Choice",
        }
    if "trial_type" in text:
        return {
            "kind": "categorical",
            "term_pattern": "trial_type",
            "sweep_col": "trial_type",
            "actual_col": "trial_type",
            "xlabel": "Trial type",
        }
    return None


def _build_unit_fit_frame(cov_df, spike_counts_masked, unit_index,
                          per_unit_transform=None):
    """Return the unit-specific design DataFrame used for fitting/plotting."""
    df = cov_df.copy()
    df["spike_count"] = spike_counts_masked[unit_index]
    if per_unit_transform is not None:
        df = per_unit_transform(df, spike_counts_masked[unit_index], unit_index)
    return df


def _coerce_coef_dict(coef_entry):
    """Normalize a stored coefficient payload from fit_glm_all_units."""
    if isinstance(coef_entry, dict):
        raw = coef_entry
    elif isinstance(coef_entry, str):
        raw = ast.literal_eval(coef_entry)
    elif pd.isna(coef_entry):
        raw = {}
    else:
        raise TypeError(f"Unsupported coefficient payload type: {type(coef_entry)!r}")

    return {str(k): float(v) for k, v in raw.items()}


def _build_design_template(formula, cov_df):
    """Build reusable Patsy design metadata for a fixed covariate table."""
    import patsy as _patsy

    _, X_fit = _patsy.dmatrices(formula, cov_df, return_type="dataframe")
    design_info = X_fit.design_info
    fit_positions = cov_df.index.get_indexer(X_fit.index)
    X_fit = X_fit.reset_index(drop=True)
    return {
        "fit_positions": fit_positions,
        "design_info": design_info,
        "X_fit": np.asarray(X_fit, dtype=float),
        "col_names": list(X_fit.columns),
    }


def _build_prediction_context(formula, cov_df, coef_entry, design_template=None):
    """Build Patsy design metadata for prediction from stored coefficients."""
    if design_template is None:
        design_template = _build_design_template(formula, cov_df)

    cov_df_fit = cov_df.iloc[design_template["fit_positions"]].copy().reset_index(drop=True)
    coef_series = pd.Series(_coerce_coef_dict(coef_entry), dtype=float)
    coef_series = coef_series.reindex(design_template["col_names"]).fillna(0.0)

    return {
        "formula": formula,
        "cov_df_fit": cov_df_fit,
        "design_info": design_template["design_info"],
        "X_fit": design_template["X_fit"],
        "coef_series": coef_series,
        "col_names": design_template["col_names"],
    }


def _compute_marginal_from_context(ctx, term_patterns, sweep_df, bin_size=None):
    """Compute a marginal prediction curve from stored coefficients."""
    import patsy as _patsy

    if bin_size is None:
        bin_size = CONFIG["bin_size"]

    if isinstance(term_patterns, str):
        term_patterns = (term_patterns,)

    sweep_df = pd.DataFrame(sweep_df).reset_index(drop=True)
    if sweep_df.empty:
        return np.array([])

    col_names = ctx["col_names"]
    params = ctx["coef_series"].reindex(col_names).fillna(0.0).to_numpy(dtype=float)
    term_cols = [
        i for i, name in enumerate(col_names)
        if any(pattern in name for pattern in term_patterns)
    ]
    if not term_cols:
        raise ValueError(
            f"Could not find any design columns matching patterns {tuple(term_patterns)!r}"
        )

    other_cols = [i for i in range(len(col_names)) if i not in term_cols]
    if other_cols:
        eta_other = ctx["X_fit"][:, other_cols] @ params[other_cols]
    else:
        eta_other = np.zeros(ctx["X_fit"].shape[0], dtype=float)

    fit_frame = ctx["cov_df_fit"].reset_index(drop=True)
    template = fit_frame.iloc[np.zeros(len(sweep_df), dtype=int)].copy().reset_index(drop=True)
    for col in template.columns:
        if col == "spike_count":
            continue
        if col in sweep_df.columns:
            template[col] = sweep_df[col].values
            continue

        source = fit_frame[col]
        if pd.api.types.is_numeric_dtype(source):
            template[col] = source.median()
        else:
            mode = source.mode(dropna=True)
            if len(mode) > 0:
                template[col] = mode.iloc[0]
            else:
                non_na = source.dropna()
                template[col] = non_na.iloc[0] if len(non_na) else np.nan

    X_sweep = np.asarray(
        _patsy.build_design_matrices([ctx["design_info"]], template, return_type="dataframe")[0],
        dtype=float,
    )
    eta_term = X_sweep[:, term_cols] @ params[term_cols]
    return np.exp(eta_term[:, None] + eta_other[None, :]).mean(axis=1) / bin_size


def _make_linearized_branch_curve_frame_from_context(ctx, cov_df, n_bins=120,
                                                     points_per_segment=200, bin_size=None,
                                                     emp_bin_cm=None, smooth_sigma_bins=None):
    """Return branch-model predicted and empirical rates from stored coefficients."""
    if bin_size is None:
        bin_size = CONFIG["bin_size"]

    axis_info = _get_linearized_branch_axis(points_per_segment=points_per_segment)
    pred_grid = axis_info["grid"].copy()
    pred_valid = pred_grid["linear_position"].notna()
    sweep_df = pred_grid.loc[pred_valid, ["branch_id", "branch_pos_scaled"]].copy()

    pred_hz = _compute_marginal_from_context(
        ctx,
        term_patterns=("branch_id", "branch_pos_scaled"),
        sweep_df=sweep_df,
        bin_size=bin_size,
    )
    pred_grid["pred_rate_hz"] = np.nan
    pred_grid.loc[pred_valid, "pred_rate_hz"] = np.asarray(pred_hz, dtype=float)

    cov_valid = cov_df.loc[
        cov_df["branch_id"].notna()
        & cov_df["branch_pos_cm"].notna()
        & cov_df["spike_count"].notna(),
        ["branch_id", "branch_pos_cm", "spike_count"],
    ].copy()

    pred_grid["emp_rate_hz"] = np.nan
    if not cov_valid.empty:
        _fill_empirical_pathwise_curve(
            pred_grid, cov_valid, n_bins=n_bins, bin_size=bin_size,
            emp_bin_cm=emp_bin_cm, smooth_sigma_bins=smooth_sigma_bins,
        )

    return pred_grid, axis_info


def _chunk_unit_list(unit_list, chunk_size):
    """Yield consecutive unit-list chunks."""
    if chunk_size is None or chunk_size <= 0:
        yield list(unit_list)
        return
    for start in range(0, len(unit_list), chunk_size):
        yield list(unit_list[start:start + chunk_size])


def _plot_predicted_vs_actual_row(ctx, cov_df, spec, ax, bin_size=None,
                                  n_bins=120, n_sweep=300,
                                  points_per_segment=200, show_legend=False):
    """Draw one unit row for the batch comparison plot."""
    if bin_size is None:
        bin_size = CONFIG["bin_size"]

    kind = spec["kind"]
    if kind == "linearized_branch":
        curve_df, _ = _make_linearized_branch_curve_frame_from_context(
            ctx, cov_df, n_bins=n_bins, points_per_segment=points_per_segment,
            bin_size=bin_size,
        )
        ax.plot(
            curve_df["linear_position"], curve_df["emp_rate_hz"],
            color="grey", alpha=0.8, lw=1.3, label="Empirical",
        )
        ax.plot(
            curve_df["linear_position"], curve_df["pred_rate_hz"],
            color="steelblue", lw=2.0, label="GLM predicted",
        )
        ax.xaxis.set_visible(False)
        sns.despine(ax=ax, bottom=True)
    elif kind == "continuous":
        actual_col = spec["actual_col"]
        actual_vals = cov_df[actual_col].dropna().to_numpy()
        if len(actual_vals) == 0:
            raise ValueError(f"No valid values available for {actual_col!r}")

        natural = np.linspace(actual_vals.min(), actual_vals.max(), n_sweep)
        if spec.get("scaled_from_actual", False):
            denom = actual_vals.max() - actual_vals.min()
            if denom > 0:
                sweep_values = (natural - actual_vals.min()) / denom
            else:
                sweep_values = np.zeros_like(natural)
        else:
            sweep_values = natural

        marginal_hz = _compute_marginal_from_context(
            ctx,
            term_patterns=spec["term_pattern"],
            sweep_df=pd.DataFrame({spec["sweep_col"]: sweep_values}),
            bin_size=bin_size,
        )
        emp_x, emp_hz = empirical_tuning_curve(
            cov_df, actual_col, bin_size=bin_size, n_bins=n_bins,
        )

        ax.plot(emp_x, emp_hz, color="grey", alpha=0.8, lw=1.3, label="Empirical")
        ax.plot(natural, marginal_hz, color="steelblue", lw=2.0, label="GLM predicted")
        sns.despine(ax=ax)
    elif kind == "categorical":
        cats = sorted(cov_df[spec["actual_col"]].dropna().unique())
        if not cats:
            raise ValueError(f"No valid categories available for {spec['actual_col']!r}")

        marginal_hz = _compute_marginal_from_context(
            ctx,
            term_patterns=spec["term_pattern"],
            sweep_df=pd.DataFrame({spec["sweep_col"]: cats}),
            bin_size=bin_size,
        )
        _, emp_hz = empirical_tuning_curve(
            cov_df, spec["actual_col"], bin_size=bin_size, categorical=True,
        )

        x = np.arange(len(cats))
        width = 0.35
        ax.bar(x - width / 2, emp_hz, width, color="grey", alpha=0.8, label="Empirical")
        ax.bar(x + width / 2, marginal_hz, width, color="steelblue", alpha=0.8,
               label="GLM predicted")
        ax.set_xticks(x)
        ax.set_xticklabels(cats)
        sns.despine(ax=ax)
    else:
        raise ValueError(f"Unsupported comparison kind: {kind!r}")

    ax.set_ylabel("Hz")
    if show_legend:
        ax.legend(fontsize=8, loc="upper right")


def plot_predicted_vs_actual_by_unit(unit_list, formula, cov_df, spike_counts_masked,
                                     fit_df, unit_ids=None, per_unit_transform=None,
                                     model_key=None, bin_size=None, n_bins=120,
                                     n_sweep=300, points_per_segment=200,
                                     figsize_per_row=1.6, max_rows_per_fig=15,
                                     title=None):
    """Plot predicted vs empirical firing for selected units from stored fits."""
    if bin_size is None:
        bin_size = CONFIG["bin_size"]
    if unit_ids is None:
        unit_ids = np.arange(len(spike_counts_masked))

    unit_list = list(unit_list)
    if len(unit_list) == 0:
        raise ValueError("unit_list must contain at least one unit")

    spec = _infer_comparison_spec(model_key=model_key, formula=formula)
    if spec is None:
        raise ValueError(
            "Could not infer a comparison axis from this model. "
            "Provide a model with branch position, position, speed, trial progress, "
            "choice, or trial type terms."
        )

    fit_table = fit_df.copy()
    if "unit" in fit_table.columns:
        fit_table = fit_table.set_index("unit", drop=False)
    elif fit_table.index.name != "unit":
        fit_table.index = pd.Index(fit_table.index, name="unit")

    unit_ids_arr = np.asarray(unit_ids)
    shared_design_template = None
    if per_unit_transform is None:
        template_df = cov_df.copy()
        template_df["spike_count"] = np.zeros(len(template_df), dtype=float)
        shared_design_template = _build_design_template(formula, template_df)

    figs = []
    unit_chunks = list(_chunk_unit_list(unit_list, max_rows_per_fig))
    n_chunks = len(unit_chunks)

    for chunk_idx, unit_chunk in enumerate(unit_chunks, start=1):
        n_rows = len(unit_chunk)

        if spec["kind"] == "linearized_branch":
            height_ratios = [4] * n_rows + [1]
            fig, axes = plt.subplots(
                n_rows + 1, 1,
                figsize=(11, max(figsize_per_row * n_rows + 1.0, 3.5)),
                sharex=True,
                gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
            )
            axes = np.atleast_1d(axes)
            plot_axes = axes[:-1]
            ax_track = axes[-1]
        else:
            fig, axes = plt.subplots(
                n_rows, 1,
                figsize=(10, max(figsize_per_row * n_rows, 2.8)),
                sharex=(spec["kind"] != "categorical"),
            )
            plot_axes = np.atleast_1d(axes)
            ax_track = None

        for row_idx, (uid, ax) in enumerate(zip(unit_chunk, plot_axes)):
            matches = np.where(unit_ids_arr == uid)[0]
            if len(matches) == 0:
                ax.text(
                    0.5, 0.5, f"unit {uid}\nnot found",
                    ha="center", va="center", transform=ax.transAxes, fontsize=9,
                )
                ax.set_axis_off()
                continue

            unit_index = int(matches[0])
            unit_df = _build_unit_fit_frame(
                cov_df, spike_counts_masked, unit_index,
                per_unit_transform=per_unit_transform,
            )

            try:
                if uid not in fit_table.index:
                    raise KeyError(f"unit {uid} missing from fit_df")

                fit_row = fit_table.loc[uid]
                if isinstance(fit_row, pd.DataFrame):
                    fit_row = fit_row.iloc[0]
                if ("converged" in fit_row.index) and (not bool(fit_row["converged"])):
                    raise RuntimeError("stored fit did not converge")

                ctx = _build_prediction_context(
                    formula, unit_df, fit_row["coef"],
                    design_template=shared_design_template,
                )
                _plot_predicted_vs_actual_row(
                    ctx, ctx["cov_df_fit"], spec, ax=ax, bin_size=bin_size,
                    n_bins=n_bins, n_sweep=n_sweep,
                    points_per_segment=points_per_segment,
                    show_legend=(row_idx == 0),
                )
                ax.set_title(f"unit {uid}", loc="left", fontsize=9)
            except Exception as e:
                ax.text(
                    0.5, 0.5, f"unit {uid}\nfit failed\n{e}",
                    ha="center", va="center", transform=ax.transAxes, fontsize=8,
                )
                continue

            if row_idx < n_rows - 1 and spec["kind"] != "categorical":
                ax.tick_params(labelbottom=False)

        if ax_track is not None:
            axis_info = _get_linearized_branch_axis(points_per_segment=points_per_segment)
            _draw_repeated_stem_track(ax_track, axis_info)
            ax_track.set_xlim(plot_axes[-1].get_xlim())
            ax_track.set_xlabel(spec["xlabel"])
        else:
            plot_axes[-1].set_xlabel(spec["xlabel"])

        if title:
            fig_title = title
            if n_chunks > 1:
                fig_title = f"{title} ({chunk_idx}/{n_chunks})"
            fig.suptitle(fig_title, fontsize=12)
            plt.tight_layout(rect=(0, 0, 1, 0.98))
        else:
            plt.tight_layout()

        figs.append(fig)

    return figs[0] if len(figs) == 1 else figs


def plot_marginal_tuning(results, term_pattern, sweep_col, actual_col,
                         cov_df, unit_idx=None, min_val=None, max_val=None,
                         xlabel=None, categorical=False, bin_size=None,
                         n_bins=50, n_sweep=300, ax=None, title=None,
                         wtrack=False, pos_run=None):
    """Plot marginal tuning curve vs empirical rate for one term of a fitted GLM.

    Parameters
    ----------
    results : statsmodels GLMResults
    term_pattern : str
        Substring to identify design-matrix columns (e.g. ``"pos_scaled"``).
    sweep_col : str
        Column name that receives sweep values.
    actual_col : str
        Column in *cov_df* with the raw behavioural variable.
    cov_df : DataFrame
        Must contain ``"spike_count"`` and all formula columns.
    unit_idx : int, optional
        Used only for the title.
    min_val, max_val : float, optional
        Natural-unit range.  When provided, sweep values are linearly spaced
        in natural units and mapped to [0, 1] for scaled columns.
    xlabel : str, optional
    categorical : bool
    bin_size, n_bins, n_sweep : float, int, int
    ax : matplotlib Axes, optional
    title : str, optional
    wtrack : bool
        If True, render a two-panel W-track heatmap (empirical vs marginal)
        instead of a line plot.  Requires *pos_run* or a prior ``set_plot_state()``.
    pos_run : DataFrame, optional
        Track projection data for W-track plots.
    """
    if bin_size is None:
        bin_size = CONFIG["bin_size"]

 # sweep values
    if categorical:
        sweep_values = sorted(cov_df[actual_col].dropna().unique())
    elif min_val is not None and max_val is not None:
        natural = np.linspace(min_val, max_val, n_sweep)
        sweep_values = (natural - min_val) / (max_val - min_val)
    else:
        sweep_values = np.linspace(0, 1, n_sweep)
        natural = sweep_values

 # marginal rate
    marginal_hz = compute_marginal_effect(
        results, term_pattern, sweep_values, sweep_col, cov_df, bin_size,
    )

 # empirical rate
    emp_x, emp_hz = empirical_tuning_curve(
        cov_df, actual_col, bin_size=bin_size, n_bins=n_bins,
        categorical=categorical,
    )

 # W-track heatmap mode
    if wtrack:
        if pos_run is None:
            pos_run = _plot_state.get("pos_run")
        if pos_run is None:
            raise ValueError("pos_run required for wtrack plot")

        if min_val is not None:
            natural = np.linspace(min_val, max_val, n_sweep)
        pred_rate_run = np.interp(pos_run["linear_position"], natural, marginal_hz)
        actual_rate_run = np.interp(pos_run["linear_position"], emp_x, emp_hz)
        vmax = np.nanpercentile(actual_rate_run, 99)

        import spyglass.linearization.v1 as sgpl
        graph = sgpl.TrackGraph & {"track_graph_name": CONFIG["wtrack_name"]}
        unit_label = f"unit {unit_idx}" if unit_idx is not None else ""

        fig, (ax_emp, ax_mod) = plt.subplots(1, 2, figsize=(14, 5))
        for _ax, rate, _title in [
            (ax_emp, actual_rate_run, f"{unit_label} — empirical"),
            (ax_mod, pred_rate_run,   f"{unit_label} — marginal ({title or term_pattern})"),
        ]:
            graph.plot_track_graph(ax=_ax, draw_edge_labels=False)
            for ln in _ax.lines:
                ln.set_color("lightgrey")
            sc = _ax.scatter(
                pos_run["projected_x_position"], pos_run["projected_y_position"],
                c=rate, cmap="hot_r", s=3, zorder=3, vmin=0, vmax=vmax,
            )
            plt.colorbar(sc, ax=_ax, label="Hz", shrink=0.8)
            _ax.set_title(_title)
            _ax.set_xlabel("x (cm)"); _ax.set_ylabel("y (cm)")
        plt.tight_layout()
        return

 # standard line / bar plot
    standalone = ax is None
    if standalone:
        _, ax = plt.subplots(figsize=(8, 4))

    unit_label = f"unit {unit_idx}, " if unit_idx is not None else ""
    _title = title or f"{unit_label}{actual_col}"

    if categorical:
        x = np.arange(len(sweep_values))
        w = 0.35
        ax.bar(x - w / 2, emp_hz,      w, color="grey",   alpha=0.8, label="Empirical")
        ax.bar(x + w / 2, marginal_hz,  w, color="tomato", alpha=0.8, label="Marginal")
        ax.set_xticks(x)
        ax.set_xticklabels(sweep_values)
    else:
        if min_val is not None:
            natural = np.linspace(min_val, max_val, n_sweep)
        ax.plot(emp_x, emp_hz, color="grey", alpha=0.7, lw=1.2, label="Empirical")
        ax.plot(natural, marginal_hz, color="tomato", lw=2, label="Marginal")

    ax.set_xlabel(xlabel or actual_col)
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title(_title)
    ax.legend(fontsize=8)
    sns.despine(ax=ax)

    if standalone:
        plt.tight_layout()


def plot_partial_dependence_grid(results_combined, var_specs, cov_df,
                                 bin_size=0.002, n_bins=50, title=""):
    """N_vars × 2 grid: left = single-variable (uncontrolled), right = partial dependence (controlled).

    Each dict in var_specs:
        sweep_col      : column swept in prediction DataFrame (e.g. "pos_scaled")
        actual_col     : column in cov_df for occupancy-normalized empirical rate
        xlabel         : x-axis label
        results_single : fitted single-variable GLM
        fixed          : {col: val} — other covariates held constant
        min_val, max_val : float or None — rescale sweep from [0,1] to natural units
        categorical    : bool (default False)
    """
    n = len(var_specs)
    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n))
    axes = np.array(axes).reshape(n, 2)
    if title:
        fig.suptitle(title, fontsize=12, y=1.01)

    for row, spec in enumerate(var_specs):
        ax_unc, ax_ctl = axes[row]
        sweep_col   = spec["sweep_col"]
        actual_col  = spec["actual_col"]
        xlabel      = spec["xlabel"]
        r_single    = spec["results_single"]
        fixed       = spec["fixed"]
        min_val     = spec.get("min_val")
        max_val     = spec.get("max_val")
        categorical = spec.get("categorical", False)

        if categorical:
            cats = sorted(cov_df[actual_col].dropna().unique())
            x = np.arange(len(cats))
            width = 0.35

            grouped = cov_df.groupby(actual_col)
            emp_hz = (grouped["spike_count"].sum() / (grouped.size() * bin_size)).reindex(cats).values

            pred_single_hz  = r_single.predict(pd.DataFrame({sweep_col: cats})).values / bin_size
            syn_comb = pd.DataFrame({sweep_col: cats,
                                     **{k: [v] * len(cats) for k, v in fixed.items()}})
            pred_comb_hz = results_combined.predict(syn_comb).values / bin_size

            for ax, pred_hz, clr, lbl in [
                (ax_unc, pred_single_hz, "steelblue", "Single model"),
                (ax_ctl, pred_comb_hz,   "tomato",    "Combined (controlled)"),
            ]:
                ax.bar(x - width / 2, emp_hz,   width, color="grey",  alpha=0.8, label="Empirical")
                ax.bar(x + width / 2, pred_hz,  width, color=clr,     alpha=0.8, label=lbl)
                ax.set_xticks(x)
                ax.set_xticklabels(cats)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Firing rate (Hz)")
                ax.legend(fontsize=8)
                sns.despine(ax=ax)
        else:
            valid = cov_df[actual_col].notna()
            actual_vals = cov_df.loc[valid, actual_col].values
            spike_vals  = cov_df.loc[valid, "spike_count"].values
            occ,  bins = np.histogram(actual_vals, bins=n_bins)
            spks, _    = np.histogram(actual_vals, bins=bins, weights=spike_vals)
            with np.errstate(invalid="ignore", divide="ignore"):
                emp_rate = np.where(occ > 0, spks / (occ * bin_size), np.nan)
            bin_centers = (bins[:-1] + bins[1:]) / 2

            sweep = np.linspace(0, 1, 200)
            x_natural = sweep * (max_val - min_val) + min_val if min_val is not None else sweep

            pred_single_hz = r_single.predict(pd.DataFrame({sweep_col: sweep})).values / bin_size
            syn_comb = pd.DataFrame({sweep_col: sweep,
                                     **{k: [v] * 200 for k, v in fixed.items()}})
            pred_comb_hz = results_combined.predict(syn_comb).values / bin_size

            for ax, pred_hz, clr, lbl in [
                (ax_unc, pred_single_hz, "steelblue", "Single model"),
                (ax_ctl, pred_comb_hz,   "tomato",    "Combined (controlled)"),
            ]:
                ax.plot(bin_centers, emp_rate, color="grey", alpha=0.7, lw=1.2, label="Empirical")
                ax.plot(x_natural, pred_hz, color=clr, lw=2, label=lbl)
                ax.set_xlabel(xlabel)
                ax.set_ylabel("Firing rate (Hz)")
                ax.legend(fontsize=8)
                sns.despine(ax=ax)

        ax_unc.set_title(f"{xlabel} — uncontrolled")
        ax_ctl.set_title(f"{xlabel} — partial dependence (controlled)")

    plt.tight_layout()


def plot_wtrack_comparison(uid, cov_df, n_bins=50, results=None, pos_run=None,
                           bin_size=None):
    """Two-panel W-track heatmap: actual (left) vs GLM-predicted (right) firing rate.

    When called before set_plot_state(), pass results= (fitted pos_spline model) and
    pos_run= (DataFrame with linear_position, projected_x/y_position columns).
    """
    if bin_size is None:
        bin_size = CONFIG["bin_size"]

    # Resolve pos_run
    if pos_run is None:
        pos_run = _plot_state["pos_run"]
    if pos_run is None:
        raise ValueError("pos_run must be provided via pos_run= or set_plot_state()")

    # Resolve predicted rate
    if _plot_state["rate_matrix"] is not None and results is None:
        pred_rate_run = np.interp(
            pos_run["linear_position"], _plot_state["pos_pred_vals"], _plot_state["rate_matrix"][uid]
        )
    elif results is not None:
        pos_min = cov_df["linear_position"].min()
        pos_max = cov_df["linear_position"].max()
        pos_pred_vals_local = np.linspace(pos_min, pos_max, 300)
        pos_pred_scaled = (pos_pred_vals_local - pos_min) / (pos_max - pos_min)
        pred_hz = results.predict(pd.DataFrame({"pos_scaled": pos_pred_scaled})) / bin_size
        pred_rate_run = np.interp(pos_run["linear_position"], pos_pred_vals_local, pred_hz)
    else:
        raise ValueError("Either call set_plot_state() first, or pass results=")

    # Predicted rate interpolated onto pos_run positions — resolved above

    # Empirical occupancy-normalized rate
    valid = cov_df["linear_position"].notna()
    actual_pos = cov_df.loc[valid, "linear_position"].values
    spike_vals = cov_df.loc[valid, "spike_count"].values

    occ,  bins = np.histogram(actual_pos, bins=n_bins)
    spks, _    = np.histogram(actual_pos, bins=bins, weights=spike_vals)
    with np.errstate(invalid="ignore", divide="ignore"):
        emp_rate_bins = np.where(occ > 0, spks / (occ * bin_size), 0.0)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Interpolate empirical onto pos_run positions
    actual_rate_run = np.interp(pos_run["linear_position"], bin_centers, emp_rate_bins)

    vmax = np.nanpercentile(actual_rate_run, 99)
    import spyglass.linearization.v1 as sgpl
    graph = sgpl.TrackGraph & {"track_graph_name": CONFIG["wtrack_name"]}

    fig, (ax_actual, ax_pred) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, rate_run, title in [
        (ax_actual, actual_rate_run, f"unit {uid} — actual"),
        (ax_pred,   pred_rate_run,   f"unit {uid} — GLM predicted"),
    ]:
        graph.plot_track_graph(ax=ax, draw_edge_labels=False)
        for ln in ax.lines:
            ln.set_color("lightgrey")
        sc = ax.scatter(pos_run["projected_x_position"], pos_run["projected_y_position"],
                        c=rate_run, cmap="hot_r", s=3, zorder=3, vmin=0, vmax=vmax)
        plt.colorbar(sc, ax=ax, label="Hz", shrink=0.8)
        ax.set_title(title)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")

    plt.tight_layout()


def plot_wtrack_branch_comparison(results, cov_df, pos_run=None, n_bins=80,
                                   bin_size=None, title=None):
    """Two-panel W-track heatmap for a branch position model: empirical vs predicted.

    Unlike plot_wtrack_comparison (which uses a 1-D pos_scaled sweep), this
    function evaluates the fitted branch model at every observed time bin so
    that the branch_id categorical covariate is handled correctly.  Predictions
    for the stem are averaged over both traversal directions.

    Parameters
    ----------
    results : statsmodels GLMResults
        Fitted model that includes branch_id and/or branch_pos_scaled terms.
    cov_df : DataFrame
        Covariate DataFrame used for fitting (must have branch_id, branch_pos_cm,
        linear_position, spike_count).
    pos_run : DataFrame, optional
        Position DataFrame with projected_x_position, projected_y_position, and
        linear_position at video frame rate.  Falls back to _plot_state["pos_run"].
    n_bins : int
        Number of linear_position bins for binning empirical and predicted rates.
    bin_size : float, optional
        Time-bin width in seconds.  Defaults to CONFIG["bin_size"].
    title : str, optional
        Prefix for subplot titles.
    """
    import spyglass.linearization.v1 as sgpl

    if bin_size is None:
        bin_size = CONFIG["bin_size"]

    if pos_run is None:
        pos_run = _plot_state.get("pos_run")
    if pos_run is None:
        raise ValueError("pos_run must be provided via pos_run= or set_plot_state()")

    valid = (
        cov_df["linear_position"].notna()
        & cov_df["spike_count"].notna()
    )
    cov_valid = cov_df.loc[valid].copy()

    pos_vals = cov_valid["linear_position"].to_numpy(dtype=float)
    spike_vals = cov_valid["spike_count"].to_numpy(dtype=float)

    pos_min = float(pos_vals.min())
    pos_max = float(pos_vals.max())
    edges = np.linspace(pos_min, pos_max, n_bins + 1)

    occ,  _ = np.histogram(pos_vals, bins=edges)
    spks, _ = np.histogram(pos_vals, bins=edges, weights=spike_vals)
    with np.errstate(invalid="ignore", divide="ignore"):
        emp_rate_bins = np.where(occ > 0, spks / (occ.astype(float) * bin_size), np.nan)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    pred_counts = results.predict(cov_valid).to_numpy(dtype=float)
    pred_hz_vals = pred_counts / bin_size
    pred_sum, _ = np.histogram(pos_vals, bins=edges, weights=pred_hz_vals)
    with np.errstate(invalid="ignore", divide="ignore"):
        pred_rate_bins = np.where(occ > 0, pred_sum / occ.astype(float), np.nan)

    run_pos = pos_run["linear_position"].to_numpy(dtype=float)
    finite_emp  = np.isfinite(emp_rate_bins)
    finite_pred = np.isfinite(pred_rate_bins)
    emp_rate_run  = np.interp(run_pos, bin_centers[finite_emp],  emp_rate_bins[finite_emp],
                              left=np.nan, right=np.nan)
    pred_rate_run = np.interp(run_pos, bin_centers[finite_pred], pred_rate_bins[finite_pred],
                              left=np.nan, right=np.nan)

    vmax = float(np.nanpercentile(emp_rate_run[np.isfinite(emp_rate_run)], 99))

    label = title or ""
    graph = sgpl.TrackGraph & {"track_graph_name": CONFIG["wtrack_name"]}

    fig, (ax_emp, ax_pred) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, rate_run, panel_title in [
        (ax_emp,  emp_rate_run,  f"{label} — empirical".lstrip(" —")),
        (ax_pred, pred_rate_run, f"{label} — GLM predicted".lstrip(" —")),
    ]:
        graph.plot_track_graph(ax=ax, draw_edge_labels=False)
        for ln in ax.lines:
            ln.set_color("lightgrey")
        sc = ax.scatter(
            pos_run["projected_x_position"], pos_run["projected_y_position"],
            c=rate_run, cmap="hot_r", s=3, zorder=3, vmin=0, vmax=vmax,
        )
        plt.colorbar(sc, ax=ax, label="Hz", shrink=0.8)
        ax.set_title(panel_title)
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")

    plt.tight_layout()


def plot_place_field_grid(unit_list, ncols=4):
    """Track heatmaps for a list of units arranged in a grid."""
    rate_matrix   = _plot_state["rate_matrix"]
    pos_pred_vals = _plot_state["pos_pred_vals"]
    pos_run       = _plot_state["pos_run"]
    peak_pos_cm   = _plot_state["peak_pos_cm"]

    n = len(unit_list)
    nrows = int(np.ceil(n / ncols))
    import spyglass.linearization.v1 as sgpl
    graph = sgpl.TrackGraph & {"track_graph_name": "Wtrack_wilbur20210512"}

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5))
    axes = np.array(axes).flatten()

    for ax, uid in zip(axes, unit_list):
        rate = rate_matrix[uid]
        track_rate = np.interp(pos_run["linear_position"], pos_pred_vals, rate)
        graph.plot_track_graph(ax=ax, draw_edge_labels=False)
        for ln in ax.lines:
            ln.set_color("lightgrey")
        sc = ax.scatter(pos_run["projected_x_position"], pos_run["projected_y_position"],
                        c=track_rate, cmap="hot_r", s=2, zorder=3,
                        vmin=rate.min(), vmax=rate.max())
        plt.colorbar(sc, ax=ax, label="Hz", shrink=0.7)
        ax.set_title(f"unit {uid}  peak={peak_pos_cm[uid]:.0f}cm", fontsize=9)
        ax.set_xlabel(""); ax.set_ylabel("")

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout()
