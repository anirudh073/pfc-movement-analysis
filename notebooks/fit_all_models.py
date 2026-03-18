"""
Fit all single-variable GLM encoding models and save results to CSV.
Run from project root:
    python notebooks/fit_all_models.py

Fits run sequentially; progress is printed with timing.
The notebook can load finished CSVs at any time while this script is running.
"""

import os, time, warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import bs, cr

os.chdir("/media/labuser/NA_1_2025/spyglass/wilbur")

base_dir = "/media/labuser/NA_1_2025/spyglass/wilbur"
BIN_SIZE = 0.002
tp_df = 6


# ── load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
trialized_position = pd.read_csv(
    f"{base_dir}/analysis/position/trialized_position.csv", index_col="time"
)
data = np.load(f"{base_dir}/analysis/final_spikes/mfpc_spikes.npz", allow_pickle=True)
mpfc_spikes = [data[f"arr_{i}"] for i in range(len(data.files))]

bin_edges = np.arange(
    trialized_position.index.min(), trialized_position.index.max() + BIN_SIZE, BIN_SIZE
)
bin_centers = bin_edges[:-1] + BIN_SIZE / 2
spike_counts = np.array([np.histogram(s, bins=bin_edges)[0] for s in mpfc_spikes])
print(f"  {len(mpfc_spikes)} units, {spike_counts.shape[1]:,} total bins")

# ── interpolate covariates to 2ms bins ───────────────────────────────────────
def interp_col(col_values, times, bin_centers):
    if pd.api.types.is_numeric_dtype(col_values):
        vals = col_values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() < 2:
            return np.full(len(bin_centers), np.nan)
        # Exclude NaN anchor points — np.interp propagates NaN from any bracketing point
        return np.interp(bin_centers, times[valid], vals[valid], left=np.nan, right=np.nan)
    else:
        idx = (np.searchsorted(times, bin_centers, side='right') - 1).clip(0, len(times) - 1)
        return col_values.iloc[idx].values

cols_to_interp = [c for c in trialized_position.columns if c != "video_frame_ind"]
times = trialized_position.index.astype(float).values
interpolated = {col: interp_col(trialized_position[col], times, bin_centers)
                for col in cols_to_interp}
interp_pos = pd.DataFrame(interpolated, columns=cols_to_interp)
interp_pos.insert(0, "time_bin_center", bin_centers)

# ── base mask: run zone, outbound + inbound ───────────────────────────────────
base_mask = (
    (interp_pos["zone"] == "run")
    & (interp_pos["trial_type"].isin(["outbound", "inbound"]))
)
cov_df = interp_pos[base_mask].rename(columns={"left/right": "choice"})
spike_counts_masked = spike_counts[:, base_mask]
unit_ids = np.arange(len(spike_counts_masked))

# ── common mask: speed + position filters ────────────────────────────────────
common_mask = (
    cov_df["speed"].notna() & (cov_df["speed"] > 5) & (cov_df["speed"] < 120)
    & cov_df["linear_position"].notna()
)
cov_df_common = cov_df[common_mask].copy()
spike_counts_common = spike_counts_masked[:, common_mask]

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

outbound_common_mask = common_mask & (cov_df["trial_type"] == "outbound")
cov_df_out_common = cov_df[outbound_common_mask].copy()
spike_counts_out_common = spike_counts_masked[:, outbound_common_mask]
cov_df_out_common["speed_scaled"] = (
    (cov_df_out_common["speed"] - speed_min_val) / (speed_max_val - speed_min_val)
)
cov_df_out_common["pos_scaled"] = (
    (cov_df_out_common["linear_position"] - pos_min_val) / (pos_max_val - pos_min_val)
)

print(f"  cov_df_common:     {len(cov_df_common):,} bins")
print(f"  cov_df_out_common: {len(cov_df_out_common):,} bins (outbound only)")

# ── fit function ──────────────────────────────────────────────────────────────
def fit_glm_all_units(formula, cov_df, spike_counts_masked, unit_ids, bin_size=0.002):
    rows = []
    for i, uid in enumerate(unit_ids):
        df = cov_df.copy()
        df["spike_count"] = spike_counts_masked[i]
        try:
            res = smf.glm(formula, data=df, family=sm.families.Poisson()).fit(disp=False)
            rows.append(dict(
                unit=uid, aic=res.aic, llf=res.llf, deviance=res.deviance,
                n_params=len(res.params), n_obs=int(res.nobs), converged=res.converged,
                coef=res.params.to_dict(), bse=res.bse.to_dict(),
                deviance_null=res.null_deviance, df_model=res.df_model
            ))
        except Exception as e:
            rows.append(dict(
                unit=uid, aic=np.nan, llf=np.nan, deviance=np.nan,
                n_params=np.nan, n_obs=np.nan, converged=False,
                coef=None, bse=None, deviance_null=np.nan, df_model=np.nan, error=str(e)
            ))
    return pd.DataFrame(rows)

def fit_and_save(name, formula, df, sc, path, keep_null=False):
    print(f"\nFitting {name} ({len(df):,} bins × {len(unit_ids)} units)...")
    t0 = time.time()
    result = fit_glm_all_units(formula, df, sc, unit_ids)
    result["model"] = name
    result.to_csv(path)
    elapsed = time.time() - t0
    n_converged = result["converged"].sum() if "converged" in result else "?"
    print(f"  Done in {elapsed:.0f}s — {n_converged}/{len(unit_ids)} converged → {path}")

# ── fit all models ─────────────────────────────────────────────────────────────
# fit_and_save(
#     "null", "spike_count ~ 1",
#     cov_df_common, spike_counts_common,
#     f"{base_dir}/analysis/null_model_all.csv"
# )

# fit_and_save(
#     "null_out", "spike_count ~ 1",
#     cov_df_out_common, spike_counts_out_common,
#     f"{base_dir}/analysis/null_model_out_all.csv"
# )

# fit_and_save(
#     "trial_type", "spike_count ~ trial_type",
#     cov_df_common, spike_counts_common,
#     f"{base_dir}/analysis/trial_type_model_all.csv"
# )

# fit_and_save(
#     "choice", "spike_count ~ choice",
#     cov_df_out_common, spike_counts_out_common,
#     f"{base_dir}/analysis/choice_model_all.csv"
# )


fit_and_save(
    "speed_spline", f"spike_count ~ bs(speed_scaled, df=4)",
    cov_df_common, spike_counts_common,
    f"{base_dir}/analysis/speed_spline_model_all.csv"
)

fit_and_save(
    "pos_spline", "spike_count ~ bs(pos_scaled, df=8)",
    cov_df_common, spike_counts_common,
    f"{base_dir}/analysis/pos_spline_model_all.csv"
)

# fit_and_save(
#     "trial_progress_spline", f"spike_count ~ cr(trial_progress, df={tp_df}, constraints='center')",
#     cov_df_common, spike_counts_common,
#     f"{base_dir}/analysis/trial_progress_spline_model_all.csv"
# )

# # ── multi-variable models ──────────────────────────────────────────────────────
fit_and_save(
    "full_model",
    "spike_count ~ trial_type + bs(pos_scaled, df=8) + bs(speed_scaled, df=4)",
    cov_df_common, spike_counts_common,
    f"{base_dir}/analysis/full_model_all.csv"
)

fit_and_save(
    "temporal_model",
    f"spike_count ~ trial_type + cr(trial_progress, df={tp_df}, constraints='center') + bs(speed_scaled, df=4)",
    cov_df_common, spike_counts_common,
    f"{base_dir}/analysis/temporal_model_all.csv"
)

# fit_and_save(
#     "choice_full_model",
#     "spike_count ~ choice + bs(pos_scaled, df=8) + bs(speed_scaled, df=4)",
#     cov_df_out_common, spike_counts_out_common,
#     f"{base_dir}/analysis/choice_full_model_all.csv"
# )

# fit_and_save(
#     "choice_temporal_model",
#     f"spike_count ~ choice + cr(trial_progress, df={tp_df}, constraints='center') + bs(speed_scaled, df=4)",
#     cov_df_out_common, spike_counts_out_common,
#     f"{base_dir}/analysis/choice_temporal_model_all.csv"
# )

print("\nAll models done.")


# ── diagnostics ───────────────────────────────────────────────────────────────
# Fits GLM once per unit and saves both scalar diagnostics (CSV) and residual
# profiles (NPZ) in a single pass. Skips models where both files already exist.
from encoding_utils import compute_model_diagnostics

COVARIATE_COLS   = ["linear_position", "speed", "trial_progress"]
CATEGORICAL_COLS = ["trial_type", "choice"]

def diag_and_save(model_name, formula, df, sc):
    print(f"\nDiagnostics+profiles: {model_name} ({len(df):,} bins × {len(unit_ids)} units)...")
    t0 = time.time()
    result, _ = compute_model_diagnostics(
        formula             = formula,
        cov_df              = df,
        spike_counts_masked = sc,
        unit_ids            = unit_ids,
        covariate_cols      = COVARIATE_COLS,
        categorical_cols    = CATEGORICAL_COLS,
        base_dir            = base_dir,
        model_name          = model_name,
    )
    elapsed = time.time() - t0
    n_ok = result["converged"].sum() if "converged" in result.columns else "?"
    print(f"  Done in {elapsed:.0f}s — {n_ok}/{len(unit_ids)} converged")

# diag_and_save(
#     "null_model_all",
#     "spike_count ~ 1",
#     cov_df_common, spike_counts_common,
# )
# diag_and_save(
#     "null_model_out_all",
#     "spike_count ~ 1",
#     cov_df_out_common, spike_counts_out_common,
# )
# diag_and_save(
#     "trial_type_model_all",
#     "spike_count ~ trial_type",
#     cov_df_common, spike_counts_common,
# )
# diag_and_save(
#     "choice_model_all",
#     "spike_count ~ choice",
#     cov_df_out_common, spike_counts_out_common,
# )
diag_and_save(
    "speed_spline_model_all",
    "spike_count ~ bs(speed_scaled, df=4)",
    cov_df_common, spike_counts_common,
)
diag_and_save(
    "pos_spline_model_all",
    "spike_count ~ bs(pos_scaled, df=8)",
    cov_df_common, spike_counts_common,
)
# diag_and_save(
#     "trial_progress_spline_model_all",
#     f"spike_count ~ cr(trial_progress, df={tp_df}, constraints='center')",
#     cov_df_common, spike_counts_common,
# )
diag_and_save(
    "full_model_all",
    "spike_count ~ trial_type + bs(pos_scaled, df=8) + bs(speed_scaled, df=4)",
    cov_df_common, spike_counts_common,
)
diag_and_save(
    "temporal_model_all",
    f"spike_count ~ trial_type + cr(trial_progress, df={tp_df}, constraints='center') + bs(speed_scaled, df=4)",
    cov_df_common, spike_counts_common,
)
# diag_and_save(
#     "choice_full_model_all",
#     "spike_count ~ choice + bs(pos_scaled, df=8) + bs(speed_scaled, df=4)",
#     cov_df_out_common, spike_counts_out_common,
# )
# diag_and_save(
#     "choice_temporal_model_all",
#     f"spike_count ~ choice + cr(trial_progress, df={tp_df}, constraints='center') + bs(speed_scaled, df=4)",
#     cov_df_out_common, spike_counts_out_common,
# )

print("\nAll diagnostics done.")
