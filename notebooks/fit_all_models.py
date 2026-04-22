"""
Author: Claude (Anthropic) — generated for the encoding project

Fit all GLM encoding models and save results to CSV.
Run from project root:
    python notebooks/fit_all_models.py

Fits run sequentially; progress is printed with timing.
The notebook can load finished CSVs at any time while this script is running.
"""

import argparse
import os, time, warnings
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
from patsy import bs, cr

from encoding_utils import (
    CONFIG, build_model_registry, load_and_prepare_data,
    fit_glm_all_units, compute_model_diagnostics, model_csv_path,
    make_history_formula, make_history_transform,
)

os.chdir(CONFIG["base_dir"])
base_dir = CONFIG["base_dir"]


def parse_args():
    parser = argparse.ArgumentParser(description="Fit GLM encoding models.")
    parser.add_argument(
        "--fit-history",
        action="store_true",
        help="Augment every selected formula with history terms and use add_spike_history.",
    )
    return parser.parse_args()


ARGS = parse_args()
FIT_HISTORY = bool(ARGS.fit_history)


def _run_name(name):
    return f"{name}_history" if FIT_HISTORY else name


HISTORY_TRANSFORM = make_history_transform() if FIT_HISTORY else None

# ── load & prepare data ──────────────────────────────────────────────────────
print("Loading data...")
data = load_and_prepare_data()
cov_df_common        = data["cov_df_common"]
cov_df_out_common    = data["cov_df_out_common"]
spike_counts_common  = data["spike_counts_common"]
spike_counts_out_common = data["spike_counts_out_common"]
unit_ids             = data["unit_ids"]

print(f"  cov_df_common:     {len(cov_df_common):,} bins")
print(f"  cov_df_out_common: {len(cov_df_out_common):,} bins (outbound only)")

# ── model registry ───────────────────────────────────────────────────────────
registry = build_model_registry()

datasets = {
    "common":   (cov_df_common, spike_counts_common),
    "outbound": (cov_df_out_common, spike_counts_out_common),
}

# ── fit helpers ──────────────────────────────────────────────────────────────

def fit_and_save(name, refit=False):
    """Fit model *name* from the registry and save to CSV.

    Skips if output CSV already exists unless *refit* is True.
    """
    entry = registry[name]
    formula = make_history_formula(entry["formula"]) if FIT_HISTORY else entry["formula"]
    ds_key  = entry["dataset"]
    cov_df, sc = datasets[ds_key]
    run_name = _run_name(name)

    print(
        f"\n{'Refitting' if refit else 'Fitting'} {run_name} "
        f"({len(cov_df):,} bins × {len(unit_ids)} units)..."
    )
    t0 = time.time()
    result = fit_glm_all_units(
        formula, cov_df, sc, unit_ids,
        model_name=run_name, refit=refit,
        per_unit_transform=HISTORY_TRANSFORM,
    )
    elapsed = time.time() - t0
    n_converged = result["converged"].sum() if "converged" in result else "?"
    print(f"  {n_converged}/{len(unit_ids)} converged ({elapsed:.1f}s)")


def diag_and_save(name):
    """Compute diagnostics for model *name* from the registry."""
    entry = registry[name]
    formula = make_history_formula(entry["formula"]) if FIT_HISTORY else entry["formula"]
    ds_key  = entry["dataset"]
    cov_df, sc = datasets[ds_key]
    run_name = _run_name(name)
    csv_stem = os.path.splitext(os.path.basename(model_csv_path(run_name)))[0]

    print(f"\nDiagnostics+profiles: {csv_stem} ({len(cov_df):,} bins × {len(unit_ids)} units)...")
    t0 = time.time()
    result, _ = compute_model_diagnostics(
        formula             = formula,
        cov_df              = cov_df,
        spike_counts_masked = sc,
        unit_ids            = unit_ids,
        covariate_cols      = ["linear_position", "speed", "trial_progress"],
        categorical_cols    = ["trial_type", "choice"],
        base_dir            = base_dir,
        model_name          = csv_stem,
        per_unit_transform  = HISTORY_TRANSFORM,
    )
    elapsed = time.time() - t0
    n_ok = result["converged"].sum() if "converged" in result.columns else "?"
    print(f"  Done in {elapsed:.0f}s — {n_ok}/{len(unit_ids)} converged")


# ── fit all models ───────────────────────────────────────────────────────────
# Uncomment models as needed.  fit_and_save() skips existing CSVs by default.

if FIT_HISTORY:
    print("History augmentation: ON")
else:
    print("History augmentation: OFF")

fit_and_save("null")
fit_and_save("null_outbound")
fit_and_save("trial_type")
fit_and_save("choice")
fit_and_save("speed_spline")
fit_and_save("pos_spline")
fit_and_save("branch_pos_spline")
fit_and_save("trial_progress_spline")
fit_and_save("full_model")
fit_and_save("temporal_model")
fit_and_save("choice_full_model")
fit_and_save("choice_temporal_model")

print("\nAll models done.")

# ── diagnostics ──────────────────────────────────────────────────────────────
diag_and_save("null")
diag_and_save("null_outbound")
diag_and_save("trial_type")
diag_and_save("choice")
diag_and_save("speed_spline")
diag_and_save("pos_spline")
diag_and_save("branch_pos_spline")
diag_and_save("trial_progress_spline")
diag_and_save("full_model")
diag_and_save("temporal_model")
diag_and_save("choice_full_model")
diag_and_save("choice_temporal_model")

print("\nAll diagnostics done.")
