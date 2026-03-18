"""
Utility functions for GLM encoding analysis (encoding.ipynb).

Sections:
  1. Data preparation  — interp_col
  2. GLM fitting       — fit_glm_all_units
  3. Drop-one analysis — make_drop_one_specs, build_reduced_formula,
                         fit_drop_one, run_drop_one_suite, compute_drop_one_lrt
  4. Diagnostics       — compute_residuals, plot_residuals,
                         compute_ks_rescaled, plot_ks, plot_diagnostics
  5. Visualization     — set_plot_state, plot_place_field, plot_place_field_grid
"""

import os, re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2, wilcoxon, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
import spyglass.linearization.v1 as sgpl
from statsmodels.stats.multitest import multipletests
import itertools
sns.set_context("talk")


# ── 1. Data preparation ───────────────────────────────────────────────────────

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


# ── 2. GLM fitting ────────────────────────────────────────────────────────────

def fit_glm_all_units(formula: str,
                      cov_df: pd.DataFrame,
                      spike_counts_masked: np.array,
                      unit_ids: np.array,
                      bin_size = 0.002):

    rows = []
    for i, uid in enumerate(unit_ids):
        df = cov_df.copy()
        df["spike_count"] = spike_counts_masked[i]  # pre-masked counts
        try:
            res = smf.glm(formula, data=df, family=sm.families.Poisson()).fit(disp=False)
            rows.append(dict(
                unit=uid,
                aic=res.aic,
                llf=res.llf,
                deviance=res.deviance,
                n_params=len(res.params),
                n_obs=int(res.nobs),
                converged=res.converged,
                coef=res.params.to_dict(),
                bse=res.bse.to_dict(),
                deviance_null = res.null_deviance,
                df_model = res.df_model
            ))

        except Exception as e:
            rows.append(dict(
                unit=uid, aic=np.nan, llf=np.nan, deviance=np.nan,
                n_params=np.nan, n_obs=np.nan, converged=False,
                coef=None, bse=None, deviance_null=np.nan,
                df_model=np.nan, error=str(e)
            ))

    return pd.DataFrame(rows)


# ── 3. Drop-one analysis ──────────────────────────────────────────────────────

def build_reduced_formula(spec: dict,
                         drop_term: str):
    included_terms = [term for term in spec["terms"] if term != drop_term]
    return spec["formula_lhs"] + "~" + " + ".join(included_terms)


def fit_drop_one(model_name: str,
                 spec: dict,
                 drop_term: str,
                 base_dir: str,
                 unit_ids: list):

    formula = build_reduced_formula(spec, drop_term)
    drop_term_safe = re.sub(r'[^\w]', '_', drop_term).strip('_')

    res = fit_glm_all_units(formula, spec["cov_df"], spec["spike_counts"], unit_ids)

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
                       check_if_exists: bool = True):

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
                     spec = spec,
                     drop_term=drop_term,
                     base_dir=base_dir,
                     unit_ids = unit_ids)
        result_list.append(result)

    return result_list


def make_drop_one_specs(cov_df_common, spike_counts_common,
                        cov_df_out_common, spike_counts_out_common,
                        base_dir, tp_df=6, pos_df=8, speed_df=4):
    tp   = f"cr(trial_progress, df = {tp_df}, constraints = 'center')"
    pos  = f"bs(pos_scaled, df = {pos_df})"
    spd  = f"bs(speed_scaled, df = {speed_df})"
    return {
        "temporal_model_all": {
            "formula_lhs": "spike_count",
            "terms": ["trial_type", tp, spd],
            "delta_df": {"trial_type": 1, tp: tp_df, spd: speed_df},
            "cov_df": cov_df_common,
            "spike_counts": spike_counts_common,
            "null_csv": f"{base_dir}/analysis/null_model_all.csv",
        },
        "full_model_all": {
            "formula_lhs": "spike_count",
            "terms": ["trial_type", pos, spd],
            "delta_df": {"trial_type": 1, pos: pos_df, spd: speed_df},
            "cov_df": cov_df_common,
            "spike_counts": spike_counts_common,
            "null_csv": f"{base_dir}/analysis/null_model_all.csv",
        },
        # "choice_full_model_all": {
        #     "formula_lhs": "spike_count",
        #     "terms": ["choice", pos, spd],
        #     "delta_df": {"choice": 1, pos: pos_df, spd: speed_df},
        #     "cov_df": cov_df_out_common,
        #     "spike_counts": spike_counts_out_common,
        #     "null_csv": f"{base_dir}/analysis/null_model_out_all.csv",
        # },
        # "choice_temporal_model_all": {
        #     "formula_lhs": "spike_count",
        #     "terms": ["choice", tp, spd],
        #     "delta_df": {"choice": 1, tp: tp_df, spd: speed_df},
        #     "cov_df": cov_df_out_common,
        #     "spike_counts": spike_counts_out_common,
        #     "null_csv": f"{base_dir}/analysis/null_model_out_all.csv",
        # },
    }


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
            partial_r2   = lrt_stat / deviance_null if deviance_null > 0 else np.nan

            rows.append(dict(
                unit         = uid,
                model        = model_name,
                dropped_term = term,
                lrt_stat     = lrt_stat,
                lrt_df       = lrt_df,
                lrt_pval     = lrt_pval,
                significant  = bool(lrt_pval < 0.05) if not np.isnan(lrt_pval) else False,
                delta_aic    = aic_full - aic_reduced,  # negative = full model better; kept for non-nested comparisons
                partial_r2   = partial_r2,               # N-invariant: lrt_stat / deviance_null
            ))

        if n_skipped:
            print(f"  [{model_name} drop={term}] skipped {n_skipped} units (non-converged)")

    return pd.DataFrame(rows)


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

    # A ── % units above partial R² threshold
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

    # B ── effect size violin
    ax = axes[1]
    sns.violinplot(data=df, x="partial_r2", y="term_label", order=term_order,
                   inner="box", cut=0, color="steelblue", ax=ax)
    ax.axvline(0, color="red", lw=1, ls="--", label="0")
    ax.set_xlabel("Partial R² (lrt_stat / null deviance)")
    ax.set_ylabel(""); ax.set_title("Effect size (partial R²)")
    ax.legend(fontsize=8); sns.despine(ax=ax)

    # C ── pairwise heatmap
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






# ── 4. Diagnostics — residuals & KS test ─────────────────────────────────────

def compute_residuals(results, spike_counts, cov_df):
    """
    Compute per-bin raw and cumulative residuals for a fitted Poisson GLM.

    raw_residual_i      = observed_i - predicted_i   (spike counts per bin)
    cumulative_residual = cumsum(raw_residual)        (integrated misfit)

    Returns a copy of cov_df with two extra columns:
      raw_residual        : observed minus E[N|X] per bin
      cumulative_residual : running cumulative sum of raw_residual over time

    Parameters
    ----------
    results      : fitted statsmodels GLM result object (res.predict() must work)
    spike_counts : 1-D array, shape (n_bins,), observed spike counts for one unit
    cov_df       : covariate DataFrame used to fit the model (n_bins rows)
    """
    predicted = np.asarray(results.predict())
    raw = np.asarray(spike_counts, dtype=float) - predicted

    df = cov_df.copy().reset_index(drop=True)
    df["raw_residual"]        = raw
    df["cumulative_residual"] = np.cumsum(raw)
    return df


def _load_trialized_position():
    """Load trialized_position.csv once and cache at module level."""
    if not hasattr(_load_trialized_position, "_cache"):
        base = os.environ.get("SPYGLASS_BASE_DIR", ".")
        path = os.path.join(base, "analysis", "position", "trialized_position.csv")
        _load_trialized_position._cache = pd.read_csv(path, index_col="time",
                                                       usecols=["time", "epoch"])
    return _load_trialized_position._cache


DEFAULT_PANELS = {
    "linear_position": "continuous",
    "speed":           "continuous",
    "trial_progress":  "continuous",
    "trial_type":      "categorical",
    "choice":          "categorical",
}


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

    # ── Optional: cumulative residual vs time (first epoch only) ─────────────
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
            bins       = np.linspace(cov_vals.min(), cov_vals.max(), n_bins + 1)
            idx        = np.clip(np.digitize(cov_vals, bins) - 1, 0, n_bins - 1)
            centers    = (bins[:-1] + bins[1:]) / 2
            mean_resid = np.array([res_vals[idx == b].mean() if (idx == b).any()
                                   else np.nan for b in range(n_bins)])

            # CUSUM sorted by covariate value
            sort_order = np.argsort(cov_vals)
            cusum      = np.cumsum(res_vals[sort_order])
            x_cusum    = cov_vals[sort_order]

            ax.plot(x_cusum, cusum, lw=0.8, color=CUSUM_COLOR, alpha=0.7)
            ax.axhline(0, color=CUSUM_COLOR, lw=0.8, ls="--")
            ax2 = ax.twinx()
            ax2.bar(centers, mean_resid, width=(bins[1] - bins[0]) * 0.9,
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


def compute_ks_rescaled(results, spike_counts):
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
    spike_counts : 1-D int array, shape (n_bins,), observed spike counts
    """
    predicted = np.asarray(results.predict())
    return np.sort(_compute_z_unsorted(predicted, spike_counts))


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

        # ── fit GLM ───────────────────────────────────────────────────────────
        df = cov_df.copy()
        df["spike_count"] = spike_counts_masked[i]
        try:
            res = smf.glm(formula, data=df, family=sm.families.Poisson()).fit(disp=False)
            if not res.converged:
                raise RuntimeError("did not converge")
        except Exception as e:
            for ax in axes[row_idx]:
                ax.text(0.5, 0.5, f"unit {uid}\nfit failed\n{e}",
                        ha="center", va="center", transform=ax.transAxes, fontsize=8)
            continue

        predicted = np.asarray(res.predict())
        raw       = spike_counts_masked[i].astype(float) - predicted

        col_idx = 0

        # ── Optional: cumulative residual vs time ─────────────────────────────
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

        # ── Covariate panels ──────────────────────────────────────────────────
        cont_ax_indices = []
        for col, kind in active.items():
            ax = axes[row_idx, col_idx]; col_idx += 1
            valid    = cov_df[col].notna().values
            res_vals = raw[valid]

            if kind == "continuous":
                cov_vals   = cov_df[col].values[valid]
                bins       = np.linspace(cov_vals.min(), cov_vals.max(), n_bins + 1)
                bin_idx    = np.clip(np.digitize(cov_vals, bins) - 1, 0, n_bins - 1)
                centers    = (bins[:-1] + bins[1:]) / 2
                mean_resid = np.array([res_vals[bin_idx == b].mean()
                                       if (bin_idx == b).any() else np.nan
                                       for b in range(n_bins)])
                ax.plot(centers, mean_resid, lw=1.2, color=RESID_COLOR)
                ax.axhline(0, color=RESID_COLOR, lw=0.8, ls="--")
                ax.set_ylabel("Mean resid.\n(spikes/bin)", fontsize=8, color=RESID_COLOR)
                cont_ax_indices.append(col_idx - 1)

            elif kind == "categorical":
                cat_vals = cov_df[col].values[valid]
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

        # ── KS panel ─────────────────────────────────────────────────────────
        ax = axes[row_idx, col_idx]
        z  = compute_ks_rescaled(res, spike_counts_masked[i])
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


def plot_diagnostics(results, spike_counts, cov_df,
                     n_bins=50, unit_label="", alpha=0.05,
                     panels=None, show_cumulative=False):
    """
    Full diagnostic dashboard for one unit's fitted Poisson GLM.

    Calls compute_residuals → plot_residuals (covariate panels),
    then compute_ks_rescaled → plot_ks in a separate figure.

    Parameters
    ----------
    results         : fitted statsmodels GLM result
    spike_counts    : 1-D array (n_bins,) for this unit
    cov_df          : covariate DataFrame (n_bins rows)
    n_bins          : bins for continuous covariate residual plots
    unit_label      : string identifier shown in plot titles
    alpha           : CI level for KS bands (default 0.05)
    panels          : dict {col: "continuous"|"categorical"}, optional.
        Forwarded to plot_residuals. Defaults to DEFAULT_PANELS.
    show_cumulative : bool — include the cumulative residual vs time panel.
    """
    residuals_df = compute_residuals(results, spike_counts, cov_df)
    plot_residuals(residuals_df, n_bins=n_bins,
                   title=f"Residuals — {unit_label}",
                   panels=panels, show_cumulative=show_cumulative)

    z_vals = compute_ks_rescaled(results, spike_counts)
    fig, ax = plt.subplots(figsize=(5, 5))
    plot_ks(z_vals, alpha=alpha, ax=ax, title=f"KS — {unit_label}",)
    plt.tight_layout()


def compute_model_diagnostics(formula, cov_df, spike_counts_masked, unit_ids,
                               covariate_cols, base_dir, model_name,
                               n_bins=50, unit_subset=None, categorical_cols=None):
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

    # ── pre-compute bin edges for profiles ────────────────────────────────────
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

        try:
            res = smf.glm(formula, data=df,
                          family=sm.families.Poisson()).fit(disp=False)
            if not res.converged:
                raise RuntimeError("not converged")
        except Exception:
            row = dict(unit=uid, converged=False,
                       ks_D=np.nan, ks_z_autocorr=np.nan, drift_auc=np.nan)
            for col in covariate_cols:
                row[f"resid_eta2_{col}"] = np.nan
            diag_rows.append(row)
            continue

        predicted = np.asarray(res.predict())
        raw       = spike_counts_masked[i].astype(float) - predicted
        n_spikes  = int((spike_counts_masked[i] > 0).sum())

        # ── drift AUC ─────────────────────────────────────────────────────────
        cumresid  = np.cumsum(raw)
        drift_auc = np.abs(cumresid).mean() / max(n_spikes, 1)

        # ── KS D + z autocorrelation ──────────────────────────────────────────
        z_unsorted = _compute_z_unsorted(predicted, spike_counts_masked[i])
        if len(z_unsorted) >= 4:
            z_sorted      = np.sort(z_unsorted)
            n             = len(z_sorted)
            ecdf          = np.arange(1, n + 1) / n
            ks_D          = float(np.max(np.abs(ecdf - z_sorted)))
            z_autocorr, _ = spearmanr(z_unsorted[:-1], z_unsorted[1:])
        else:
            ks_D, z_autocorr = np.nan, np.nan

        # ── residual η² per covariate ─────────────────────────────────────────
        row      = dict(unit=uid, converged=True,
                        ks_D=ks_D, ks_z_autocorr=float(z_autocorr), drift_auc=drift_auc)
        ss_total = float(np.sum(raw ** 2))
        for col in covariate_cols:
            valid_mask = ~np.isnan(cov_df[col].values)
            if valid_mask.sum() > 10 and ss_total > 0:
                cov_vals = cov_df[col].values[valid_mask]
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

        # ── residual profiles ─────────────────────────────────────────────────
        for col in covariate_cols:
            valid_mask = ~np.isnan(cov_df[col].values)
            cov_vals   = cov_df[col].values[valid_mask]
            res_vals   = raw[valid_mask]
            idx        = np.clip(np.digitize(cov_vals, bin_edges[col]) - 1, 0, n_bins - 1)
            means      = np.array([
                res_vals[idx == b].mean() if (idx == b).any() else np.nan
                for b in range(n_bins)
            ])
            profile_rows[col].append(means)
        # ── categorical profiles ──────────────────────────────────────────────
        for col in categorical_cols:
            if col not in cov_df.columns:
                continue
            cat_vals = cov_df[col].values
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
                                   unit_subset=None):
    """Thin wrapper — calls compute_model_diagnostics and returns only diag_df."""
    diag_df, _ = compute_model_diagnostics(
        formula, cov_df, spike_counts_masked, unit_ids,
        covariate_cols, base_dir, model_name,
        unit_subset=unit_subset,
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

    # ── Figure 1: timing & stationarity ──────────────────────────────────────
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

    # ── Figure 2: covariate residual structure ────────────────────────────────
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


def plot_residual_heterogeneity(profiles, panels=None, row_normalise=True):
    """
    Population-level residual heterogeneity: one heatmap per covariate.

    Parameters
    ----------
    profiles      : dict from compute_model_diagnostics.
    panels        : dict {col: "continuous"|"categorical"}, optional.
                    Defaults to all panels detected in profiles.
    row_normalise : bool (default True). If True, each row is divided by its
                    max absolute value so all neurons share the same ±1 scale
                    (useful for comparing structure across neurons). If False,
                    raw residuals in spikes/bin are shown with a shared colorscale
                    (useful for comparing magnitude across models).
    """
    if panels is None:
        panels = {}
        for k in profiles:
            if k.endswith("_cat_profiles"):
                panels[k.replace("_cat_profiles", "")] = "categorical"
            elif k.endswith("_profiles"):
                panels[k.replace("_profiles", "")] = "continuous"

    panels = {col: kind for col, kind in panels.items()
              if (kind == "continuous" and f"{col}_profiles" in profiles)
              or (kind == "categorical" and f"{col}_cat_profiles" in profiles)}

    if not panels:
        raise ValueError("No matching panels found in profiles dict.")

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 8))
    axes = np.atleast_1d(axes)

    for ax, (col, kind) in zip(axes, panels.items()):
        if kind == "continuous":
            mat     = np.array(profiles[f"{col}_profiles"])
            centers = np.array(profiles[f"{col}_centers"])
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
                extent=[centers[0], centers[-1], mat.shape[0], 0],
                interpolation="nearest",
            )
            ax.set_xlabel(col)

        elif kind == "categorical":
            mat    = np.array(profiles[f"{col}_cat_profiles"])
            labels = list(profiles[f"{col}_cat_labels"])
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
            ax.set_xlabel(col)

        cbar_label = "Normalised residual (a.u.)" if row_normalise else "Residual (spikes/bin)"
        title_suffix = "(row-normalised)" if row_normalise else "(raw)"
        plt.colorbar(im, ax=ax, label=cbar_label, shrink=0.8)
        ax.set_ylabel("Unit (sorted by peak)")
        ax.set_title(f"Residual profiles — {col}\n{title_suffix}")
        sns.despine(ax=ax)

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

    if covariate_cols is None:
        first = next(iter(profiles_dicts.values()))
        covariate_cols = [k.replace("_profiles", "")
                          for k in first if k.endswith("_profiles")]

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
            mat   = prof[f"{col}_profiles"]   # (n_units, n_bins)
            cents = prof[f"{col}_centers"]    # (n_bins,)

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


# ── 5. Visualization ──────────────────────────────────────────────────────────
# Shared state for plot functions — set via set_plot_state() after fitting

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

    # ── rate curve ────────────────────────────────────────────────────────────
    ax_curve.plot(pos_pred_vals, rate, color="steelblue", lw=1.5)
    ax_curve.axvline(peak_pos_cm[uid], color="red", lw=1, ls="--",
                     label=f"peak @ {peak_pos_cm[uid]:.0f} cm")
    ax_curve.set_xlabel("linear position (cm)")
    ax_curve.set_ylabel("firing rate (Hz)")
    ax_curve.set_title(f"unit {uid}")
    ax_curve.legend(fontsize=8)
    sns.despine(ax=ax_curve)

    # ── track heatmap ─────────────────────────────────────────────────────────
    track_rate = np.interp(pos_run["linear_position"], pos_pred_vals, rate)
    if graph is None:
        graph = sgpl.TrackGraph & {"track_graph_name": "Wtrack_wilbur20210512"}
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


def plot_wtrack_comparison(uid, cov_df, n_bins=50, results=None, pos_run=None):
    """Two-panel W-track heatmap: actual (left) vs GLM-predicted (right) firing rate.

    When called before set_plot_state(), pass results= (fitted pos_spline model) and
    pos_run= (DataFrame with linear_position, projected_x/y_position columns).
    """
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
        pred_hz = results.predict(pd.DataFrame({"pos_scaled": pos_pred_scaled})) / 0.002
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
        emp_rate_bins = np.where(occ > 0, spks / (occ * 0.002), 0.0)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Interpolate empirical onto pos_run positions
    actual_rate_run = np.interp(pos_run["linear_position"], bin_centers, emp_rate_bins)

    vmax = np.nanpercentile(actual_rate_run, 99)
    graph = sgpl.TrackGraph & {"track_graph_name": "Wtrack_wilbur20210512"}

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


def plot_place_field_grid(unit_list, ncols=4):
    """Track heatmaps for a list of units arranged in a grid."""
    rate_matrix   = _plot_state["rate_matrix"]
    pos_pred_vals = _plot_state["pos_pred_vals"]
    pos_run       = _plot_state["pos_run"]
    peak_pos_cm   = _plot_state["peak_pos_cm"]

    n = len(unit_list)
    nrows = int(np.ceil(n / ncols))
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
