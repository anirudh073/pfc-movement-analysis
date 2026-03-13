"""
Utility functions for GLM encoding analysis (encoding.ipynb).

Sections:
  1. Data preparation  — interp_col
  2. GLM fitting       — fit_glm_all_units
  3. Drop-one analysis — make_drop_one_specs, build_reduced_formula,
                         fit_drop_one, run_drop_one_suite, compute_drop_one_lrt
  4. Visualization     — set_plot_state, plot_place_field, plot_place_field_grid
"""

import os, re
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns
import spyglass.linearization.v1 as sgpl
from statsmodels.stats.multitest import multipletests
from scipy.stats import wilcoxon
import itertools



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
        idx = np.searchsorted(times, bin_centers).clip(0, len(times) - 1)
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
        "choice_full_model_all": {
            "formula_lhs": "spike_count",
            "terms": ["choice", pos, spd],
            "delta_df": {"choice": 1, pos: pos_df, spd: speed_df},
            "cov_df": cov_df_out_common,
            "spike_counts": spike_counts_out_common,
            "null_csv": f"{base_dir}/analysis/null_model_out_all.csv",
        },
        "choice_temporal_model_all": {
            "formula_lhs": "spike_count",
            "terms": ["choice", tp, spd],
            "delta_df": {"choice": 1, tp: tp_df, spd: speed_df},
            "cov_df": cov_df_out_common,
            "spike_counts": spike_counts_out_common,
            "null_csv": f"{base_dir}/analysis/null_model_out_all.csv",
        },
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
    "bs(pos_scaled, df = 8)":                             "position",
    "bs(speed_scaled, df = 4)":                           "speed",
    "cr(trial_progress, df = 6, constraints = 'center')": "trial_progress",
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

    df = drop_one_results.copy()
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






# ── 4. Visualization ──────────────────────────────────────────────────────────
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
