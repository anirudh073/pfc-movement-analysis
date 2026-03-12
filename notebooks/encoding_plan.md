# Encoding Model Workflow Plan
Generated: 2026-03-10 by Claude Sonnet 4.6

Based on: NCBS Lecture 4, `02b-poisson-regression_.ipynb`, `encoding.ipynb`

---

## Key differences from tutorial notebook

- Data has **linear position** (1D along W-track) — use this instead of 2D x,y. Fewer parameters, more occupancy per bin, better conditioned.
- 263 units — store only `(AIC, log-likelihood, n_params, coefficients)` per unit per model, not full model objects.
- Covariates already available: `speed`, `linear_position`, `trial_type` (inbound/outbound), `left/right`, `zone`, `epoch`, `trial_number`.

---

## Step 0 — Shared data prep

Already started in `encoding.ipynb`. Add a run-zone mask before fitting any models:

```python
run_mask = (interp_trialised_position["zone"] == "run") & \
           (interp_trialised_position["trial_type"].isin(["inbound", "outbound"]))

cov_df = interp_trialised_position[run_mask].copy().reset_index(drop=True)
```

**Important:** spike counts must be re-histogrammed using only `bin_centers[run_mask]` so rows align with `cov_df`.

---

## Step 1 — Single variable models (all 263 units)

Prototype each formula on unit 9 first, then loop.

### 1a. Null model — constant rate
```python
# spike_count ~ 1
```
Gives: mean firing rate, baseline AIC/LL for all downstream comparisons.

### 1b. Trial direction — inbound vs outbound
```python
# spike_count ~ trial_type
```
Categorical: "inbound" / "outbound". β₁ = log-ratio of inbound/outbound rates.

### 1c. Left/right choice — outbound trials only
```python
# Subset mask: trial_type == "outbound"
# spike_count ~ C(Q("left/right"))
```
Compare to an outbound-only null model (not the global null).

### 1d. Speed — linear
```python
# spike_count ~ speed
```
β₁ = % change in rate per cm/s.

### 1e. Speed — spline
```python
# spike_count ~ bs(speed, df=5)
```
Compare AIC vs linear speed to detect non-linear tuning.

### 1f. Linear position — spline (place field)
```python
# spike_count ~ bs(linear_position, df=8)
```
1D equivalent of the 2D xy spline model. This is the primary place field model.

---

## Step 2 — Combined models (across-stable units only, n=177)

Restrict to `across_stable_unit_ids` from `unit_tuning.ipynb` to reduce compute.

### 2a. Position + direction
```python
# spike_count ~ bs(linear_position, df=8) + trial_type
```
Does the place field shift between inbound and outbound? β_direction = marginal direction effect after accounting for position.

### 2b. Position + speed
```python
# spike_count ~ bs(linear_position, df=8) + bs(speed, df=5)
```
Are speed effects independent of where on the track the animal is?

### 2c. Full model — position + speed + direction
```python
# spike_count ~ bs(linear_position, df=8) + bs(speed, df=5) + trial_type
```

### 2d. Full + left/right (outbound only)
```python
# spike_count ~ bs(linear_position, df=8) + bs(speed, df=5) + C(Q("left/right"))
```
Does position tuning depend on the animal's upcoming arm choice?

---

## Step 3 — Population summary

After fitting all units × models, collect into a results DataFrame:

```python
# columns: unit_id, model_name, AIC, log_lik, n_params, converged
```

Key comparisons:
1. **Rank models by AIC** per unit — which variable best explains each unit?
2. **Likelihood ratio test** (each single-variable model vs null) — which units are significantly tuned to each variable? (Note: LRT requires nested models; AIC can compare any models on the same data.)
3. **LRT: full vs position-only** — does speed/direction add explanatory power over and above position?
4. **Count** how many of 263 units are significantly tuned to each variable.

---

## Helper function for looping over units

```python
def fit_glm_all_units(formula, cov_df, spike_counts_masked, unit_ids, bin_size=0.002):
    """Fit one GLM formula to all units. Returns summary DataFrame."""
    rows = []
    for i, uid in enumerate(unit_ids):
        df = cov_df.copy()
        df["spike_count"] = spike_counts_masked[i]
        try:
            res = smf.glm(formula, data=df, family=sm.families.Poisson()).fit(disp=False)
            rows.append(dict(unit=uid, aic=res.aic, llf=res.llf,
                             n_params=len(res.params), converged=res.converged))
        except Exception as e:
            rows.append(dict(unit=uid, aic=np.nan, llf=np.nan,
                             n_params=np.nan, converged=False))
    return pd.DataFrame(rows)
```

---

## Model comparison table (target output)

| Model | Variables | n_params | Notes |
|---|---|---|---|
| Null | — | 1 | Baseline |
| Direction | trial_type | 2 | Inbound vs outbound |
| Speed (linear) | speed | 2 | Linear effect |
| Speed (spline) | bs(speed, 5) | 6 | Non-linear tuning |
| Position | bs(linear_position, 8) | 9 | Place field |
| Position + direction | above + trial_type | 10 | Marginal direction |
| Position + speed | above + bs(speed, 5) | 14 | Marginal speed |
| Full | position + speed + direction | 15 | Best combined |
| Full + choice | above + left/right | 16 | Outbound only |
