# NCBS Lecture 4 — Spike Stimulus Analysis: Summary

---

## Why explicit statistical models?

Descriptive measures (PSTH, ISI histogram, rate map) only describe averages. Explicit models allow:
- Measuring how well the data is described by a particular model
- Comparing models and testing hypotheses
- Calculating confidence intervals on parameters
- Controlling for multiple simultaneous effects (e.g. position *and* speed)
- Making modelling assumptions explicit

---

## Temporal Point Processes

A **point process** is a stochastic process generating discrete events at random times (spikes, earthquakes, etc.). A **temporal** point process evolves in time only.

A spike train can be represented as:
- **Spike times**: {S₁, S₂, S₃, ...}
- **Waiting times (ISIs)**: {X₁, X₂, X₃, ...}
- **Counting process**: N(t) — cumulative spike count up to time t
- **Discrete increments**: binary sequence of 0s and 1s at each time bin

---

## Homogeneous Poisson Process

The simplest model. **Spiking probability does not depend on past spikes** (memoryless). Rate λ is constant over time.

Spike count in a window:

$$P(k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

**Key properties:**
- Stationary increments: count distribution depends only on interval length, not position in time
- Non-overlapping intervals are independent
- Memoryless: time to next spike does not depend on time since last spike
- ISIs are **exponentially distributed**: f(x) = λ exp(−λx)

**Why Poisson over Gaussian:**
- Spike counts are non-negative integers — Gaussian allows negative values
- Poisson links mean and variance with a single parameter λ (higher rate → higher variability); Gaussian requires two separate parameters

**Limitation:** real neurons have covariate-dependent rates, refractory periods, bursting, and oscillatory modulation — none of which is captured by a constant-rate memoryless model.

---

## Inhomogeneous Poisson Process

Allow the rate to vary: **λ(t)** — the probability of a spike in a small interval is λ(tᵢ)Δt.

$$\lambda(t) = \lim_{\Delta t \to 0} \frac{\Pr(\Delta N_{(t,t+\Delta t]} = 1)}{\Delta t}$$

Expected spike count in [a, b]:

$$\mu = \int_a^b \lambda(u)\, du$$

Probability of k spikes in [a, b]:

$$P(N(b) - N(a) = k) = \frac{\left(\int_a^b \lambda(u)\,du\right)^k e^{-\int_a^b \lambda(u)\,du}}{k!}$$

The rate can depend on **any covariate**, not just time:

$$\lambda(t) = f(X(t))$$

This makes place fields a natural inhomogeneous Poisson model: λ is higher at certain spatial positions. Counts in intervals remain Poisson distributed.

**Log-likelihood of a spike train:**

$$\log L = \sum_{i=1}^{n} \log \lambda(t_i) - \int_0^T \lambda(t)\,dt$$

ISI distribution of the inhomogeneous Poisson:

$$f_{S_i}(s_i \mid S_{i-1} = s_{i-1}) = \lambda(s_i)\exp\left\{-\int_{s_{i-1}}^{s_i} \lambda(t)\,dt\right\}$$

---

## Non-Poisson Point Processes: Renewal Processes

To handle **history dependence** (refractory periods, bursting), use a **renewal process**: spiking probability can depend on the time of the *last* spike (but not earlier spikes).

Specified by an ISI distribution. Common choices:

- **Gamma distribution**: flexible 2-parameter distribution; exponential is a special case (α=1). Higher α → mode shifts away from zero (models refractory period).
- **Inverse Gaussian**: another 2-parameter distribution; useful for modelling sharp refractory cutoffs.

**Limitations of renewal processes:**
- Only one-step memory
- Cannot model dependence on multiple past spikes
- Cannot naturally incorporate external covariates

---

## Conditional Intensity Function (CIF)

A unified framework that generalises all of the above.

$$\lambda(t \mid \mathcal{H}_t) = \lim_{\Delta t \to 0} \frac{\Pr(\Delta N_{(t,t+\Delta t]} = 1 \mid \mathcal{H}_t)}{\Delta t}$$

where **H_t** is the full history of past spikes up to time t.

Special cases:
- λ(t | H_t) = λ₀ → **Homogeneous Poisson**
- λ(t | H_t) = λ(t) → **Inhomogeneous Poisson**

The full model conditions on both history and covariates: **λ(t | H_t, X(t))**

Notes:
- If λ depends deterministically on history → not Poisson overall, but still conditionally Poisson in infinitesimal bins
- If λ depends on stochastic spike history → **doubly stochastic** process

---

## Poisson Regression (GLM for Spiking)

To make the CIF estimable, parameterise the log firing rate as a **linear function of covariates**:

$$\lambda(t) = \exp(\eta(t)), \quad \eta(t) = \beta^\top X(t)$$

This guarantees:
- λ(t) > 0 always
- Convex log-likelihood (unique global maximum)
- Tractable optimisation

**Parameter estimation:** maximise the log-likelihood over β. The peak of the log-likelihood gives the MLE; the curvature gives uncertainty (Fisher information).

### Basis functions for continuous covariates

**Indicator functions** (histogram bins): many parameters, no smoothness enforced between adjacent bins.

**Splines**: piecewise polynomials that enforce smoothness at the joins. Far fewer effective parameters for the same flexibility. B-splines are a common choice.

### History dependence in the GLM

Add a **post-spike filter h** as a covariate: the recent spike history is convolved with h and added to the linear predictor. This captures refractory suppression (h < 0 at short lags) and bursting (h > 0 at intermediate lags).

### Full GLM architecture (Pillow et al.)

```
stimulus → [stimulus filter k] → (+) → [exp nonlinearity] → [probabilistic spiking] → spike train
                                   ↑
                     [post-spike filter h] ←──────────────────────────────────────────┘
```

- **k**: linear stimulus filter (spatial/temporal receptive field)
- **h**: post-spike filter (captures spike-history effects: refractory period, bursting)
- **exp nonlinearity**: ensures non-negative rates
- Output: Poisson spiking conditioned on the instantaneous rate

---

## Summary of Model Hierarchy

| Model | Memory | Covariates |
|---|---|---|
| Homogeneous Poisson | None | No |
| Inhomogeneous Poisson | None | Yes (time-varying rate) |
| Renewal process | Last spike only | No |
| CIF / Poisson GLM | All past spikes (via filter h) | Yes |

*Reference: Kass, Eden, Brown (2014)*
