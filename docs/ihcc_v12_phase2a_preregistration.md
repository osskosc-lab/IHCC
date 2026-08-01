# IHCC v1.2 Phase 2A preregistration

## Locked question

Can a history-augmented predictor detect synthetic memory structure that cannot be adequately represented by an optimized three-exponential state class, under intervention-environment holdout?

The claim is explicitly relative to the fixed candidate class

\[
\mathfrak F_3 = \left\{\sum_{j=1}^{3} w_j e^{-\lambda_j\tau}: \lambda_j>0\right\}.
\]

It is not a claim of representation-invariant non-Markovianity.

## One proposition, one metric, one baseline, one falsification

- Proposition: state-class-external history is detectable in unknown-intervention prediction.
- Primary metric: relative OOD MSE gain `G_hist,3`.
- Baseline: optimized three-exponential state model.
- Main falsification: best three-exponential projection of the true finite power-law kernel.
- Practical threshold: `G_hist,3 > 0.01`.
- Confirmatory rule: the lower endpoint of the 95% seed-level CI must exceed `0.01`.

## Locked primary condition

- Sequence length: `T=64`.
- Input: independent `{-1,+1}` interventions.
- Training environments: `P(+1) in {0.35, 0.50, 0.65}`.
- Primary holdout: `P(+1)=0.80`.
- Reverse-direction holdout: `P(+1)=0.20`.
- True kernel: `k(tau) proportional to (tau+1)^(-0.50)`.
- Noise standard deviation: `0.50`.
- Primary state budget: `B=3`.

## Generation-before-analysis gate

Before outcomes are generated, the strongest allowed three-exponential approximation is computed under the pooled training input second moment:

\[
E_{\mathrm{proj},3}=
\frac{\|k^\star-k_{\Phi_3}\|^2_{\Sigma_U}}
{\|k^\star\|^2_{\Sigma_U}}.
\]

The positive control is admissible only when

\[
E_{\mathrm{proj},3}\ge 0.05.
\]

This gate is intentionally hard. If it fails, the confirmatory run stops with `DESIGN_BLOCKED_X1`. The implementation must not lower the threshold, weaken the optimizer, or switch kernels after seeing the result.

## Oracle gain calibration

After the projection residual variance is fixed, the outcome coefficient is calibrated to target

\[
G_{\mathrm{oracle},3}=0.05
\]

using

\[
\beta=\sqrt{\frac{G\sigma^2}{(1-G)V_{\mathrm{res},3}}}.
\]

Calibration does not rescue a failed projection-floor gate. It controls effect size only after the positive control has been shown to lie sufficiently far outside the candidate class.

## Formal gates

- `X1`: relative projection error at least `0.05`.
- `X2`: Scenario B 95% LCB for `G_hist,3` above `0.01`, with seed detection rate at least `0.80`.
- `X3`: Scenario A 90% CI contained in `[-0.01, 0.01]`.
- `X4`: budget-gradient tolerance in at least 95 of 100 seeds.
- `X5`: at least 80% power for oracle gain at least `0.03`, with false-positive rate at most 5% at gain zero.
- `X6`: Energy, Wasserstein-1, and classifier-TV diagnostics must not reverse positive/negative-control direction.
- `X7`: positive gain in both `P(+1)=0.80` and `P(+1)=0.20` holdouts.
- `X8`: Python 3.11 and 3.12 gate decisions identical, primary means within `0.002`, identical exclusions and row counts.

## Current implementation stage

This branch implements the preregistered projection-floor preflight, oracle-gain calibration, stop decision, required artifact contract, PDF/Markdown report generation, and Python-version CI matrix.

Downstream outcome generation is deliberately blocked whenever `X1` fails. This is a scientific stop rather than a software error.

## Scope boundary

Even a later full pass would support only this statement:

> Within the fixed finite intervention design and optimized three-exponential state class, a synthetic history component outside that class can be detected by OOD prediction.

It would not support an absolute non-Markov entity, impossibility of all finite-dimensional state representations, a quantum mechanism, Orch OR, consciousness, or a soul-information particle.
