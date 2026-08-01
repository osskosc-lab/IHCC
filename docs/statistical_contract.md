# IHCC v1.1 statistical contract

## Estimand

For a candidate history subset `A`, the estimator targets the largest classifier-induced total-variation lower bound over balanced intervention assignments. Assignment-pair selection is performed only on the outer training fold. The fitted classifier is evaluated only on its untouched outer fold.

For equal priors, the operational estimate is

```text
TV_f = max(0, 2 * balanced_accuracy_f - 1).
```

The subscript is material: unless the classifier family contains the Bayes rule, this is not the unrestricted TV distance.

Within each outer training fold, gradient boosting is fitted on 75% of the balanced training observations and Platt-calibrated on the remaining locked 25%. The untouched outer fold alone is used for TV evaluation. This avoids the roughly threefold training multiplication of nested calibration while preserving strict selection/calibration/evaluation separation.

## Simultaneous lower bounds

With 16 history atoms and candidate order at most two there are 136 hypotheses. A 1000-draw bootstrap cannot resolve the Bonferroni tail `0.05 / 136`. Candidate inclusion therefore uses one-sided exact binomial bounds on both class-conditional accuracies, with the family alpha split over candidates and classes. This is conservative and finite-sample valid conditional on the cross-fitted predictions.

The configured bootstrap count is used for unadjusted candidate uncertainty and summary confidence intervals. Permutations are used for audit p-values. Neither replaces the simultaneous inclusion bound.

## Empty-basis convention

If both truth and estimate are empty, precision, recall, and F1 are defined as 1. If only one is empty, all three are 0.

## Confirmatory lock

Only the `confirmatory` profile is eligible for the final G/M decision. Smoke and pilot profiles report `DIAGNOSTIC` or `PROVISIONAL`; their gate values must not be promoted to Phase 1 support.

## Mechanism boundary

Scenario 9 deliberately gives the quantum-like generator and contextual classical simulator the same finite intervention response table. Equality is the correct result. Any mechanism label other than `indistinguishable` is an implementation failure.
