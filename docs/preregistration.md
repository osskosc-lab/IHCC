# IHCC v1.1 Phase 1 preregistration

## Locked primary question

Can the estimator recover the embedded inclusion-minimal IHCC basis under a finite intervention set, fixed state class, and `epsilon = 0.10`?

- Primary metric: exact-set macro `F1_basis` over Scenarios 1--3.
- Baseline: the unpruned significant subset family.
- Primary falsification conditions: pure synergy (Scenario 2) and redundant supersets (Scenario 3).
- Adoption threshold: `F1_basis >= 0.80` and `F1_minimal - F1_unpruned >= 0.10`.
- Seeds: 100 confirmatory seeds, generated deterministically from base seed 20260801.
- Trials: 12000 randomized intervention trials per seed and scenario.
- Candidate family: all singletons and pairs among 16 atoms.
- Main noise: Gaussian with sigma 0.50.

## Locked audits

G1 controls null false cones. G2 detects the positive order control. G3 tests reset-budget monotonicity. G4 requires disappearance after oracle reset. G5/G6 distinguish missing-state gain from complete-state equivalence. G7 requires the finite-response classical mimic to remain indistinguishable.

M1 assesses minimal-basis recovery. M2 compares classifier TV with analytic Gaussian TV and records lower-bound false negatives. M3 separates algebraic noncommutativity from observed order effects. M4 separates post-treatment conditioning from an actual state reset.

## Operational clarifications made before data generation

1. XOR on `{-1,+1}` is fixed as product parity. This yields identical marginal intervention distributions for either input alone and distinct joint distributions.
2. Scenario 3 uses a singleton effect and its pairwise redundant supersets. Adding redundancy to the Scenario 2 pair would require a third-order candidate, outside the locked search family.
3. Main simultaneous lower bounds use exact Bonferroni-distributed Clopper--Pearson limits; 1000 bootstrap draws are retained for ordinary uncertainty summaries because they cannot resolve the required simultaneous tail.
4. Gate failure is a scientific result and does not make CI fail. CI fails only on tests, runtime exceptions, malformed output, or non-finite values.

## Stopping rule

The confirmatory run ends in exactly one of:

- `SUPPORTED_IN_SYNTHETIC_SCOPE`
- `NOT_SUPPORTED`
- `IMPLEMENTATION_FAILURE`

No Phase 1 path may output support for a quantum mechanism, Orch OR, or consciousness.
