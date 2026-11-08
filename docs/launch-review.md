# Launch review and experiment plan

This document specifies the product experiment and operational governance plan.

1. Product defines the MVP action: a scored review queue with human decisions. Policy writes the harm taxonomy, ambiguous-label examples and appeal rules. Operations supplies reviewer capacity and welfare practices. Engineering implements stable assignment, score logging and rollback. Data science owns metric definitions, sample size, randomization diagnostics and analysis.
2. Begin with shadow scoring. Review threshold stability and the capacity frontier. Inspect errors with minimum necessary exposure to source content. Escalate disagreements rather than treating an annotation as an unquestionable policy verdict.
3. The experiment randomizes users, retains all assigned users for intention-to-treat analysis, clusters uncertainty by user, and checks sample-ratio mismatch before outcome analysis. Avoid assignment by post when users generate multiple correlated observations. Freeze the primary metric, analysis window and sequential monitoring rule before launch.
4. Primary metric: independently audited violating exposures per eligible user. Guardrails: false positive action rate on an independently adjudicated sample, creator appeal overturn rate, reviewer queue age/capacity, creator retention and user engagement. Log each denominator and missing-data rate.
5. Planning inputs in `results/metrics.json`: 1% baseline user harm rate, a 0.2 percentage point absolute reduction, two-sided alpha .05, 80% power, 50/50 assignment, independent-user approximation. The normal-approximation calculation returns users per arm.
6. A release reviewer checks data coverage, policy approval, capacity, precision/recall intervals, guardrail limits and rollback readiness. Keep the model in shadow mode until operational prerequisites are met.

Root-cause questions: Is a review-rate change driven by source mix, text length, data loss, annotation drift or a new score distribution? Reconcile event counts and denominators first. Compare stable cohorts and overlapping slices carefully; investigate label disagreements and threshold effects before changing enforcement.
