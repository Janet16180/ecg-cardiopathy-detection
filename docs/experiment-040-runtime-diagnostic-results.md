# Experiment 040: runtime diagnosis

All 18 full training fits remain unexecuted. This diagnosis measures runtime only; it does not authorize a new full training schedule.

All six 200-update probes completed. Estimated remaining execution is **2.639 hours centrally** and **3.853 hours in the conservative scenario**, including a separate 900-second reporting allowance.
These are nonprobabilistic extrapolations, not measured full-run durations or confidence intervals.

## Original rejected projection

| Component | Seconds |
| --- | ---: |
| training | 12329.189081 |
| checkpoint | 265.902918 |
| prepare | 114.246842 |
| readout | 473.112840 |
| audit | 284.717292 |
| profiles | 331.107078 |
| features | 128.558097 |
| report allowance | 900.000000 |
| total | 14826.834148 |

The immutable admission used 10868.191081 seconds already charged, 14826.834148 seconds remaining and a 3,600-second correction reserve: 29295.025229 seconds against 28,800.

The historical full readout includes feature extraction; adding another extraction estimate overlaps costs. The new scenarios subtract the historical extraction proxy before adding the measured new path. That subtraction is inferred from old profiles, because the old readout did not separately time extraction.

Original checkpoint profiling times serialization and reload. Routine checkpoint projections here include CPU capture and writing only. Active all-in training pace would already include checkpoint/guard overhead; these projections instead use isolated update timers and add checkpoint writing once. The original forecast also omitted separate training-stage setup: its matched historical residual is retained here.

The original spent-plus-active accounting does not itself duplicate completed stage charges. The 1.5 multiplier is an explicit conservative allowance, not a measured error.

## Measured sustained timing

| Package | First 24 s/update | 25–100 | 101–200 selected | Full feature pass 1 s | Pass 2 selected s |
| --- | ---: | ---: | ---: | ---: | ---: |
| cpc_gru | 0.140110 | 0.103788 | 0.104470 | 3.294286 | 2.541491 |
| cpc_xlstm | 0.104880 | 0.098725 | 0.099224 | 2.831762 | 2.970472 |
| simdino_gru | 0.158801 | 0.151488 | 0.154886 | 2.491945 | 2.605760 |
| simdino_xlstm | 0.192248 | 0.191990 | 0.194883 | 2.940369 | 2.896505 |
| hybrid_gru | 0.230593 | 0.228013 | 0.228270 | 2.578258 | 2.577391 |
| hybrid_xlstm | 0.282376 | 0.287036 | 0.289041 | 2.954944 | 2.973968 |

The prespecified sustained window uses synchronized block wall times for updates 101–200, excluding checkpoint I/O. Each full fit uses 1,953 normal batches plus one measured 16-record update, with 20 checkpoint boundaries. The second complete 15,359-record training feature pass is selected regardless of whether it is faster. Its per-record rate is extrapolated to the 1,306 development records; no development waveform was timed or scored.

## Future execution scenarios

| Component | Central seconds | Conservative seconds |
| --- | ---: | ---: |
| normal updates | 6273.659014 | 9410.488521 |
| final partial update | 1.578860 | 2.368290 |
| checkpoint capture and write | 98.903863 | 148.355794 |
| feature extraction | 53.922557 | 80.883836 |
| model initialization | 7.201316 | 10.801974 |
| all package validation | 1081.968802 | 1622.953202 |
| time gate evaluation proxy | 166.940512 | 250.410768 |
| prepare proxy | 75.481463 | 114.246842 |
| readout nonfeature proxy | 253.935873 | 395.935895 |
| audit proxy | 188.339885 | 284.717292 |
| train stage setup proxy | 178.121101 | 317.102677 |
| remaining profile proxy | 220.738052 | 331.107078 |
| report allowance | 900.000000 | 900.000000 |
| total | 9500.791296 | 13869.372168 |

Central historical proxies use the mean of three matched 039 patch cells; conservative historical proxies use their maximum with the 1.5 multiplier. Other variable components are multiplied by 1.5. The 900-second reporting allowance and 3,600-second correction reserve are each counted once.

The new all-package source-validation cost is counted for 27 train/readout/audit stage gates. The separately measured time gate is counted for nine training starts and 360 checkpoint pace checks. Nested validation reproduces the execution wrapper. These costs are additional to matched 039 proxies, which lacked the 040 gate.

## Accounting

| Measured diagnostic subdivision | Seconds |
| --- | ---: |
| all package validation | 40.072919 |
| dataset load | 35.326929 |
| final parent verification | 1.550480 |
| framework init | 0.122835 |
| source verification | 3.433715 |
| time gate evaluation | 0.452413 |
| cpc_gru whole package | 31.376638 |
| cpc_xlstm whole package | 27.152742 |
| simdino_gru whole package | 37.820726 |
| simdino_xlstm whole package | 46.472250 |
| hybrid_gru whole package | 52.624062 |
| hybrid_xlstm whole package | 64.928473 |
| Whole probe process body | 341.651230 |

Central spent-plus-future scenario: 24551.837581 seconds (6.820 hours), including the correction reserve.
Conservative spent-plus-future scenario: 28920.418453 seconds (8.033 hours), including the correction reserve.

Closed parent work: 11005.541631 seconds. New diagnostic work charged before this report: 445.504653 seconds. Package and phase times are subdivisions of whole-process work and are not added again to the ledger. The final report invocation and any later audit are charged separately; consult the live diagnostic ledger for the final total.

## Limits and evidence

Cold metadata validation, dataset loading and framework/model initialization are outside update timers. Historical setup and CPU head/audit costs are proxies; new objective features can change classifier convergence and full-run costs. Two hundred updates, one initialization and repeated warm-cache feature reads cannot establish three-seed, long-duration throughput. Later time gates also rehash completed-cell artifacts, so their cost may exceed this pretraining measurement; per-update elapsed-guard JSON overhead was not isolated. Drift is retained in all four 50-update windows in the analysis receipt. No fastest-window selection is used.

Original 039/040 receipts and scientific code remain immutable. No head fitting, development performance scoring, closed-set access, new label definition or Experiment 008 restart is part of this diagnosis. Any repaired execution or full training needs a prospective successor identity and user authorization.

Evidence: `outputs/experiment040_runtime_diagnostic/result.json`, package receipts, `projected_runtime.json` and `day_ledger.json`; the analysis records hashes of parent, protocol, source and measured inputs, plus the accounting snapshot used for its estimates.

## Executed timing and independent verification

The full diagnostic process completed in **350.766804 seconds (5m 50.77s)**. This includes 9.111754 seconds of measured interpreter/CLI overhead outside the inner probe timer. These times are already represented in the diagnostic ledger; they are not added to package times again.

The projected optimizer updates across all 18 fits total **6275.237874 seconds (1h 44m 35s)**. The full remaining plan is **9500.791296 seconds (2h 38m 21s)** centrally and **13869.372168 seconds (3h 51m 09s)** conservatively. The full plan includes readouts, validation, preparation, checkpoints and the 900-second reporting allowance. The separate one-hour correction reserve is added only in the spent-plus-future comparison.

The independent audit verified six package receipts, twelve saved checkpoints, exact 16-record model/optimizer/global-and-auxiliary-RNG recovery, the prescribed EMA clock, synchronized timing blocks, 31 unchanged parent artifacts and 9 pinned runtime sources. It performed no additional GPU updates or waveform reads. Evidence is `independent_audit.json`, with its auditor hash and input hashes.

An initial launch failed on a syntax/import error before the runtime manifest or GPU measurements. It is retained in `launch-error.log` and the ledger; the corrected source was committed before the successful probe. This was a startup/tooling failure, not a numerical model failure.

## Original short profile versus new measurements

| Package | Original 24-update s/update | New first 24 | New 101–200 |
| --- | ---: | ---: | ---: |
| cpc_gru | 0.223869 | 0.140110 | 0.104470 |
| cpc_xlstm | 0.147943 | 0.104880 | 0.099224 |
| simdino_gru | 0.307570 | 0.158801 | 0.154886 |
| simdino_xlstm | 0.191258 | 0.192248 | 0.194883 |
| hybrid_gru | 0.241790 | 0.230593 | 0.228270 |
| hybrid_xlstm | 0.289729 | 0.282376 | 0.289041 |

The original and new first-24 measurements already differ, especially for CPC+GRU and SimDINO+GRU. The within-probe first-24 versus sustained comparison therefore does not explain the entire change. Startup, timing moment, hardware load and other execution conditions were not experimentally isolated; no causal attribution to warmup alone is supported.

The 27 source-validation calls represent nine cells times train/readout/audit entry. Their measurement reproduced the outer `study.configured` wrapper and the nested three-objective manifest validation. The 369 time-gate evaluations represent nine training-stage starts plus 20 checkpoint guards per arm across 18 arms. Future completed-cell hashing and every-update elapsed-guard costs remain limitations. Model initialization is counted separately because the 039 arm timers already included their own initialization, so subtracting those arm times from the historical train stage leaves a residual that excludes training-model creation.

## Post-report accounting snapshot

After the report CLI and its measured outer startup/import gap, new diagnostic charges were **453.329598 seconds**. The central spent-plus-future-plus-reserve snapshot is 24559.662525 seconds; the conservative snapshot is 28928.243398 seconds, **128.243398 seconds over eight hours**. This does not pass or replace the original admission gate. This evidence append and any later checks are charged after the snapshot; the final ledger remains authoritative.

## Final local verification and accounting

Ruff passed for all project Python files and the forward test fixture. The full CPU suite produced 1,207 passes and one existing frozen v11 path-guard failure caused by the worktree's shared-output symlink. The identical test/helper/verifier source passed in the canonical main checkout (one test; 3.28 seconds reported, 6.789124 seconds whole process). No test assertion was removed or deselected. The forward root `conftest.py` supplies the complete context identity only to the frozen runtime synthetic case; the measured scientific files remain unchanged.

The normal isolated build produced wheel and source distributions. A preceding attempt without build isolation failed because that environment lacked hatchling; it is retained as a failed tooling charge, not an experiment failure. The new ledger also discloses a separate 30-second allowance for final metadata, build and documentation work whose individual clocks were not isolated; this allowance is not a GPU measurement.

Final diagnostic charge: **598.426965 seconds**, including retained failures, reported CPU checks, whole-process GPU probes, independent audit, reporting and the explicit allowance. The original parent charge stays **11005.541631 seconds**. The final central spent-plus-future-plus-reserve scenario is **24704.759892 seconds**; the conservative scenario is **29073.340765 seconds**, still above 28,800. Closure and ledger binding are in `outputs/experiment040_runtime_diagnostic/execution_closed.json`.

A second already-started isolated check also passed using the worktree source with only the test-local root pointed at the canonical checkout (0.02 seconds reported; 4.047792 seconds whole process). It was charged on agent handoff before publication. The preceding accounting snapshot is retained as `accounting_before_late_validation.json`; the final closure binds the updated ledger. Finalization imports/metadata are covered by the disclosed 30-second allowance.
