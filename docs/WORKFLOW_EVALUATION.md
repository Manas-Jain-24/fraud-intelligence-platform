# Whole-workflow evaluation and review capacity

The complete original PaySim stream contains 6,362,620 events. This evaluation reconstructs history from that stream, rather than treating the earlier 250,000-row sample as complete account history. It uses the previously fitted chronological `wire_pre` model without retraining.

The earlier GitHub Actions run for commit `86232dc` passed its Python test matrix and container build: [workflow run](https://github.com/Manas-Jain-24/fraud-intelligence-platform/actions/runs/44166125212).

## Reproduce

```powershell
python scripts/evaluate_workflow.py --input "C:\path\to\PS_20174392719_1491204439457_log.csv"
```

Generate the chronological pre-transfer artifact with `scripts/evaluate_operations.py` first. The replay checks the original source checksum against the fitted model's provenance. It evaluates all 954,990 events in the original validation hours (323–376) and all 973,905 events in test hours 377–741. Validation contains 560 frauds; test contains 3,998. These are full-population time windows, so their counts differ from the earlier sampled model evaluation.

History and rule expressions are shared with the investigation service. Array-based reconstruction is checked against the scalar history implementation. Equal-hour events never enter one another's history. Fraud labels are not used for history or scores. Model cutoffs and rule settings are selected on validation; test results measure the locked choices.

PaySim has hourly timestamp precision. Its records cannot establish five-minute bursts or fifteen-minute cash-outs. Those short windows stay unobserved in this replay; no minute-level timestamps or within-hour ordering are invented. Synthetic minute-level examples continue to exercise those code paths in tests and the demonstration workflow.

## Held-out queue comparison

The default capacity scenario assumes **two analysts, eight working hours per day, and eight minutes per case: 120 reviews per full day**. Partial evaluation days have proportionately smaller budgets. These staffing inputs are assumptions, not observed productivity.

| Routing policy | Review cases | False alerts | Precision | Fraud recall | Peak backlog | Final backlog |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing model cutoff | 5,826 | 2,161 | 62.9% | 91.7% | 4,001 | 4,001 |
| Model plus original default rules | 5,834 | 2,169 | 62.8% | 91.7% | 4,009 | 4,009 |
| Capacity-selected cutoff, model only | 1,729 | 9 | 99.5% | 43.0% | 33 | 0 |
| Capacity-selected combined policy | 1,729 | 9 | 99.5% | 43.0% | 33 | 0 |

The observable default rules added **13 false alerts and no additional frauds on validation**, then **eight false alerts and no additional frauds on test**. Under the stated costs, the validation search selected disabled extra rules. These results do not establish that behavioural rules are generally ineffective; the dataset's account identities and hourly resolution limit the observable patterns.

The capacity-selected cutoff is approximately 0.89183, compared with the existing 0.01814 cutoff. Although it met every validation day's budget, three later test days exceeded capacity. Its backlog eventually cleared, while recall fell substantially. The report exposes both facts; it does not suppress excess alerts or reinterpret them as approved transactions.

Each original source row creates at most one case after the union of model and rule alerts. Overlapping triggers are counted separately in `overlapping_alerts_merged`. Repeated submission idempotence is tested through the SQLite service; this experiment does not claim that the original source contains duplicate submission IDs.

## Staffing sensitivity

Each scenario selects its own model cutoff on validation before measuring the same held-out period. Rule routing is disabled in this sensitivity comparison.

| Analysts | Daily capacity | Test precision | Test recall | Review cases | Days over capacity | Peak backlog |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 60 | 100.0% | 18.9% | 755 | 1 | 13 |
| 2 | 120 | 99.5% | 43.0% | 1,729 | 3 | 33 |
| 4 | 240 | 95.8% | 73.1% | 3,050 | 1 | 46 |
| 8 | 480 | 88.6% | 78.8% | 3,557 | 0 | 0 |
| 16 | 960 | 75.6% | 86.9% | 4,595 | 0 | 0 |
| 24 | 1,440 | 66.2% | 89.4% | 5,395 | 0 | 0 |

Queue processing is an aggregate daily simulation: processed cases are limited to the stated daily capacity, and unfinished work carries forward. It does not model shift calendars, intraday case arrivals, or actual investigator accuracy.

## Local rule selection

To select between original and disabled rules at the **existing model cutoff**, using validation cost only:

```powershell
python scripts/evaluate_workflow.py --activate-rules-only
```

This was applied locally for the evaluated pre-transfer model. It disables the extra wire rules and retains the existing model cutoff. The full 120-case/day cutoff remains a comparison scenario. The decision policy and rule policy are recorded together and apply only when the exact fitted model version matches. A new model version cannot accidentally inherit this selection.

For an explicitly chosen capacity scenario, the replay command also supports `--analysts`, `--hours-per-day`, `--minutes-per-case`, and `--activate`. That activation changes both cutoff and rule policy. Inspect the report's recall, missed amounts and overload days first; capacity fit is not a guarantee of future recall or workload.

## Outputs and dashboard

Open **Measured Evaluation → Whole-workflow evaluation and analyst capacity**. It shows queue comparisons, staffing sensitivity, daily backlog and the currently active local policy.

Small reports are stored under `data/artifacts/workflow/`:

- `report.json`: provenance, model/policy versions, assumptions, selection and results.
- `queue_comparison.csv`: validation and test metrics for all four routing strategies.
- `daily_workload.csv`: new cases, capacity, processed cases and carry-over backlog.
- `validation_policy_search.csv`: feasible rule/cutoff candidates selected without test tuning.
- `capacity_sensitivity.csv`: staffing scenarios with independently validation-selected cutoffs.

The approximately 91 MB row-level test export and local active policy are ignored by Git. Recreate them with the commands above. Model scores remain separate from investigation decisions and confirmed labels. Financial cost fields use the earlier illustrative dataset-unit assumptions; they are not verified savings. PaySim remains a simulation, so this work does not demonstrate performance on a live payment system.
