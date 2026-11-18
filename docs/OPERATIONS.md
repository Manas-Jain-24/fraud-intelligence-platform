# Local fraud investigation and evaluation

This extension turns the research dashboard into a local investigation workflow. The academic notebooks, saved legacy weights and historical experiments are retained. No external deployment or payment integration is performed.

## Run locally

Activate the project environment, then install `requirements_full.txt`.

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements_full.txt
python scripts/prepare_paysim_natural.py --input "C:\path\to\PS_20174392719_1491204439457_log.csv"
python scripts/evaluate_operations.py
python -m streamlit run app.py
```

The preparation script uses the locally cached original PaySim file by default if it is in the usual KaggleHub location. It never downloads a dataset implicitly. Card evaluation requires `data/creditcard.csv`.

The app adds **Investigation Queue** and **Measured Evaluation** tabs. Click **Create synthetic investigation scenario** to create recipient fan-in, rapid cash-out and repeated sender transactions. Open a case, enter an analyst name, assign it, add a note, and resolve it with a reason. Reopening preserves the earlier resolution in the event timeline. Refreshing or restarting the app retains records.

Run the API separately if needed:

```powershell
python -m uvicorn api:app --port 8000
```

Open `/docs` on the API for interactive requests. The dashboard calls the same Python scoring service and SQLite store; it does not require the API process to be running.

## Models and data available at decision time

| Profile | Inputs | Stage | Recommendation for a high score |
| --- | --- | --- | --- |
| `card` | V1–V28 and amount | Pre-transaction research scoring | BLOCK recommendation |
| `wire_pre` | Type, amount, sender/recipient initial balances; derived amount/(initial balance+1) and transfer indicator | Pre-transaction | BLOCK recommendation |
| `wire` | Initial and observed post-transaction balances, ledger discrepancies, drain ratio | Post-transaction | REVIEW |

Recommendations do not execute payment actions. Card PCA coordinates are anonymized features, without asserted merchant, device or velocity meanings. Post-transaction balances are never supplied as zeros to the pre-transfer model; the pre-transfer schema and artifact omit them entirely.

An `is_fraud` response retains the historical API name and means **the model score exceeded the configured high-risk threshold**, rather than confirmed fraud. Analyst-confirmed outcomes are recorded separately.

## Evaluation protocol

`scripts/evaluate_operations.py` evaluates all three profiles using two protocols:

1. Chronological partitions: approximately 55% fit, 15% calibration, 15% validation and 15% final test. Equal timestamps always remain together. Class prevalence can change over time.
2. Stratified random partitions with the same proportions, preserving the original/sample prevalence. These are supplementary comparisons rather than evidence of future performance.

The card dataset is split **before** any training resampling. Only the scratch training subset is balanced, capped at 4,000 examples per class. Scaling statistics use the unresampled training partition. Calibration, validation and test retain their observed class proportions. The library candidates train on the natural training partition.

The natural PaySim sample assigns independent random priorities to all 6,362,620 original rows and retains the smallest 250,000. Fraud labels are never used to select rows. The seed-42 sample contains 308 frauds (0.1232%). Its manifest records the original source checksum and sampling method. This is distinct from the old 100,000-row fraud-enriched benchmark. A sample's observed prevalence will differ from the full dataset by sampling variation; a chronological period can differ further due to temporal concentration of fraud.

Scratch baseline/cost-sensitive logistic regression, MLP, stump GBDT and the serving engine's exact ensemble compete with scikit-learn logistic regression and histogram gradient boosting. The pre-transfer profile compares the library candidates on its separate feature schema. NumPy models remain available for educational inspection.

For each candidate, a sigmoid calibration model fits the logit of its raw score on the **calibration partition only**. Raw and calibrated variants compete on validation. Policy selection minimizes illustrative cost within a maximum validation alert rate (default 2%). Ties use Brier score and log loss. A single selected cutoff controls both case routing and high-risk classification, so there is no extra unmeasured lower alert threshold in the selected policy.

Default cost:

```text
missed fraud amount × loss_fraction
+ all model alerts × review_cost
+ false alerts × false_decline_cost
```

Defaults are loss fraction 1.0, review cost 2.5, false-decline cost 5.0. These are configurable assumptions in dataset amount units, not observed financial savings. For post-transaction review, missed fraud amount represents unresolved exposure; identification after settlement does not establish recovery. The 2% budget is enforced on validation, not guaranteed on future/test data. These assumptions materially affect which model and cutoff win.

```powershell
python scripts/evaluate_operations.py --max-alert-rate 0.01 --review-cost 3 --false-decline-cost 10 --loss-fraction 0.5
```

Only after model and threshold selection does evaluation score the final test partition. It writes `report.json`, validation comparisons, exact split membership and test predictions under `data/artifacts/<profile>/<protocol>/`. The chosen model is saved, reloaded, and required to reproduce its predictions exactly. Reports record runtime versions, source checksums and fitted-model fingerprints. Model metrics and versions exposed by `/v1/models` come from that bundle.

The chronological bundle becomes the locally active model via `active.json`. Use `--no-activate` to generate reports without changing selection. `--output` can create an independent run directory. `FRAUD_ARTIFACT_DIR` selects the artifact directory used by the service. Model bundles, active pointers and row-level prediction files are local generated outputs; recreate them with the command above. Without active artifacts, card/wire fall back to explicitly **uncalibrated legacy** weights; `wire_pre` returns 503 until its artifact exists.

## Current chronological results

| Profile | Selected model | Calibrated | Test frauds / rows | Precision | Recall | False alerts / 10,000 |
| --- | --- | --- | --- | --- | --- | --- |
| card | Scratch soft-voting ensemble | Yes | 52 / 42,724 | 65.5% | 73.1% | 4.68 |
| wire | Sklearn histogram gradient boosting | No | 163 / 38,126 | 100% | 87.1% | 0.00 |
| wire_pre | Sklearn histogram gradient boosting | Yes | 163 / 38,126 | 63.4% | 91.4% | 22.56 |

The post-transaction wire result uses information unavailable to the pre-transfer model. Its high ranking metrics also coexist with large missed fraud amounts. Consult loss and missed-amount fields as well as precision/recall. Calibration is fitted and measured, not assumed: an uncalibrated winner is labelled `uncalibrated_score`. Reliability diagrams, populated-bin counts, Brier scores and log loss are included in reports. Sparse fraud counts make these estimates uncertain.

## Replay the serving artifact

```powershell
python simulations/replay_served_artifact.py --profile card --bootstrap-runs 100
python simulations/replay_served_artifact.py --profile wire --bootstrap-runs 100
python simulations/replay_served_artifact.py --profile wire_pre --bootstrap-runs 100
```

Replay verifies the source checksum, reads the original test membership and scores the exact saved bundle without refitting. Confusion counts, loss and Brier score must match the evaluation report. Optional one-hour block bootstraps describe conditional uncertainty for this fixed model and observed holdout. Reusing observations does not create new independent evidence. Card data spans only a short period; the current card holdout has six hourly blocks, so its interval estimates are particularly limited.

The old `simulation_worker.py` and 1,000-run reports use different model implementations and remain historical research. The replay tool is the operational validation entry point.

## Score and explain consistently

The score and explanation endpoints accept the same optional high-risk `threshold` override. By default they use the selected bundle's policy. They return the same score, model version, decision and applied threshold. `model_breakdown` contains uncalibrated legacy research diagnostics; it is separate from the active model's `risk_score`.

Explanations replace training-mean reference features with the transaction values in schema order and measure the actual scored model after each replacement. Contributions sum exactly to **score minus reference score**, including any calibration. This is an ordered, path-dependent decomposition, not SHAP or a causal explanation. The response supplies the reference, feature order, method, baseline score and explained model. Intermediate reference combinations can be unrealistic, especially for correlated ledger features.

## Persistent transactions and cases

`POST /v1/transactions` ingests, scores and records a transaction. Example:

```json
{
  "profile": "wire_pre",
  "payload": {
    "type": "TRANSFER", "amount": 250,
    "oldbalanceOrg": 1200, "oldbalanceDest": 500
  },
  "context": {
    "transaction_id": "EXAMPLE-001",
    "event_time": "2026-11-18T10:00:00+05:30",
    "sender_id": "CUSTOMER-A", "recipient_id": "CUSTOMER-B",
    "synthetic": true
  },
  "force_case": true
}
```

Other endpoints:

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/score/card` | Stateless active card score plus scratch diagnostics |
| `POST /v1/score/wire` | Stateless post-transaction wire assessment |
| `POST /v1/score/wire/pre` | Stateless pre-transfer score and explanation |
| `POST /v1/explain/card`, `POST /v1/explain/wire` | Explain the actual scored model and policy |
| `GET /v1/transactions` | List persisted events; optional sender/recipient filters |
| `GET /v1/transactions/{id}` | Original score, inputs, policy and history snapshot |
| `GET /v1/cases` | Queue filters for status, assignee and synthetic source |
| `GET /v1/cases/{id}` | Case details and event timeline |
| `PATCH /v1/cases/{id}` | Assign, note, resolve or reopen |
| `GET /v1/feedback` | Confirmed external outcomes; synthetic records excluded by default |

Case action example:

```json
{
  "action": "RESOLVE",
  "actor": "Asha",
  "expected_version": 2,
  "decision": "LEGITIMATE",
  "reason": "Customer verified the transfer."
}
```

Actions are `ASSIGN`, `NOTE`, `RESOLVE`, `REOPEN`. Outcomes are `CONFIRMED_FRAUD`, `LEGITIMATE`, `INCONCLUSIVE`. Assignment requires an assignee; resolution and reopening require reasons. `expected_version` prevents stale overwrites; conflicts return 409. A repeated identical transaction ID/input returns its original snapshot; changed input with that ID returns 409. All case writes and their audit events commit together.

SQLite defaults to `data/investigations.sqlite3`; set `FRAUD_DB_PATH` to use another local database. Synthetic and external histories are isolated. Feedback exports label only currently resolved fraud/legitimate cases; inconclusive or reopened cases have no training label. Feedback does not automatically retrain models.

## History features and rules

On ingestion, history uses only **strictly earlier events already observed locally**. The current event, events at the same timestamp, future events and other source types are excluded. Late-arriving events do not rewrite previous snapshots. Histories are separated by payment rail. A missing entity ID disables its entity history rather than inventing an identity.

Recorded features include sender counts and amounts over five minutes and one hour, time since the previous event, recipient distinct-sender count and incoming amount, recent incoming funds for a sender, and beneficiary novelty within observed history.

Transparent, versioned rules route a case for review when there are:

- At least three prior sender transactions in five minutes.
- At least three distinct prior senders to the recipient in one hour.
- A cash-out within fifteen minutes of received funds.
- A transfer of at least 1,000 to an unseen beneficiary after two prior sender transactions in one hour.

These thresholds are illustrative, in the dataset's amount units. Rules **do not alter the model score** and are not described as calibrated probabilities. The generated scenario exercises them. Model-only holdout metrics do not establish the combined rule queue's precision. The uniformly sampled PaySim file cannot reconstruct complete account histories, so sparse sampled histories are not used as if they were complete training features.

## Verification and remaining scope

For complete-history queue replay, daily backlog and validation-selected staffing scenarios, see [the whole-workflow evaluation](WORKFLOW_EVALUATION.md). This supplements the earlier model-only holdout results. The locally applied rule selection removes extra pre-transfer rules without raising the model cutoff.

```powershell
python -m pytest
```

Tests cover original mathematical checks, score/explanation agreement, threshold boundaries, natural-prevalence splits, artifact round trips, causal history, rule routing, idempotence, persistent case lifecycle, stale-update conflicts, feedback export, and the Streamlit investigation workflow. `pytest.ini` limits discovery to `tests/`, avoiding historical simulation scripts that execute expensive experiments during import.

Deployment hardening and concurrent load testing are deferred. This local workflow does not add authentication or change the existing CORS configuration.
