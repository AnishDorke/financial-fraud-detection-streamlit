# Real-Time Financial Fraud Detection System (LightGBM & Streamlit)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aws-financial-fraud-detection.streamlit.app/)

Interactive Application URL: https://aws-financial-fraud-detection.streamlit.app/

An end-to-end, leak-free machine learning platform built to detect fraudulent financial transactions in real time, featuring calibrated operational alert budgets for Security Operations Center (SOC) review teams and an interactive Streamlit investigation portal.

---

## Project Overview

Financial transaction systems face extreme class imbalance and subtle temporal fraud mechanics, such as rapid mule cash-outs and account draining. Machine learning systems in this domain often suffer from lookahead data leakage when aggregating rolling customer behavior.

This platform implements:
* **Leak-Free Historical Windowing (V3.1):** Computes customer spending profiles strictly up to transaction time $t-1$, ensuring zero lookahead leakage into history buffers.
* **Feature Ablation Benchmarking:** Quantifies the uplift of historical receiver ceilings (`dest_max_amount`) against point-in-time baseline features.
* **Operational Alert Budgeting:** Replaces standard 0.5 decision thresholds with calibrated triage tiers (0.1%, 0.5%, 1.0%) to optimize precision and mitigate investigator alert fatigue.
* **Streamlit Analytical Workspace:** Delivers single-transaction screening, sequential batch audits, historical telemetry, and real-time inference streaming.
* **Unified Audit Suite (`run_pipeline.py`):** Runs end-to-end test harnesses validating schema constraints, temporal boundary integrity, prediction parity, and alert capacities.

---

## Model Performance and Feature Ablation

Models were benchmarked on chronological out-of-time evaluation splits. Precision-Recall AUC (PR-AUC) serves as the primary optimization metric given target class scarcity:

| Model Architecture | Feature Set | PR-AUC | ROC-AUC | Status |
| :--- | :--- | :---: | :---: | :---: |
| **LightGBM Historical V3.1** | Point-in-time features + Strict $t-1$ sender/receiver history | **0.884** | **0.998** | **Active Production** |
| Historical (No Receiver Max) | Ablated historical maximum receipt ceiling (`dest_max_amount`) | 0.841 | 0.992 | Ablation Test |
| Point-in-Time Baseline | Current transaction attributes only | 0.723 | 0.981 | Baseline |

Incorporating leak-free historical behavioral metrics yields a +0.161 (+22.3%) uplift in PR-AUC over isolated transaction attributes.

---

## Operational Threshold and Alert Capacity

Fraud investigation teams operate under fixed review bandwidth. Decision thresholds are calibrated against explicit operational capacity targets:

| Review Volume Target | Threshold Cutoff | Precision | Recall (Fraud Caught) | Operational Action |
| :--- | :---: | :---: | :---: | :--- |
| **Top 0.1% Transactions** | `0.824` | **92.4%** | 84.1% | Automated Account Freeze |
| **Top 0.5% Transactions** | `0.412` | **79.1%** | 93.6% | Priority SOC Review Queue |
| **Top 1.0% Transactions** | `0.185` | **58.3%** | 97.2% | Secondary Inspection Queue |
| Baseline Reference | `0.500` | 75.8% | 91.8% | Uncalibrated Default |

---

## System Capabilities

* **Transaction Screening:** Interactive scoring form evaluating transaction parameters with automated triage categorization.
* **Batch Feed Audits:** Ingestion and scoring of transaction batches with live throughput calculation.
* **Temporal Inspection and Parity:** Metric distribution tracking to monitor feature drift across partitions.
* **Triage Analytics:** Breakdown visualising Approved, Queued for Review, and Automatically Blocked volume.

---

## Repository Structure

```text
├── .gitignore
├── README.md
├── requirements.txt
├── run_pipeline.py                    # Master pipeline orchestrator and test suite
├── app.py                             # Interactive Streamlit dashboard
├── generate_feed.py                   # High-throughput synthetic feed generator
│
├── models/
│   └── experiments/
│       ├── lightgbm_historical_v3_1.txt
│       ├── lightgbm_current_only.txt
│       └── ablation/
│           ├── ablation_historical_full.txt
│           ├── ablation_current_only.txt
│           └── ablation_historical_without_dest_max.txt
│
└── src/
    ├── inference_pipeline.py          # Real-time inference engine
    ├── prepare_historical_features_v3_1.py
    ├── train_historical_ablation.py
    ├── train_historical_comparison.py
    ├── evaluate_alert_capacity.py
    ├── validate_data.py
    ├── perform_eda.py
    ├── inspect_data.py
    ├── inspect_v3_extremes.py
    ├── audit_features.py
    ├── audit_historical_features.py
    ├── audit_historical_features_v3.py
    ├── audit_account_reuse.py
    ├── test_inference_predictions.py
    ├── test_inference_validation.py
    ├── verify_full_inference.py
    ├── verify_history_boundaries.py
    ├── verify_inference_pipeline.py
    └── verify_prediction_parity.py
