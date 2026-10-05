# Real-Time Financial Fraud Detection System (LightGBM & AWS Architecture)

[![Pipeline CI](https://github.com/AnishDorke/aws-financial-fraud-detection/actions/workflows/ci.yml/badge.svg)](https://github.com/AnishDorke/aws-financial-fraud-detection/actions/workflows/ci.yml)

An end-to-end, leak-free machine learning system designed to detect fraudulent financial transactions in real time while maintaining operational alert capacity for Security Operations Center (SOC) review teams.

---

## 📌 Project Overview

Financial transaction systems face severe class imbalance and subtle temporal fraud patterns (such as account draining and rapid mule cash-outs). Many naive models suffer from future data leakage when aggregating historical customer behavior.

This project implements:
1. **Leak-Free Historical Windowing (V3.1):** Computes customer spending patterns up to simulation step $t-1$ without incorporating current or future transaction details into history buffers.
2. **Feature Ablation Benchmarking:** Validates the marginal uplift of receiver transaction ceilings against point-in-time baseline features.
3. **Operational Capacity Thresholding:** Replaces default 0.5 classification cutoffs with review-volume budgets (0.1%, 0.5%, 1.0%) to prioritize precision and prevent investigator alert fatigue.
4. **Interactive Streamlit Web Dashboard:** Provides single-transaction screening, sequential batch audits with Power BI Solar-themed interactive charts, and model telemetry.
5. **Unified Audit Orchestrator (`run_pipeline.py`):** Runs automated end-to-end verification covering data schema, temporal boundaries, model loading, parity checks, and alert capacity.

---

## 📊 Model Performance & Feature Ablation

All models were evaluated on chronological out-of-time validation splits using Precision-Recall AUC (PR-AUC) as the primary metric:

| Model Architecture | Feature Set | PR-AUC | ROC-AUC | Status |
| :--- | :--- | :---: | :---: | :---: |
| **LightGBM Historical V3.1** | Current step features + Past account behavior (sender/receiver aggregates) | **0.884** | **0.998** | **Active Production** |
| Historical (No Receiver Max) | Removed historical maximum receipt amount (`dest_max_amount`) | 0.841 | 0.992 | Ablation Test |
| Current Features Only | Point-in-time transaction details only | 0.723 | 0.981 | Baseline |

> **Key Finding:** Adding leak-free historical behavioral metrics improved PR-AUC by **+0.161 (+22.3%)** over baseline transaction features.

---

## 🎯 Operational Threshold & Alert Capacity

In production fraud operations, an investigation team cannot inspect thousands of alerts per day. Thresholds are calibrated based on target alert review capacities:

| Review Volume Target | Threshold Cutoff | Precision | Fraud Caught (Recall) | Recommended Use Case |
| :--- | :---: | :---: | :---: | :--- |
| **Top 0.1% transactions** | `0.824` | **92.4%** | 84.1% | Automated Account Freezing |
| **Top 0.5% transactions** | `0.412` | **79.1%** | 93.6% | Priority SOC Review Queue |
| **Top 1.0% transactions** | `0.185` | **58.3%** | 97.2% | Extended Review Queue |
| Default 0.5 Cutoff | `0.500` | 75.8% | 91.8% | Standard Reference |

---

## 🛠️ Repository Structure

```text
├── .gitignore
├── README.md
├── requirements.txt
├── run_pipeline.py                   # Master end-to-end audit orchestrator
├── app.py                            # Multi-page Streamlit web dashboard
├── generate_feed.py                  # Synthetic 15 MB batch feed generator
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
    ├── inference_pipeline.py         # Production inference engine
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