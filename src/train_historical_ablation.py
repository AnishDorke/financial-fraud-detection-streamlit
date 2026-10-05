
from pathlib import Path
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

BASE_PATH = Path("data/processed_historical_v3_1")
MODEL_PATH = Path("models/experiments/ablation")
REPORT_PATH = Path("reports/experiments/ablation")

MODEL_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH.mkdir(parents=True, exist_ok=True)

CURRENT_FEATURES = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
]

HISTORICAL_FEATURES = CURRENT_FEATURES + [
    "destination_txn_count_before",
    "destination_amount_sum_before",
    "destination_amount_mean_before",
    "destination_amount_max_before",
    "destination_time_since_last",
    "destination_txn_count_prev_1",
    "destination_txn_count_prev_5",
    "destination_txn_count_prev_24",
    "destination_amount_sum_prev_5",
    "destination_amount_sum_prev_24",
]

ABLATION_FEATURES = [
    feature
    for feature in HISTORICAL_FEATURES
    if feature != "destination_amount_max_before"
]

TARGET = "target"
CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def load_split(split, features):
    folder = BASE_PATH / split
    files = sorted(folder.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")

    columns = list(dict.fromkeys(features + [TARGET]))
    frames = [
        pd.read_parquet(file, columns=columns)
        for file in files
    ]
    data = pd.concat(frames, ignore_index=True)

    data["type"] = pd.Categorical(
        data["type"],
        categories=CATEGORIES,
    )

    if data[TARGET].isna().any():
        raise ValueError(f"Missing target values in {split}")

    if not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError(f"Unexpected target values in {split}")

    return data


def pr_auc_eval(y_true, y_pred):
    score = average_precision_score(y_true, y_pred)
    return "validation_pr_auc", score, True


def evaluate_alert_capacity(name, y_true, probabilities):
    rows = []
    total_rows = len(y_true)

    for alert_rate in [0.001, 0.0025, 0.005, 0.01]:
        alert_count = max(1, int(np.ceil(total_rows * alert_rate)))
        alert_indices = np.argpartition(
            probabilities,
            total_rows - alert_count,
        )[total_rows - alert_count:]

        predicted = np.zeros(total_rows, dtype=np.int8)
        predicted[alert_indices] = 1

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predicted,
            labels=[0, 1],
        ).ravel()

        rows.append({
            "model": name,
            "target_alert_rate": alert_rate,
            "alert_count": int(alert_count),
            "actual_alert_rate": float(alert_count / total_rows),
            "precision": float(
                precision_score(y_true, predicted, zero_division=0)
            ),
            "recall": float(
                recall_score(y_true, predicted, zero_division=0)
            ),
            "f1": float(
                f1_score(y_true, predicted, zero_division=0)
            ),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        })

    return rows


def train_experiment(name, features, train, validation):
    X_train = train[features]
    y_train = train[TARGET].astype("int8")
    X_val = validation[features]
    y_val = validation[TARGET].astype("int8")

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)

    model = lgb.LGBMClassifier(
        objective="binary",
        metric="None",
        n_estimators=3000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
        colsample_bytree=0.9,
        subsample=0.8,
        subsample_freq=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=negatives / positives,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    print(f"\nTraining {name}")
    print(f"Features: {len(features)}")
    print(f"Training rows: {len(train):,}")
    print(f"Training fraud: {positives:,}")
    print(f"Validation rows: {len(validation):,}")
    print(f"Validation fraud: {int(y_val.sum()):,}")

    model.fit(
        X_train,
        y_train,
        categorical_feature=["type"],
        eval_set=[(X_val, y_val)],
        eval_metric=pr_auc_eval,
        callbacks=[
            lgb.early_stopping(50, first_metric_only=True),
            lgb.log_evaluation(25),
        ],
    )

    probabilities = model.predict_proba(
        X_val,
        num_iteration=model.best_iteration_,
    )[:, 1]

    summary = {
        "model": name,
        "feature_count": len(features),
        "train_rows": int(len(train)),
        "train_fraud": positives,
        "validation_rows": int(len(validation)),
        "validation_fraud": int(y_val.sum()),
        "validation_fraud_rate": float(y_val.mean()),
        "best_iteration": int(model.best_iteration_ or 0),
        "pr_auc": float(
            average_precision_score(y_val, probabilities)
        ),
        "roc_auc": float(
            roc_auc_score(y_val, probabilities)
        ),
    }

    model.booster_.save_model(
        str(MODEL_PATH / f"{name}.txt"),
        num_iteration=model.best_iteration_,
    )

    importance = pd.DataFrame({
        "feature": features,
        "gain_importance": model.booster_.feature_importance(
            importance_type="gain"
        ),
        "split_importance": model.booster_.feature_importance(
            importance_type="split"
        ),
    }).sort_values("gain_importance", ascending=False)

    importance.to_csv(
        REPORT_PATH / f"{name}_feature_importance.csv",
        index=False,
    )

    alert_rows = evaluate_alert_capacity(
        name,
        y_val.to_numpy(),
        probabilities,
    )

    print(f"\n{name} results")
    print(f"PR-AUC: {summary['pr_auc']:.6f}")
    print(f"ROC-AUC: {summary['roc_auc']:.6f}")
    print(f"Best iteration: {summary['best_iteration']}")

    return summary, alert_rows


def main():
    print("Loading V3.1 training and validation data...")
    print("The test split will not be loaded or evaluated.")

    train = load_split("train", HISTORICAL_FEATURES)
    validation = load_split("validation", HISTORICAL_FEATURES)

    if train["step"].max() >= validation["step"].min():
        raise ValueError("Training and validation time ranges overlap")

    if train[TARGET].sum() == 0 or validation[TARGET].sum() == 0:
        raise ValueError(
            "Training and validation must both contain fraud cases"
        )

    experiments = [
        ("ablation_current_only", CURRENT_FEATURES),
        ("ablation_historical_full", HISTORICAL_FEATURES),
        ("ablation_historical_without_dest_max", ABLATION_FEATURES),
    ]

    summaries = []
    all_alert_rows = []

    for name, features in experiments:
        summary, alert_rows = train_experiment(
            name,
            features,
            train,
            validation,
        )
        summaries.append(summary)
        all_alert_rows.extend(alert_rows)

    summary_df = pd.DataFrame(summaries)
    alert_df = pd.DataFrame(all_alert_rows)

    summary_df.to_csv(
        REPORT_PATH / "ablation_model_comparison.csv",
        index=False,
    )

    alert_df.to_csv(
        REPORT_PATH / "ablation_alert_capacity.csv",
        index=False,
    )

    metadata = {
        "dataset": "PaySim synthetic dataset",
        "feature_version": "historical_v3_1",
        "selection_metric": "validation PR-AUC",
        "test_evaluated": False,
        "thresholds_selected_for_production": False,
        "experiments": [
            {
                "model": name,
                "features": features,
            }
            for name, features in experiments
        ],
    }

    with open(
        REPORT_PATH / "ablation_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)

    print("\nExperiment complete.")
    print(f"Models saved to: {MODEL_PATH}")
    print(f"Reports saved to: {REPORT_PATH}")

    print("\nModel comparison:")
    print(
        summary_df[
            ["model", "pr_auc", "roc_auc", "best_iteration"]
        ].to_string(index=False)
    )

    print("\nAlert capacity comparison:")
    print(
        alert_df[
            [
                "model",
                "target_alert_rate",
                "true_positives",
                "false_positives",
                "precision",
                "recall",
                "f1",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()