
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
MODEL_PATH = Path("models/experiments")
REPORT_PATH = Path("reports/experiments")

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

TARGET = "target"
CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def load_split(split, features):
    folder = BASE_PATH / split
    files = sorted(folder.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")

    columns = list(dict.fromkeys(features + [TARGET]))
    frames = [pd.read_parquet(file, columns=columns) for file in files]
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


def evaluate_model(name, features, train, validation):
    X_train = train[features]
    y_train = train[TARGET].astype("int8")

    X_val = validation[features]
    y_val = validation[TARGET].astype("int8")

    negatives = int((y_train == 0).sum())
    positives = int((y_train == 1).sum())

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=1500,
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
    print(f"Training rows: {len(train):,}")
    print(f"Training fraud: {positives:,}")
    print(f"Validation rows: {len(validation):,}")

    model.fit(
        X_train,
        y_train,
        categorical_feature=["type"],
        eval_set=[(X_val, y_val)],
        eval_metric="auc",
        callbacks=[
            lgb.early_stopping(50, first_metric_only=True),
            lgb.log_evaluation(50),
        ],
    )

    probabilities = model.predict_proba(
        X_val,
        num_iteration=model.best_iteration_,
    )[:, 1]

    predictions = {}
    summary = {
        "model": name,
        "train_rows": int(len(train)),
        "train_fraud": positives,
        "validation_rows": int(len(validation)),
        "validation_fraud": int(y_val.sum()),
        "validation_fraud_rate": float(y_val.mean()),
        "best_iteration": int(model.best_iteration_ or 0),
        "pr_auc": float(average_precision_score(y_val, probabilities)),
        "roc_auc": float(roc_auc_score(y_val, probabilities)),
    }

    threshold_rows = []

    for alert_rate in [0.001, 0.0025, 0.005, 0.01]:
        threshold = float(np.quantile(probabilities, 1 - alert_rate))
        predicted = (probabilities >= threshold).astype("int8")

        tn, fp, fn, tp = confusion_matrix(
            y_val,
            predicted,
            labels=[0, 1],
        ).ravel()

        row = {
            "model": name,
            "target_alert_rate": alert_rate,
            "threshold": threshold,
            "actual_alert_rate": float(predicted.mean()),
            "precision": float(precision_score(y_val, predicted, zero_division=0)),
            "recall": float(recall_score(y_val, predicted, zero_division=0)),
            "f1": float(f1_score(y_val, predicted, zero_division=0)),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        }
        threshold_rows.append(row)

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

    summary["feature_importance"] = importance.to_dict(orient="records")
    predictions[name] = probabilities

    print(f"\n{name} validation summary")
    print(f"PR-AUC:  {summary['pr_auc']:.6f}")
    print(f"ROC-AUC: {summary['roc_auc']:.6f}")
    print(f"Best iteration: {summary['best_iteration']}")

    return summary, threshold_rows, predictions


def main():
    print("Loading V3.1 training and validation data...")

    all_features = list(dict.fromkeys(
        HISTORICAL_FEATURES + [TARGET]
    ))

    train = load_split("train", all_features[:-1])
    validation = load_split("validation", all_features[:-1])

    print("\nChecking data integrity...")
    print(f"Train steps: {train['step'].min()} - {train['step'].max()}")
    print(
        f"Validation steps: {validation['step'].min()} - "
        f"{validation['step'].max()}"
    )

    if train["step"].max() >= validation["step"].min():
        raise ValueError("Training and validation time ranges overlap")

    if train[TARGET].sum() == 0 or validation[TARGET].sum() == 0:
        raise ValueError("Training and validation must both contain fraud cases")

    results = []
    threshold_results = []

    experiments = [
        ("lightgbm_current_only", CURRENT_FEATURES),
        ("lightgbm_historical_v3_1", HISTORICAL_FEATURES),
    ]

    for name, features in experiments:
        summary, thresholds, _ = evaluate_model(
            name,
            features,
            train,
            validation,
        )
        summary.pop("feature_importance", None)
        results.append(summary)
        threshold_results.extend(thresholds)

    pd.DataFrame(results).to_csv(
        REPORT_PATH / "model_comparison.csv",
        index=False,
    )

    pd.DataFrame(threshold_results).to_csv(
        REPORT_PATH / "validation_threshold_analysis.csv",
        index=False,
    )

    metadata = {
        "dataset": "PaySim synthetic dataset",
        "feature_version": "historical_v3_1",
        "test_evaluated": False,
        "thresholds_selected_for_production": False,
        "models": [item["model"] for item in results],
    }

    with open(REPORT_PATH / "experiment_metadata.json", "w") as file:
        json.dump(metadata, file, indent=2)

    print("\nExperiment complete.")
    print(f"Reports saved to: {REPORT_PATH}")
    print(f"Models saved to: {MODEL_PATH}")
    print("\nModel comparison:")
    print(pd.DataFrame(results)[
        ["model", "pr_auc", "roc_auc", "best_iteration"]
    ].to_string(index=False))

    print("\nValidation threshold analysis:")
    print(pd.DataFrame(threshold_results)[[
        "model",
        "target_alert_rate",
        "actual_alert_rate",
        "precision",
        "recall",
        "f1",
        "false_positives",
        "false_negatives",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()