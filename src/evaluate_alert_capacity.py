
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)

BASE_PATH = Path("data/processed_historical_v3_1")
MODEL_PATH = Path("models/experiments")
REPORT_PATH = Path("reports/experiments")

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

CATEGORIES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]


def load_validation():
    files = sorted((BASE_PATH / "validation").glob("*.parquet"))
    columns = list(dict.fromkeys(HISTORICAL_FEATURES + ["target"]))

    frames = [
        pd.read_parquet(file, columns=columns)
        for file in files
    ]
    data = pd.concat(frames, ignore_index=True)
    data["type"] = pd.Categorical(data["type"], categories=CATEGORIES)

    return data


def evaluate_capacity(name, features, data):
    model_file = MODEL_PATH / f"{name}.txt"

    if not model_file.exists():
        raise FileNotFoundError(f"Model not found: {model_file}")

    model = lgb.Booster(model_file=str(model_file))
    X = data[features]
    y = data["target"].astype("int8").to_numpy()

    scores = model.predict(X)
    rows = []

    print(f"\n{name}")
    print(f"PR-AUC: {average_precision_score(y, scores):.6f}")
    print(f"ROC-AUC: {roc_auc_score(y, scores):.6f}")

    for rate in [0.001, 0.0025, 0.005, 0.01]:
        k = max(1, int(np.ceil(len(scores) * rate)))

        selected = np.argpartition(scores, len(scores) - k)[-k:]
        predictions = np.zeros(len(scores), dtype="int8")
        predictions[selected] = 1

        tp = int(((predictions == 1) & (y == 1)).sum())
        fp = int(((predictions == 1) & (y == 0)).sum())
        fn = int(((predictions == 0) & (y == 1)).sum())
        tn = int(((predictions == 0) & (y == 0)).sum())

        row = {
            "model": name,
            "requested_alert_rate": rate,
            "alert_count": int(k),
            "actual_alert_rate": float(predictions.mean()),
            "score_cutoff": float(scores[selected].min()),
            "precision": float(precision_score(y, predictions, zero_division=0)),
            "recall": float(recall_score(y, predictions, zero_division=0)),
            "f1": float(f1_score(y, predictions, zero_division=0)),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
        }
        rows.append(row)

    result = pd.DataFrame(rows)

    print(result[[
        "requested_alert_rate",
        "alert_count",
        "actual_alert_rate",
        "precision",
        "recall",
        "f1",
        "true_positives",
        "false_positives",
        "false_negatives",
    ]].to_string(index=False))

    return rows


def main():
    print("Loading validation data only...")
    validation = load_validation()
    print(f"Rows: {len(validation):,}")
    print(f"Fraud cases: {int(validation['target'].sum()):,}")

    results = []

    experiments = [
        ("lightgbm_current_only", CURRENT_FEATURES),
        ("lightgbm_historical_v3_1", HISTORICAL_FEATURES),
    ]

    for name, features in experiments:
        results.extend(evaluate_capacity(name, features, validation))

    REPORT_PATH.mkdir(parents=True, exist_ok=True)
    output = REPORT_PATH / "validation_alert_capacity.csv"
    pd.DataFrame(results).to_csv(output, index=False)

    print(f"\nSaved: {output}")
    print("No test data was loaded or evaluated.")


if __name__ == "__main__":
    main()