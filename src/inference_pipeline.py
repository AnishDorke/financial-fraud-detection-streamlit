
from pathlib import Path
from collections import defaultdict

import lightgbm as lgb
import numpy as np
import pandas as pd


MODEL_PATH = Path(
    "models/experiments/ablation/ablation_historical_full.txt"
)

CATEGORIES = [
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
]

FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "log_amount",
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

REQUIRED_INPUT_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "oldbalanceDest",
    "nameDest",
]


class FraudInferencePipeline:
    def __init__(self, model_path=MODEL_PATH):
        self.model = lgb.Booster(model_file=str(model_path))
        self.model_features = self.model.feature_name()

        if self.model_features != FEATURE_COLUMNS:
            raise ValueError(
                "Model features do not match the expected feature contract."
            )

        self.destination_state = {}
        self.recent_by_step = {}

        self.rolling_5_count = defaultdict(int)
        self.rolling_5_amount = defaultdict(float)

        self.rolling_24_count = defaultdict(int)
        self.rolling_24_amount = defaultdict(float)

        self.last_processed_step = None

    def _build_features(self, batch, step):
        records = []
        previous_step_destinations = self.recent_by_step.get(
            step - 1, {}
        )

        for row in batch.itertuples(index=False):
            destination = str(row.nameDest)

            state = self.destination_state.get(destination)

            if state is None:
                previous_count = 0
                previous_sum = 0.0
                previous_max = 0.0
                previous_last_step = None
            else:
                (
                    previous_count,
                    previous_sum,
                    previous_max,
                    previous_last_step,
                ) = state

            time_since_last = (
                -1
                if previous_last_step is None
                else step - previous_last_step
            )

            previous_1 = previous_step_destinations.get(
                destination, (0, 0.0)
            )

            records.append({
                "step": int(row.step),
                "type": row.type,
                "amount": float(row.amount),
                "oldbalanceOrg": float(row.oldbalanceOrg),
                "oldbalanceDest": float(row.oldbalanceDest),
                "log_amount": float(np.log1p(row.amount)),
                "destination_txn_count_before": int(previous_count),
                "destination_amount_sum_before": float(previous_sum),
                "destination_amount_mean_before": float(
                    previous_sum / previous_count
                    if previous_count > 0 else 0.0
                ),
                "destination_amount_max_before": float(previous_max),
                "destination_time_since_last": int(time_since_last),
                "destination_txn_count_prev_1": int(previous_1[0]),
                "destination_txn_count_prev_5": int(
                    self.rolling_5_count.get(destination, 0)
                ),
                "destination_txn_count_prev_24": int(
                    self.rolling_24_count.get(destination, 0)
                ),
                "destination_amount_sum_prev_5": float(
                    self.rolling_5_amount.get(destination, 0.0)
                ),
                "destination_amount_sum_prev_24": float(
                    self.rolling_24_amount.get(destination, 0.0)
                ),
            })

        features = pd.DataFrame(records, columns=FEATURE_COLUMNS)

        features["type"] = pd.Categorical(
            features["type"],
            categories=CATEGORIES,
        )

        return features

    def _update_history(self, batch, step):
        grouped = batch.groupby("nameDest", sort=False).agg(
            current_count=("amount", "size"),
            current_sum=("amount", "sum"),
            current_max=("amount", "max"),
        )

        current_step_history = {}

        for destination, group in grouped.iterrows():
            destination = str(destination)

            current_count = int(group["current_count"])
            current_sum = float(group["current_sum"])
            current_max = float(group["current_max"])

            current_step_history[destination] = (
                current_count,
                current_sum,
            )

            state = self.destination_state.get(destination)

            if state is None:
                previous_count = 0
                previous_sum = 0.0
                previous_max = 0.0
            else:
                (
                    previous_count,
                    previous_sum,
                    previous_max,
                    _,
                ) = state

            self.destination_state[destination] = (
                previous_count + current_count,
                previous_sum + current_sum,
                max(previous_max, current_max),
                step,
            )

            self.rolling_5_count[destination] += current_count
            self.rolling_5_amount[destination] += current_sum

            self.rolling_24_count[destination] += current_count
            self.rolling_24_amount[destination] += current_sum

        self.recent_by_step[step] = current_step_history

        remove_5_step = step - 5

        if remove_5_step in self.recent_by_step:
            old_step = self.recent_by_step[remove_5_step]

            for destination, values in old_step.items():
                old_count, old_amount = values

                self.rolling_5_count[destination] -= old_count
                self.rolling_5_amount[destination] -= old_amount

                if self.rolling_5_count[destination] <= 0:
                    self.rolling_5_count.pop(destination, None)

                if self.rolling_5_amount[destination] <= 0:
                    self.rolling_5_amount.pop(destination, None)

        remove_24_step = step - 24

        if remove_24_step in self.recent_by_step:
            old_step = self.recent_by_step[remove_24_step]

            for destination, values in old_step.items():
                old_count, old_amount = values

                self.rolling_24_count[destination] -= old_count
                self.rolling_24_amount[destination] -= old_amount

                if self.rolling_24_count[destination] <= 0:
                    self.rolling_24_count.pop(destination, None)

                if self.rolling_24_amount[destination] <= 0:
                    self.rolling_24_amount.pop(destination, None)

            del self.recent_by_step[remove_24_step]

    def predict_step(self, transactions):
        missing = [
            column
            for column in REQUIRED_INPUT_COLUMNS
            if column not in transactions.columns
        ]

        if missing:
            raise ValueError(f"Missing input columns: {missing}")

        if transactions.empty:
            raise ValueError("Transaction batch cannot be empty.")

        batch = transactions[REQUIRED_INPUT_COLUMNS].copy()

        if batch.isna().any().any():
            raise ValueError("Input transactions contain missing values.")

        if not batch["step"].nunique() == 1:
            raise ValueError(
                "Each batch must contain transactions from exactly one step."
            )

        step_values = batch["step"].unique()

        if len(step_values) != 1:
            raise ValueError("Invalid step values.")

        step_value = step_values[0]

        if not np.isfinite(float(step_value)) or float(step_value) < 1:
            raise ValueError("Step must be a positive finite number.")

        if float(step_value) != int(step_value):
            raise ValueError("Step must be an integer.")

        step = int(step_value)

        if self.last_processed_step is not None:
            if step != self.last_processed_step + 1:
                raise ValueError(
                    "Steps must be processed consecutively and in order. "
                    f"Expected {self.last_processed_step + 1}, received {step}."
                )

        if not pd.api.types.is_numeric_dtype(batch["amount"]):
            raise ValueError("Amount must be numeric.")

        numeric_columns = [
            "amount",
            "oldbalanceOrg",
            "oldbalanceDest",
        ]

        for column in numeric_columns:
            batch[column] = pd.to_numeric(
                batch[column], errors="raise"
            )

            if not np.isfinite(batch[column]).all():
                raise ValueError(f"{column} contains non-finite values.")

        if (batch["amount"] < 0).any():
            raise ValueError("Amount cannot be negative.")

        if (~batch["type"].isin(CATEGORIES)).any():
            raise ValueError("Unknown transaction type detected.")

        if batch["nameDest"].astype(str).str.len().eq(0).any():
            raise ValueError("Destination account cannot be empty.")

        features = self._build_features(batch, step)

        probabilities = self.model.predict(features)

        results = batch.copy()
        results["fraud_score"] = probabilities

        self._update_history(batch, step)
        self.last_processed_step = step

        return results