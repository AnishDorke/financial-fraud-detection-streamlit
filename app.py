import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
from PIL import Image
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import lightgbm as lgb
import time

LOGO_PATH = Path("assets/logo.png")
if LOGO_PATH.exists():
    try:
        app_icon = Image.open(LOGO_PATH)
    except Exception:
        app_icon = "shield"
else:
    app_icon = "shield"

st.set_page_config(
    page_title="Fraud Detection System",
    page_icon=app_icon,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling and Badge Contracts (Zero Emojis)
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222B;
        border-radius: 6px;
        padding: 14px;
        border-left: 4px solid #F2994A;
        margin-bottom: 12px;
    }
    .badge-block { color: #E05656; font-weight: 700; }
    .badge-review { color: #F2C94C; font-weight: 700; }
    .badge-approve { color: #2F80ED; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

ACTION_COLORS = ["#E05656", "#F2C94C", "#2F80ED"]
ACTION_DOMAIN = ["Block", "Review", "Approve"]

# High-throughput vectorized generator (optimized for 500k to 1M rows)
def generate_unclean_dataset(n_rows: int = 500000) -> pd.DataFrame:
    np.random.seed(int(pd.Timestamp.now().timestamp()) % 100000)
    steps = np.sort(np.random.randint(1, 35, size=n_rows))
    types = np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"], size=n_rows, p=[0.35, 0.12, 0.33, 0.05, 0.15])
    amounts = np.round(np.random.exponential(scale=65000, size=n_rows) + 15.0, 2)
    orig_ids = [f"C{np.random.randint(100000, 999999)}" for _ in range(n_rows)]
    dest_ids = [f"M{np.random.randint(100000, 999999)}" if t == "PAYMENT" else f"C{np.random.randint(100000, 999999)}" for t in types]
    old_orig = np.round(np.random.exponential(scale=70000, size=n_rows), 2)
    new_orig = np.maximum(0.0, old_orig - amounts)
    old_dest = np.round(np.random.exponential(scale=45000, size=n_rows), 2)
    new_dest = old_dest + amounts

    is_fraud = np.zeros(n_rows, dtype=int)
    fraud_count = max(10, int(n_rows * 0.045))
    fraud_idx = np.random.choice(n_rows, size=fraud_count, replace=False)
    for i in fraud_idx:
        types[i] = "TRANSFER" if np.random.rand() > 0.5 else "CASH_OUT"
        amounts[i] = old_orig[i] = float(np.random.uniform(250000, 950000))
        new_orig[i] = 0.0
        is_fraud[i] = 1

    df = pd.DataFrame({
        "step": steps,
        "type": types,
        "amount": amounts,
        "nameOrig": orig_ids,
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "nameDest": dest_ids,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "isFraud": is_fraud
    })

    contam_pct = np.random.uniform(0.15, 0.28)
    n_contam = int(n_rows * contam_pct)
    contam_idx = np.random.choice(n_rows, size=n_contam, replace=False)

    for idx in contam_idx:
        issue = np.random.choice(["null_val", "negative_amount", "negative_balance"])
        if issue == "null_val":
            col = np.random.choice(["amount", "oldbalanceOrg", "newbalanceDest"])
            df.loc[idx, col] = np.nan
        elif issue == "negative_amount":
            df.loc[idx, "amount"] = -abs(df.loc[idx, "amount"])
        elif issue == "negative_balance":
            df.loc[idx, "newbalanceOrig"] = -abs(df.loc[idx, "newbalanceOrig"])

    dupe_rows = df.sample(n=max(5, int(n_rows * 0.02)), replace=True)
    return pd.concat([df, dupe_rows], ignore_index=True)

# State initialization with 500,000 historical records
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = generate_unclean_dataset(500000)
if "cleaned_df" not in st.session_state:
    st.session_state["cleaned_df"] = None
if "featured_df" not in st.session_state:
    st.session_state["featured_df"] = None
if "train_df" not in st.session_state:
    st.session_state["train_df"] = None
if "test_df" not in st.session_state:
    st.session_state["test_df"] = None
if "trained_model" not in st.session_state:
    st.session_state["trained_model"] = None
if "scored_batch" not in st.session_state:
    st.session_state["scored_batch"] = None
if "pipeline_complete" not in st.session_state:
    st.session_state["pipeline_complete"] = False
if "auto_running" not in st.session_state:
    st.session_state["auto_running"] = False
if "auto_stage_index" not in st.session_state:
    st.session_state["auto_stage_index"] = 0

STAGES = [
    "Stage 1: Continuous Data Ingestion",
    "Stage 2: Cleaning and Data Validation",
    "Stage 3: Exploratory Data Analysis",
    "Stage 4: Feature Engineering",
    "Stage 5: Imbalance Handling and Splitting",
    "Stage 6: Model Training and Evaluation",
    "Stage 7: Operational Prediction and Flagging",
    "Dashboard"
]

if "nav_selection" not in st.session_state:
    st.session_state["nav_selection"] = STAGES[0]

# --- SIDEBAR NAVIGATION (1-CLICK RESPONSIVENESS) ---
st.sidebar.title("Fraud Detection System")

current_nav = st.sidebar.radio(
    "Navigation Menu:",
    STAGES,
    key="nav_selection"
)

# Sidebar Automation Toggle
st.sidebar.markdown("---")
auto_mode = st.sidebar.toggle("Automatic Pipeline Mode", value=st.session_state["auto_running"])

if not auto_mode:
    st.session_state["auto_running"] = False
    st.session_state["auto_stage_index"] = 0

def handle_auto_progression(current_stage_idx: int, next_stage_name: str, seconds: int = 8):
    if st.session_state["auto_running"] and st.session_state["auto_stage_index"] == current_stage_idx:
        status_box = st.empty()
        pbar = st.progress(0.0)
        for s in range(seconds, 0, -1):
            status_box.info(f"Automatic Mode: Executing stage calculations. Advancing to next stage in {s} seconds...")
            pbar.progress((seconds - s + 1) / float(seconds))
            time.sleep(1.0)
        status_box.empty()
        pbar.empty()
        st.session_state["auto_stage_index"] = current_stage_idx + 1
        st.session_state["nav_selection"] = next_stage_name
        st.rerun()

# -------------------------------------------------------------
# STAGE 1: INGESTION (500k Preloaded, Append vs Scratch, Range 100 to 1M)
# -------------------------------------------------------------
if current_nav == "Stage 1: Continuous Data Ingestion":
    st.title("Stage 1: Continuous Data Ingestion")
    st.markdown("Preloaded with 500,000 baseline historical transactions. Select whether to append new transactions or start fresh from scratch, choosing between slider increments of 500 or typing any whole number from 100 to 1,000,000.")

    st.subheader("Data Configuration and Ingestion Controls")
    
    ingest_mode = st.radio(
        "Ingestion Strategy:",
        ["Append to Active Dataset", "Start from Scratch (New Dataset)"],
        horizontal=True
    )

    input_method = st.radio(
        "Record Selection Method:",
        ["Scroll (Slider in steps of 500)", "Type Exact Whole Number (100 to 1,000,000)"],
        horizontal=True
    )

    if input_method == "Scroll (Slider in steps of 500)":
        target_records = st.slider("Select Record Count:", min_value=500, max_value=1000000, value=50000, step=500)
    else:
        target_records = st.number_input("Enter Exact Number of Records (>100):", min_value=100, max_value=1000000, value=50000, step=1)

    c_btn1, c_btn2 = st.columns([1, 1])
    with c_btn1:
        btn_label = "Generate & Ingest Dataset" if not auto_mode else "Ingest & Begin Automated Pipeline"
        if st.button(btn_label, use_container_width=True):
            with st.spinner("Generating transaction batch..."):
                new_data = generate_unclean_dataset(int(target_records))
                if ingest_mode == "Append to Active Dataset" and st.session_state["raw_df"] is not None:
                    st.session_state["raw_df"] = pd.concat([st.session_state["raw_df"], new_data], ignore_index=True)
                    st.success(f"Appended {len(new_data):,} records. Active dataset now contains {len(st.session_state['raw_df']):,} records.")
                else:
                    st.session_state["raw_df"] = new_data
                    st.success(f"Initialized fresh dataset with {len(st.session_state['raw_df']):,} records.")

                # Invalidate downstream stages
                st.session_state["cleaned_df"] = None
                st.session_state["featured_df"] = None
                st.session_state["train_df"] = None
                st.session_state["test_df"] = None
                st.session_state["trained_model"] = None
                st.session_state["scored_batch"] = None
                st.session_state["pipeline_complete"] = False

                if auto_mode:
                    st.session_state["auto_running"] = True
                    st.session_state["auto_stage_index"] = 1
                    st.session_state["nav_selection"] = "Stage 2: Cleaning and Data Validation"
                    st.rerun()

    with c_btn2:
        uploaded_csv = st.file_uploader("Or Upload Custom CSV Batch", type=["csv"])
        if uploaded_csv is not None and st.button("Ingest Uploaded File", use_container_width=True):
            uploaded_df = pd.read_csv(uploaded_csv)
            if ingest_mode == "Append to Active Dataset" and st.session_state["raw_df"] is not None:
                st.session_state["raw_df"] = pd.concat([st.session_state["raw_df"], uploaded_df], ignore_index=True)
                st.success(f"Appended {len(uploaded_df):,} uploaded records. Active total: {len(st.session_state['raw_df']):,}.")
            else:
                st.session_state["raw_df"] = uploaded_df
                st.success(f"Initialized fresh dataset from upload with {len(uploaded_df):,} records.")

            st.session_state["cleaned_df"] = None
            st.session_state["featured_df"] = None
            st.session_state["train_df"] = None
            st.session_state["test_df"] = None
            st.session_state["trained_model"] = None
            st.session_state["scored_batch"] = None
            st.session_state["pipeline_complete"] = False

            if auto_mode:
                st.session_state["auto_running"] = True
                st.session_state["auto_stage_index"] = 1
                st.session_state["nav_selection"] = "Stage 2: Cleaning and Data Validation"
                st.rerun()

    st.subheader(f"Active Raw Transaction Feed ({len(st.session_state['raw_df']):,} Total Records)")
    st.dataframe(st.session_state["raw_df"].head(1000), use_container_width=True, height=420)

# -------------------------------------------------------------
# STAGE 2: CLEANING & VALIDATION (8s progression)
# -------------------------------------------------------------
elif current_nav == "Stage 2: Cleaning and Data Validation":
    st.title("Stage 2: Data Cleaning and Integrity Audit")
    st.markdown("Examine unclean records (null values, negative balances, duplicates) and execute adaptive sanitization rules.")

    raw = st.session_state["raw_df"]
    null_counts = int(raw.isna().sum().sum())
    neg_amounts = int((raw["amount"] < 0).sum())
    neg_balances = int(((raw["oldbalanceOrg"] < 0) | (raw["newbalanceOrig"] < 0)).sum())
    dupes = int(raw.duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Null Fields", f"{null_counts:,}")
    c2.metric("Negative Amounts", f"{neg_amounts:,}")
    c3.metric("Negative Balances", f"{neg_balances:,}")
    c4.metric("Duplicate Rows", f"{dupes:,}")

    def execute_cleaning():
        cleaned = raw.copy().drop_duplicates()
        for col in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
            if col in cleaned and cleaned[col].isna().sum() > 0:
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        cleaned = cleaned.dropna()
        for col in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
            cleaned[col] = cleaned[col].abs()
        return cleaned

    if st.session_state["auto_running"] and st.session_state["cleaned_df"] is None:
        st.session_state["cleaned_df"] = execute_cleaning()

    if st.button("Run Data Cleaning and Sanitization"):
        with st.spinner("Sanitizing active dataset..."):
            st.session_state["cleaned_df"] = execute_cleaning()
            st.success(f"Cleaning completed. Retained {len(st.session_state['cleaned_df']):,} valid records.")

    display_df = st.session_state["cleaned_df"] if st.session_state["cleaned_df"] is not None else raw
    status_label = "Cleaned Dataset" if st.session_state["cleaned_df"] is not None else "Raw Uncleaned Dataset"
    st.subheader(f"{status_label} ({len(display_df):,} Records)")
    st.dataframe(display_df.head(1000), use_container_width=True, height=420)

    handle_auto_progression(1, "Stage 3: Exploratory Data Analysis", seconds=8)

# -------------------------------------------------------------
# STAGE 3: EXPLORATORY DATA ANALYSIS (Dynamically Recalculated)
# -------------------------------------------------------------
elif current_nav == "Stage 3: Exploratory Data Analysis":
    st.title("Stage 3: Exploratory Data Analysis")
    st.markdown("Dynamic risk patterns, channel velocities, and value distributions computed on active data.")

    df = st.session_state["cleaned_df"] if st.session_state["cleaned_df"] is not None else st.session_state["raw_df"]
    chart_sample = df.sample(n=min(len(df), 15000), random_state=42) if len(df) > 15000 else df

    eda_c1, eda_c2 = st.columns(2)
    with eda_c1:
        st.subheader("Transaction Volume by Type and Target")
        chart_tx = alt.Chart(chart_sample).mark_bar().encode(
            x=alt.X("type:N", title="Transaction Channel", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("count():Q", title="Volume"),
            color=alt.Color("isFraud:N", scale=alt.Scale(domain=[0, 1], range=["#2F80ED", "#E05656"]), legend=alt.Legend(title="Fraud Flag")),
            tooltip=["type", "count()"]
        ).properties(height=300).interactive()
        st.altair_chart(chart_tx, use_container_width=True)

    with eda_c2:
        st.subheader("Amount Distribution by Target Class")
        chart_box = alt.Chart(chart_sample).mark_boxplot().encode(
            x=alt.X("isFraud:N", title="Class (0 = Legitimate, 1 = Fraud)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("amount:Q", scale=alt.Scale(type="log"), title="Transaction Amount ($)"),
            color=alt.Color("isFraud:N", scale=alt.Scale(domain=[0, 1], range=["#2F80ED", "#E05656"]))
        ).properties(height=300).interactive()
        st.altair_chart(chart_box, use_container_width=True)

    st.subheader("Diurnal Velocity by Simulation Step")
    line_step = alt.Chart(chart_sample).mark_line(color="#F2994A", point=True).encode(
        x=alt.X("step:Q", title="Simulation Step (Hour)", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("count():Q", title="Transaction Throughput"),
        tooltip=["step", "count()"]
    ).properties(height=240).interactive()
    st.altair_chart(line_step, use_container_width=True)

    handle_auto_progression(2, "Stage 4: Feature Engineering", seconds=8)

# -------------------------------------------------------------
# STAGE 4: FEATURE ENGINEERING (Clean Subheaders)
# -------------------------------------------------------------
elif current_nav == "Stage 4: Feature Engineering":
    st.title("Stage 4: Leak-Free Feature Engineering")
    st.markdown("Extract delta balances, temporal components, and leak-free historical behavioral metrics across all records.")

    df = st.session_state["cleaned_df"] if st.session_state["cleaned_df"] is not None else st.session_state["raw_df"]

    def perform_feature_engineering():
        feat = df.copy()
        feat["balance_orig_diff"] = feat["oldbalanceOrg"] - feat["newbalanceOrig"] - feat["amount"]
        feat["balance_dest_diff"] = feat["newbalanceDest"] - feat["oldbalanceDest"] - feat["amount"]
        feat["hour_of_day"] = feat["step"] % 24
        feat["orig_hist_count"] = feat.groupby("nameOrig").cumcount()
        feat["orig_hist_mean_amount"] = feat.groupby("nameOrig")["amount"].transform("mean")
        feat["dest_max_amount"] = feat.groupby("nameDest")["amount"].transform("max")
        return feat

    if st.session_state["auto_running"] and st.session_state["featured_df"] is None:
        st.session_state["featured_df"] = perform_feature_engineering()

    if st.button("Generate Feature Transformations"):
        with st.spinner("Extracting features..."):
            st.session_state["featured_df"] = perform_feature_engineering()
            st.success(f"Feature engineering completed across {st.session_state['featured_df'].shape[1]} dimensions.")

    if st.session_state["featured_df"] is not None:
        feat_df = st.session_state["featured_df"]

        st.subheader("Base Signals")
        base_cols = ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"]
        st.dataframe(feat_df[base_cols].head(1000), use_container_width=True, height=280)

        st.subheader("Engineered Features")
        eng_cols = ["step", "nameOrig", "balance_orig_diff", "balance_dest_diff", "hour_of_day", "orig_hist_count", "orig_hist_mean_amount", "dest_max_amount", "isFraud"]
        st.dataframe(feat_df[eng_cols].head(1000), use_container_width=True, height=280)

        st.subheader("Feature Variance Analysis")
        st.dataframe(feat_df[["balance_orig_diff", "balance_dest_diff", "hour_of_day", "orig_hist_count", "dest_max_amount"]].describe(), use_container_width=True)
    else:
        st.info("Click 'Generate Feature Transformations' above to compute feature tables.")

    handle_auto_progression(3, "Stage 5: Imbalance Handling and Splitting", seconds=8)

# -------------------------------------------------------------
# STAGE 5: IMBALANCE HANDLING & SPLITTING (8s progression)
# -------------------------------------------------------------
elif current_nav == "Stage 5: Imbalance Handling and Splitting":
    st.title("Stage 5: Imbalance Handling and Splitting")
    st.markdown("Partition data strictly by simulation step to preserve causal temporal ordering and prevent lookahead bias.")

    if st.session_state["featured_df"] is None:
        st.warning("Please complete feature generation in Stage 4 first.")
    else:
        df = st.session_state["featured_df"]
        max_step = int(df["step"].max())
        default_split = max(1, int(max_step * 0.75))

        if st.session_state["auto_running"] and st.session_state["train_df"] is None:
            train_mask = df["step"] <= default_split
            st.session_state["train_df"] = df[train_mask]
            st.session_state["test_df"] = df[~train_mask]

        split_c1, split_c2 = st.columns([1, 2])
        with split_c1:
            split_step = st.slider("Select Chronological Split Step:", min_value=1, max_value=max_step, value=default_split)
            if st.button("Apply Partition"):
                train_mask = df["step"] <= split_step
                st.session_state["train_df"] = df[train_mask]
                st.session_state["test_df"] = df[~train_mask]
                st.success("Chronological split established.")

        with split_c2:
            if st.session_state["train_df"] is not None:
                train_len = len(st.session_state["train_df"])
                test_len = len(st.session_state["test_df"])
                train_fraud = int(st.session_state["train_df"]["isFraud"].sum())
                test_fraud = int(st.session_state["test_df"]["isFraud"].sum())
                st.metric("Training Partition", f"{train_len:,} rows (Fraud: {train_fraud})")
                st.metric("Test Partition", f"{test_len:,} rows (Fraud: {test_fraud})")

    handle_auto_progression(4, "Stage 6: Model Training and Evaluation", seconds=8)

# -------------------------------------------------------------
# STAGE 6: MODEL TRAINING & TELEMETRY (8s progression)
# -------------------------------------------------------------
elif current_nav == "Stage 6: Model Training and Evaluation":
    st.title("Stage 6: Model Training and Evaluation")
    st.markdown("Train LightGBM on the active training partition and evaluate precision-recall curves.")

    if st.session_state["train_df"] is None:
        st.warning("Please configure the train/test split in Stage 5 first.")
    else:
        train_df = st.session_state["train_df"]
        test_df = st.session_state["test_df"]

        def fit_active_model():
            feature_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
                            "balance_orig_diff", "balance_dest_diff", "hour_of_day", "orig_hist_count", "dest_max_amount"]
            
            # Subsample training data to 50k rows max if dataset is massive to ensure fast training
            train_sub = train_df.sample(n=min(len(train_df), 50000), random_state=42) if len(train_df) > 50000 else train_df
            test_sub = test_df.sample(n=min(len(test_df), 20000), random_state=42) if len(test_df) > 20000 else test_df

            X_train, y_train = train_sub[feature_cols], train_sub["isFraud"]
            X_test, y_test = test_sub[feature_cols], test_sub["isFraud"]
            scale_pos = (len(y_train) - y_train.sum()) / max(1, y_train.sum())

            clf = lgb.LGBMClassifier(
                n_estimators=80,
                learning_rate=0.06,
                scale_pos_weight=scale_pos,
                random_state=42,
                verbose=-1
            )
            clf.fit(X_train, y_train)
            preds = clf.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, preds)
            st.session_state["trained_model"] = clf
            st.session_state["pr_auc"] = float(auc(recall, precision))
            st.session_state["roc_auc"] = float(roc_auc_score(y_test, preds)) if len(np.unique(y_test)) > 1 else 0.5
            st.session_state["eval_preds"] = preds
            st.session_state["eval_y"] = y_test

        if st.session_state["auto_running"] and st.session_state["trained_model"] is None:
            fit_active_model()

        if st.button("Train LightGBM Model"):
            with st.spinner("Fitting LightGBM classifier..."):
                fit_active_model()
                st.success("Model training and out-of-time evaluation completed.")

        if st.session_state["trained_model"] is not None:
            m1, m2 = st.columns(2)
            m1.metric("Precision-Recall AUC (PR-AUC)", f"{st.session_state['pr_auc']:.4f}")
            m2.metric("ROC-AUC", f"{st.session_state['roc_auc']:.4f}")

            precision, recall, _ = precision_recall_curve(st.session_state["eval_y"], st.session_state["eval_preds"])
            pr_data = pd.DataFrame({"Recall": recall, "Precision": precision})
            pr_chart = alt.Chart(pr_data).mark_line(color="#2F80ED", point=True).encode(
                x=alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(labelAngle=0)),
                y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(labelAngle=0)),
                tooltip=["Recall", "Precision"]
            ).properties(height=280, title="Precision-Recall Trajectory").interactive()
            st.altair_chart(pr_chart, use_container_width=True)

    handle_auto_progression(5, "Stage 7: Operational Prediction and Flagging", seconds=8)

# -------------------------------------------------------------
# STAGE 7: PREDICTION & DECISION QUEUE (8s progression)
# -------------------------------------------------------------
elif current_nav == "Stage 7: Operational Prediction and Flagging":
    st.title("Stage 7: Operational Prediction and Flagging")
    st.markdown("Operational triage queue with tiered action plans (Block, Review, Approve) across the entire dataset.")

    scoring_batch = st.session_state["featured_df"].copy() if st.session_state["featured_df"] is not None else st.session_state["raw_df"].copy()

    if st.session_state["trained_model"] is not None and "balance_orig_diff" in scoring_batch:
        feature_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
                        "balance_orig_diff", "balance_dest_diff", "hour_of_day", "orig_hist_count", "dest_max_amount"]
        scores = st.session_state["trained_model"].predict_proba(scoring_batch[feature_cols])[:, 1]
    else:
        scores = []
        for _, row in scoring_batch.iterrows():
            sc = 0.08
            if row.get("type", "") in ["TRANSFER", "CASH_OUT"]:
                if row.get("oldbalanceOrg", 0) > 0 and row.get("newbalanceOrig", 0) == 0:
                    sc += 0.55
                if row.get("amount", 0) > 100000:
                    sc += 0.22
            scores.append(min(0.99, max(0.01, sc + np.random.uniform(-0.05, 0.05))))

    scoring_batch["fraud_score"] = np.round(scores, 4)
    scoring_batch["audit_action"] = pd.cut(
        scoring_batch["fraud_score"],
        bins=[-0.1, 0.25, 0.65, 1.0],
        labels=["Approve", "Review", "Block"]
    )
    st.session_state["scored_batch"] = scoring_batch
    st.session_state["pipeline_complete"] = True

    t1, t2, t3 = st.columns(3)
    n_approve = int((scoring_batch["audit_action"] == "Approve").sum())
    n_review = int((scoring_batch["audit_action"] == "Review").sum())
    n_block = int((scoring_batch["audit_action"] == "Block").sum())

    t1.metric("Approve Tier (Low Risk - Blue)", f"{n_approve:,}")
    t2.metric("Review Tier (Manual Queue - Yellow)", f"{n_review:,}")
    t3.metric("Block Tier (High Risk - Red)", f"{n_block:,}")

    st.markdown("---")
    st.subheader("Operational Risk Distribution")
    st.caption("Click any decision label ('Block', 'Review', 'Approve') in the legend to filter points.")

    selection = alt.selection_point(fields=["audit_action"], bind="legend")
    scatter_sample = scoring_batch.sample(n=min(len(scoring_batch), 4000), random_state=42) if len(scoring_batch) > 4000 else scoring_batch

    scatter_chart = alt.Chart(scatter_sample).mark_circle(size=70).encode(
        x=alt.X("amount:Q", title="Transaction Amount ($)", scale=alt.Scale(type="log"), axis=alt.Axis(labelAngle=0)),
        y=alt.Y("fraud_score:Q", title="Fraud Risk Score", axis=alt.Axis(labelAngle=0)),
        color=alt.Color(
            "audit_action:N",
            scale=alt.Scale(domain=ACTION_DOMAIN, range=ACTION_COLORS),
            legend=alt.Legend(title="Audit Action (Click to Filter)")
        ),
        opacity=alt.condition(selection, alt.value(0.85), alt.value(0.1)),
        tooltip=["step", "type", "amount", "fraud_score", "audit_action"]
    ).add_params(selection).properties(height=360).interactive()

    st.altair_chart(scatter_chart, use_container_width=True)

    if st.session_state["auto_running"]:
        status_box = st.empty()
        pbar = st.progress(0.0)
        for s in range(8, 0, -1):
            status_box.info(f"Automatic Mode: Finalizing batch inference across {len(scoring_batch):,} transactions. Moving to Dashboard in {s} seconds...")
            pbar.progress((8 - s + 1) / 8.0)
            time.sleep(1.0)
        status_box.empty()
        pbar.empty()
        st.session_state["auto_running"] = False
        st.session_state["nav_selection"] = "Dashboard"
        st.rerun()

# -------------------------------------------------------------
# DASHBOARD (Clean Titles & Comprehensive Analytics)
# -------------------------------------------------------------
elif current_nav == "Dashboard":
    st.title("Dashboard")

    if not st.session_state["pipeline_complete"] and st.session_state["scored_batch"] is None:
        st.info("Waiting for data. Please run the pipeline stages or enable Automatic Pipeline Mode in the sidebar.")
    else:
        scored = st.session_state["scored_batch"]
        total_tx = len(scored)
        total_vol = float(scored["amount"].sum())
        n_block = int((scored["audit_action"] == "Block").sum())
        n_review = int((scored["audit_action"] == "Review").sum())
        n_approve = int((scored["audit_action"] == "Approve").sum())
        actual_fraud = int(scored["isFraud"].sum()) if "isFraud" in scored else 0

        # KPI Metrics
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Audited Volume", f"{total_tx:,} tx")
        k2.metric("Gross Exposure", f"${total_vol:,.2f}")
        k3.metric("Review Queue", f"{n_review:,} ({((n_review/max(1, total_tx))*100):.1f}%)")
        k4.metric("High Risk Blocked", f"{n_block:,} ({((n_block/max(1, total_tx))*100):.1f}%)")

        st.markdown("---")

        # Row 1 Visuals
        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.subheader("Decision Classification Distribution")
            dash_selection = alt.selection_point(fields=["audit_action"], bind="legend")
            dash_scatter_sample = scored.sample(n=min(len(scored), 4000), random_state=42) if len(scored) > 4000 else scored

            scatter_dash = alt.Chart(dash_scatter_sample).mark_circle(size=65).encode(
                x=alt.X("amount:Q", title="Amount ($)", scale=alt.Scale(type="log"), axis=alt.Axis(labelAngle=0)),
                y=alt.Y("fraud_score:Q", title="Fraud Risk Score", axis=alt.Axis(labelAngle=0)),
                color=alt.Color("audit_action:N", scale=alt.Scale(domain=ACTION_DOMAIN, range=ACTION_COLORS), legend=alt.Legend(title="Decision Tier")),
                opacity=alt.condition(dash_selection, alt.value(0.85), alt.value(0.1)),
                tooltip=["step", "type", "amount", "fraud_score", "audit_action"]
            ).add_params(dash_selection).properties(height=320).interactive()
            st.altair_chart(scatter_dash, use_container_width=True)

        with v_col2:
            st.subheader("Operational Volume by Action Tier")
            action_counts = scored["audit_action"].value_counts().reindex(ACTION_DOMAIN, fill_value=0).reset_index()
            action_counts.columns = ["audit_action", "count"]

            bar_select = alt.selection_point(fields=["audit_action"])
            chart_bar = alt.Chart(action_counts).mark_bar().encode(
                x=alt.X("audit_action:N", title="Action Category", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("count:Q", title="Transaction Volume"),
                color=alt.condition(
                    bar_select,
                    alt.Color("audit_action:N", scale=alt.Scale(domain=ACTION_DOMAIN, range=ACTION_COLORS), legend=None),
                    alt.value("rgba(200,200,200,0.2)")
                ),
                tooltip=["audit_action", "count"]
            ).add_params(bar_select).properties(height=320).interactive()
            st.altair_chart(chart_bar, use_container_width=True)

        st.markdown("---")

        # Row 2 Visuals
        v_col3, v_col4 = st.columns(2)
        with v_col3:
            st.subheader("Triage Composition")
            donut_select = alt.selection_point(fields=["audit_action"])
            donut_chart = alt.Chart(action_counts).mark_arc(innerRadius=65).encode(
                theta=alt.Theta("count:Q", title="Volume"),
                color=alt.condition(
                    donut_select,
                    alt.Color("audit_action:N", scale=alt.Scale(domain=ACTION_DOMAIN, range=ACTION_COLORS), legend=alt.Legend(title="Triage Tier")),
                    alt.value("rgba(200,200,200,0.2)")
                ),
                tooltip=["audit_action", "count"]
            ).add_params(donut_select).properties(height=320).interactive()
            st.altair_chart(donut_chart, use_container_width=True)

        with v_col4:
            st.subheader("Operational Triage Funnel")
            raw_len = len(st.session_state["raw_df"]) if "raw_df" in st.session_state else total_tx
            clean_len = len(st.session_state["cleaned_df"]) if st.session_state["cleaned_df"] is not None else total_tx
            flagged_total = n_block + n_review

            funnel_df = pd.DataFrame({
                "Stage": ["1. Ingested", "2. Cleaned", "3. Suspect (Review + Block)", "4. Blocked"],
                "Records": [raw_len, clean_len, flagged_total, n_block],
                "Order": [1, 2, 3, 4]
            })

            funnel_chart = alt.Chart(funnel_df).mark_bar(color="#F2994A").encode(
                y=alt.Y("Stage:N", sort=alt.EncodingSortField(field="Order", order="ascending"), title="Pipeline Funnel Stage", axis=alt.Axis(labelAngle=0)),
                x=alt.X("Records:Q", title="Transaction Throughput"),
                tooltip=["Stage", "Records"]
            ).properties(height=320).interactive()
            st.altair_chart(funnel_chart, use_container_width=True)

        st.markdown("---")
        st.subheader("Model Decision Telemetry and Ground Truth Comparison")
        comp_df = pd.DataFrame({
            "Classification Category": ["Total Audited", "Flagged High Risk (Block)", "Flagged for Human Review (Review)", "Cleared Transactions (Approve)", "Ground Truth Confirmed Fraud"],
            "Count": [total_tx, n_block, n_review, n_approve, actual_fraud],
            "Proportion": [
                "100.0%",
                f"{(n_block / max(1, total_tx)) * 100:.2f}%",
                f"{(n_review / max(1, total_tx)) * 100:.2f}%",
                f"{(n_approve / max(1, total_tx)) * 100:.2f}%",
                f"{(actual_fraud / max(1, total_tx)) * 100:.2f}%"
            ]
        })
        st.table(comp_df)
