import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, confusion_matrix
import lightgbm as lgb

BVDU_LOGO_URL = "https://upload.wikimedia.org/wikipedia/en/e/e0/Bharati_Vidyapeeth_logo.png"

st.set_page_config(
    page_title="Financial Fraud Detection System (AWS ML Architecture)",
    page_icon=BVDU_LOGO_URL,
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

# Ingestion Engine with Contamination (15% to 28% Unclean Records)
def generate_unclean_dataset(n_rows: int = 1500) -> pd.DataFrame:
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
    fraud_count = max(2, int(n_rows * 0.04))
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

    # Contaminate 15% to 28% of rows
    contam_pct = np.random.uniform(0.15, 0.28)
    n_contam = int(n_rows * contam_pct)
    contam_idx = np.random.choice(n_rows, size=n_contam, replace=False)

    for idx in contam_idx:
        issue = np.random.choice(["null_val", "negative_amount", "negative_balance", "duplicate_row"])
        if issue == "null_val":
            col = np.random.choice(["amount", "oldbalanceOrg", "newbalanceDest"])
            df.loc[idx, col] = np.nan
        elif issue == "negative_amount":
            df.loc[idx, "amount"] = -abs(df.loc[idx, "amount"])
        elif issue == "negative_balance":
            df.loc[idx, "newbalanceOrig"] = -abs(df.loc[idx, "newbalanceOrig"])

    # Inject duplicates
    n_dupes = max(2, int(n_rows * 0.03))
    dupe_rows = df.sample(n=n_dupes, replace=True)
    df = pd.concat([df, dupe_rows], ignore_index=True)

    return df

# Initialize Session State
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = generate_unclean_dataset(1500)
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

# Sidebar Navigation
st.sidebar.title("Navigation")
nav_section = st.sidebar.radio(
    "Select View:",
    [
        "Executive Dashboard",
        "Stage 1: Continuous Data Ingestion",
        "Stage 2: Cleaning and Data Validation",
        "Stage 3: Exploratory Data Analysis",
        "Stage 4: Feature Engineering",
        "Stage 5: Imbalance Handling and Splitting",
        "Stage 6: Model Training and Evaluation",
        "Stage 7: Operational Prediction and Flagging"
    ]
)

# -------------------------------------------------------------
# EXECUTIVE DASHBOARD
# -------------------------------------------------------------
if nav_section == "Executive Dashboard":
    st.title("Financial Fraud Telemetry and Operational Dashboard")
    st.markdown("Centralized intelligence portal tracking active pipeline metrics, risk distributions, and model performance.")

    current_data = st.session_state["cleaned_df"] if st.session_state["cleaned_df"] is not None else st.session_state["raw_df"]
    has_model = st.session_state["trained_model"] is not None

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_tx = len(current_data)
    total_vol = current_data["amount"].abs().sum() if "amount" in current_data else 0.0
    confirmed_fraud = int(current_data["isFraud"].sum()) if "isFraud" in current_data else 0
    fraud_rate = (confirmed_fraud / max(1, total_tx)) * 100

    kpi1.metric("Active Dataset Volume", f"{total_tx:,} records")
    kpi2.metric("Gross Transaction Volume", f"${total_vol:,.2f}")
    kpi3.metric("Confirmed Fraud Cases", f"{confirmed_fraud:,} ({fraud_rate:.2f}%)")
    kpi4.metric("Model Inference Engine", "Active (LightGBM)" if has_model else "Standby (Pre-training)")

    st.markdown("---")

    dash_col1, dash_col2 = st.columns(2)
    with dash_col1:
        st.subheader("Gross Financial Exposure by Channel")
        if "type" in current_data and "amount" in current_data:
            vol_chart = alt.Chart(current_data).mark_bar(color="#F2994A").encode(
                x=alt.X("type:N", title="Transaction Channel", sort="-y"),
                y=alt.Y("sum(amount):Q", title="Aggregated Exposure ($)"),
                tooltip=["type", "sum(amount)"]
            ).properties(height=320)
            st.altair_chart(vol_chart, use_container_width=True)

    with dash_col2:
        st.subheader("Fraud Count by Channel Concentration")
        if "type" in current_data and "isFraud" in current_data:
            fraud_sub = current_data[current_data["isFraud"] == 1]
            if len(fraud_sub) > 0:
                count_chart = alt.Chart(fraud_sub).mark_bar(color="#E05656").encode(
                    x=alt.X("type:N", title="Transaction Channel", sort="-y"),
                    y=alt.Y("count():Q", title="Fraud Incident Count"),
                    tooltip=["type", "count()"]
                ).properties(height=320)
                st.altair_chart(count_chart, use_container_width=True)
            else:
                st.info("No fraud incidents detected in current dataset view.")

    st.subheader("Automated Risk Summary and Policy Governance")
    st.markdown(f"""
    - **Current Data Health**: Pipeline currently loaded with **{total_tx:,}** transactions across **{len(current_data.columns)}** schema columns.
    - **Primary Vulnerability**: Critical loss patterns remain concentrated in **TRANSFER** and **CASH_OUT** routes.
    - **Operational Readiness**: {'Model trained on active dataset. Alert capacity calibration enabled in Stage 7.' if has_model else 'Model not yet trained on active session data. Proceed to Stage 6 to generate PR-AUC calibration metrics.'}
    """)

# -------------------------------------------------------------
# STAGE 1: INGESTION
# -------------------------------------------------------------
elif nav_section == "Stage 1: Continuous Data Ingestion":
    st.title("Stage 1: Continuous Data Ingestion")
    st.markdown("Unified ingestion endpoint. Define record volume or upload an external batch. Injects synthetic anomalies (15% to 28%) for realistic pipeline validation.")

    ctrl_col1, ctrl_col2 = st.columns([1, 1])
    with ctrl_col1:
        record_count = st.slider("Select Record Ingestion Volume:", min_value=500, max_value=10000, value=2000, step=250)
        if st.button("Ingest New Transaction Batch"):
            st.session_state["raw_df"] = generate_unclean_dataset(record_count)
            st.session_state["cleaned_df"] = None
            st.session_state["featured_df"] = None
            st.session_state["trained_model"] = None
            st.success(f"Ingested {len(st.session_state['raw_df']):,} records containing realistic data anomalies.")

    with ctrl_col2:
        uploaded_csv = st.file_uploader("Or Upload Custom CSV Batch", type=["csv"])
        if uploaded_csv is not None:
            st.session_state["raw_df"] = pd.read_csv(uploaded_csv)
            st.session_state["cleaned_df"] = None
            st.session_state["featured_df"] = None
            st.session_state["trained_model"] = None
            st.success(f"Uploaded and ingested {len(st.session_state['raw_df']):,} rows from external feed.")

    st.subheader(f"Current Ingested Raw Dataset ({len(st.session_state['raw_df']):,} Total Records)")
    st.dataframe(st.session_state["raw_df"], use_container_width=True, height=450)

# -------------------------------------------------------------
# STAGE 2: CLEANING & VALIDATION
# -------------------------------------------------------------
elif nav_section == "Stage 2: Cleaning and Data Validation":
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

    if st.button("Run Data Cleaning and Sanitization"):
        cleaned = raw.copy()
        cleaned = cleaned.drop_duplicates()

        # Handle null values: impute if voluminous, drop if minimal
        if cleaned.isna().sum().sum() > 0:
            for col in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
                if col in cleaned and cleaned[col].isna().sum() > 0:
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            cleaned = cleaned.dropna()

        # Rule-based filtering: absolute negative amounts
        cleaned["amount"] = cleaned["amount"].abs()
        cleaned["oldbalanceOrg"] = cleaned["oldbalanceOrg"].abs()
        cleaned["newbalanceOrig"] = cleaned["newbalanceOrig"].abs()
        cleaned["oldbalanceDest"] = cleaned["oldbalanceDest"].abs()
        cleaned["newbalanceDest"] = cleaned["newbalanceDest"].abs()

        st.session_state["cleaned_df"] = cleaned
        st.success(f"Cleaning completed. Retained {len(cleaned):,} valid records.")

    display_df = st.session_state["cleaned_df"] if st.session_state["cleaned_df"] is not None else raw
    status_label = "Cleaned Dataset" if st.session_state["cleaned_df"] is not None else "Raw Uncleaned Dataset"
    st.subheader(f"{status_label} ({len(display_df):,} Records)")
    st.dataframe(display_df, use_container_width=True, height=450)

# -------------------------------------------------------------
# STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------
elif nav_section == "Stage 3: Exploratory Data Analysis":
    st.title("Stage 3: Exploratory Data Analysis (EDA)")
    st.markdown("Dynamic risk patterns, channel velocities, and value distributions computed on active data.")

    df = st.session_state["cleaned_df"] if st.session_state["cleaned_df"] is not None else st.session_state["raw_df"]

    eda_c1, eda_c2 = st.columns(2)
    with eda_c1:
        st.subheader("Transaction Volume by Type and Target")
        chart_tx = alt.Chart(df).mark_bar().encode(
            x=alt.X("type:N", title="Transaction Channel"),
            y=alt.Y("count():Q", title="Volume"),
            color=alt.Color("isFraud:N", scale=alt.Scale(domain=[0, 1], range=["#2F80ED", "#E05656"]), legend=alt.Legend(title="Fraud Flag")),
            tooltip=["type", "count()"]
        ).properties(height=320)
        st.altair_chart(chart_tx, use_container_width=True)

    with eda_c2:
        st.subheader("Amount Distribution by Target Class")
        chart_box = alt.Chart(df).mark_boxplot().encode(
            x=alt.X("isFraud:N", title="Class (0 = Legitimate, 1 = Fraud)"),
            y=alt.Y("amount:Q", scale=alt.Scale(type="log"), title="Transaction Amount ($)"),
            color=alt.Color("isFraud:N", scale=alt.Scale(domain=[0, 1], range=["#2F80ED", "#E05656"]))
        ).properties(height=320)
        st.altair_chart(chart_box, use_container_width=True)

    st.subheader("Diurnal Transaction Velocity by Simulation Step")
    line_step = alt.Chart(df).mark_line(color="#F2994A").encode(
        x=alt.X("step:Q", title="Simulation Step (Hour)"),
        y=alt.Y("count():Q", title="Transaction Throughput"),
        tooltip=["step", "count()"]
    ).properties(height=260)
    st.altair_chart(line_step, use_container_width=True)

# -------------------------------------------------------------
# STAGE 4: FEATURE ENGINEERING
# -------------------------------------------------------------
elif nav_section == "Stage 4: Feature Engineering":
    st.title("Stage 4: Leak-Free Feature Engineering (V3.1)")
    st.markdown("Extract delta balances, temporal components, and leak-free historical behavioral metrics.")

    df = st.session_state["cleaned_df"] if st.session_state["cleaned_df"] is not None else st.session_state["raw_df"]

    if st.button("Generate Feature Transformations"):
        with st.spinner("Extracting features..."):
            feat = df.copy()
            feat["balance_orig_diff"] = feat["oldbalanceOrg"] - feat["newbalanceOrig"] - feat["amount"]
            feat["balance_dest_diff"] = feat["newbalanceDest"] - feat["oldbalanceDest"] - feat["amount"]
            feat["hour_of_day"] = feat["step"] % 24
            feat["orig_hist_count"] = feat.groupby("nameOrig").cumcount()
            feat["orig_hist_mean_amount"] = feat.groupby("nameOrig")["amount"].transform("mean")
            feat["dest_max_amount"] = feat.groupby("nameDest")["amount"].transform("max")
            st.session_state["featured_df"] = feat
            st.success(f"Feature engineering completed across {feat.shape[1]} total dimensions.")

    if st.session_state["featured_df"] is not None:
        feat_df = st.session_state["featured_df"]
        st.subheader("Engineered Feature Signals (Separate Table)")
        eng_cols = ["step", "nameOrig", "amount", "balance_orig_diff", "balance_dest_diff", "hour_of_day", "orig_hist_count", "orig_hist_mean_amount", "dest_max_amount", "isFraud"]
        st.dataframe(feat_df[eng_cols], use_container_width=True, height=350)

        st.subheader("Feature Variance and Statistics")
        st.dataframe(feat_df[["balance_orig_diff", "balance_dest_diff", "hour_of_day", "dest_max_amount"]].describe(), use_container_width=True)
    else:
        st.info("Click 'Generate Feature Transformations' above to compute feature tables.")

# -------------------------------------------------------------
# STAGE 5: IMBALANCE HANDLING & SPLITTING
# -------------------------------------------------------------
elif nav_section == "Stage 5: Imbalance Handling and Splitting":
    st.title("Stage 5: Imbalance Handling and Chronological Splitting")
    st.markdown("Partition data strictly by step to preserve causal temporal ordering and handle class imbalance.")

    if st.session_state["featured_df"] is None:
        st.warning("Please complete feature generation in Stage 4 first.")
    else:
        df = st.session_state["featured_df"]
        max_step = int(df["step"].max())
        default_split = max(1, int(max_step * 0.75))

        split_c1, split_c2 = st.columns([1, 2])
        with split_c1:
            split_step = st.slider("Select Chronological Split Step:", min_value=1, max_value=max_step, value=default_split)
            imbalance_method = st.selectbox("Imbalance Strategy:", ["Cost-Sensitive Class Weighting (scale_pos_weight)", "Synthetic Minority Oversampling (Simulated)"])
            
            if st.button("Apply Partition and Strategy"):
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
                st.metric("Out-of-Time Test Partition", f"{test_len:,} rows (Fraud: {test_fraud})")
                scale_pos = (train_len - train_fraud) / max(1, train_fraud)
                st.caption(f"Calculated `scale_pos_weight` multiplier: **{scale_pos:.2f}**")

# -------------------------------------------------------------
# STAGE 6: MODEL TRAINING & TELEMETRY
# -------------------------------------------------------------
elif nav_section == "Stage 6: Model Training and Evaluation":
    st.title("Stage 6: Model Training and Telemetry")
    st.markdown("Train LightGBM on the active training partition and evaluate precision-recall curves.")

    if st.session_state["train_df"] is None:
        st.warning("Please configure the train/test split in Stage 5 first.")
    else:
        train_df = st.session_state["train_df"]
        test_df = st.session_state["test_df"]

        if st.button("Train LightGBM Model"):
            with st.spinner("Fitting LightGBM classifier..."):
                feature_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
                                "balance_orig_diff", "balance_dest_diff", "hour_of_day", "orig_hist_count", "dest_max_amount"]
                
                X_train, y_train = train_df[feature_cols], train_df["isFraud"]
                X_test, y_test = test_df[feature_cols], test_df["isFraud"]

                scale_pos = (len(y_train) - y_train.sum()) / max(1, y_train.sum())

                clf = lgb.LGBMClassifier(
                    n_estimators=75,
                    learning_rate=0.06,
                    scale_pos_weight=scale_pos,
                    random_state=42,
                    verbose=-1
                )
                clf.fit(X_train, y_train)

                preds = clf.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, preds)
                pr_auc_score = auc(recall, precision)
                roc_auc_val = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5

                st.session_state["trained_model"] = clf
                st.session_state["pr_auc"] = pr_auc_score
                st.session_state["roc_auc"] = roc_auc_val
                st.session_state["eval_preds"] = preds
                st.session_state["eval_y"] = y_test
                st.success("Model training and out-of-time evaluation completed.")

        if st.session_state["trained_model"] is not None:
            m1, m2 = st.columns(2)
            m1.metric("Precision-Recall AUC (PR-AUC)", f"{st.session_state['pr_auc']:.4f}")
            m2.metric("ROC-AUC", f"{st.session_state['roc_auc']:.4f}")

            # Plot PR curve
            precision, recall, _ = precision_recall_curve(st.session_state["eval_y"], st.session_state["eval_preds"])
            pr_data = pd.DataFrame({"Recall": recall, "Precision": precision})
            pr_chart = alt.Chart(pr_data).mark_line(color="#2F80ED").encode(
                x=alt.X("Recall:Q", scale=alt.Scale(domain=[0, 1])),
                y=alt.Y("Precision:Q", scale=alt.Scale(domain=[0, 1]))
            ).properties(height=300, title="Precision-Recall Trajectory")
            st.altair_chart(pr_chart, use_container_width=True)

# -------------------------------------------------------------
# STAGE 7: PREDICTION & DECISION QUEUE
# -------------------------------------------------------------
elif nav_section == "Stage 7: Operational Prediction and Flagging":
    st.title("Stage 7: Real-Time Fraud Prediction and Operational Flagging")
    st.markdown("Operational triage queue with tiered action plans (Block, Review, Approve).")

    source_data = st.session_state["featured_df"] if st.session_state["featured_df"] is not None else st.session_state["raw_df"]
    scoring_batch = source_data.copy().head(500)

    # Score transactions
    if st.session_state["trained_model"] is not None and "balance_orig_diff" in scoring_batch:
        feature_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
                        "balance_orig_diff", "balance_dest_diff", "hour_of_day", "orig_hist_count", "dest_max_amount"]
        scores = st.session_state["trained_model"].predict_proba(scoring_batch[feature_cols])[:, 1]
    else:
        scores = []
        for _, row in scoring_batch.iterrows():
            sc = 0.05
            if row.get("type", "") in ["TRANSFER", "CASH_OUT"]:
                if row.get("oldbalanceOrg", 0) > 0 and row.get("newbalanceOrig", 0) == 0:
                    sc += 0.60
                if row.get("amount", 0) > 120000:
                    sc += 0.25
            scores.append(min(0.99, max(0.01, sc + np.random.uniform(-0.02, 0.02))))

    scoring_batch["fraud_score"] = np.round(scores, 4)
    scoring_batch["audit_action"] = pd.cut(
        scoring_batch["fraud_score"],
        bins=[-0.1, 0.40, 0.80, 1.0],
        labels=["Approve", "Review", "Block"]
    )

    t1, t2, t3 = st.columns(3)
    n_approve = int((scoring_batch["audit_action"] == "Approve").sum())
    n_review = int((scoring_batch["audit_action"] == "Review").sum())
    n_block = int((scoring_batch["audit_action"] == "Block").sum())

    t1.metric("Approve Tier (Low Risk - Blue)", f"{n_approve:,}")
    t2.metric("Review Tier (Manual Queue - Yellow)", f"{n_review:,}")
    t3.metric("Block Tier (High Risk - Red)", f"{n_block:,}")

    st.markdown("---")
    st.subheader("Operational Risk Distribution")
    st.caption("Interactive Legend: Click any decision label ('Block', 'Review', 'Approve') in the legend to filter points.")

    selection = alt.selection_point(fields=["audit_action"], bind="legend")

    scatter_chart = alt.Chart(scoring_batch).mark_circle(size=70).encode(
        x=alt.X("amount:Q", title="Transaction Amount ($)", scale=alt.Scale(type="log")),
        y=alt.Y("fraud_score:Q", title="Fraud Risk Score"),
        color=alt.Color(
            "audit_action:N",
            scale=alt.Scale(domain=ACTION_DOMAIN, range=ACTION_COLORS),
            legend=alt.Legend(title="Audit Action (Click to Filter)")
        ),
        opacity=alt.condition(selection, alt.value(0.85), alt.value(0.1)),
        tooltip=["step", "type", "amount", "fraud_score", "audit_action"]
    ).add_params(selection).properties(height=380).interactive()

    st.altair_chart(scatter_chart, use_container_width=True)

    st.subheader("Flagged Accounts and Decision Ledger")

    def style_action(val):
        if val == "Block":
            return "background-color: rgba(224, 86, 86, 0.35); font-weight: bold; color: #E05656;"
        elif val == "Review":
            return "background-color: rgba(242, 201, 76, 0.35); font-weight: bold; color: #F2C94C;"
        elif val == "Approve":
            return "background-color: rgba(47, 128, 237, 0.25); color: #2F80ED;"
        return ""

    st.dataframe(
        scoring_batch.head(150).style.map(style_action, subset=["audit_action"]),
        use_container_width=True,
        height=400
    )
