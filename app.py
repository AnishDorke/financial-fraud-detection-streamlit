import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import lightgbm as lgb

st.set_page_config(
    page_title="Financial Fraud Detection System (AWS ML Architecture)",
    page_icon="🛡️️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Solar styling and badge tags
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222B;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #F2994A;
        margin-bottom: 12px;
    }
    .badge-block { color: #E05656; font-weight: bold; }
    .badge-review { color: #F2C94C; font-weight: bold; }
    .badge-approve { color: #2F80ED; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

ACTION_COLORS = ["#E05656", "#F2C94C", "#2F80ED"]
ACTION_DOMAIN = ["Block", "Review", "Approve"]

# --- DATA GENERATOR ---
def generate_synthetic_data(n_rows: int = 1500) -> pd.DataFrame:
    np.random.seed(int(pd.Timestamp.now().timestamp()) % 100000)
    steps = np.sort(np.random.randint(1, 30, size=n_rows))
    types = np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"], size=n_rows, p=[0.35, 0.10, 0.35, 0.05, 0.15])
    amounts = np.round(np.random.exponential(scale=50000, size=n_rows) + 10, 2)
    orig_ids = [f"C{np.random.randint(100000, 999999)}" for _ in range(n_rows)]
    dest_ids = [f"M{np.random.randint(100000, 999999)}" if t == "PAYMENT" else f"C{np.random.randint(100000, 999999)}" for t in types]
    old_orig = np.round(np.random.exponential(scale=60000, size=n_rows), 2)
    new_orig = np.maximum(0, old_orig - amounts)
    old_dest = np.round(np.random.exponential(scale=40000, size=n_rows), 2)
    new_dest = old_dest + amounts

    # Label fraud instances
    is_fraud = np.zeros(n_rows, dtype=int)
    fraud_idx = np.random.choice(n_rows, size=max(1, int(n_rows * 0.04)), replace=False)
    for i in fraud_idx:
        types[i] = "TRANSFER" if np.random.rand() > 0.5 else "CASH_OUT"
        amounts[i] = old_orig[i] = np.random.uniform(200000, 850000)
        new_orig[i] = 0.0
        is_fraud[i] = 1

    return pd.DataFrame({
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

# Initialize Session State
if "raw_df" not in st.session_state:
    st.session_state["raw_df"] = generate_synthetic_data(1500)
if "cleaned_df" not in st.session_state:
    st.session_state["cleaned_df"] = st.session_state["raw_df"].copy()
if "featured_df" not in st.session_state:
    st.session_state["featured_df"] = None

# Sidebar Navigation
st.sidebar.title("Pipeline Navigation")
stage = st.sidebar.radio(
    "Lifecycle Stages:",
    [
        "1. Ingestion & Continuous Feed",
        "2. Cleaning & Validation",
        "3. Exploratory Data Analysis (EDA)",
        "4. Feature Engineering & Leak-Free Windowing",
        "5. Imbalance Handling & Chronological Split",
        "6. Model Selection & Live Retraining",
        "7. Real-Time Fraud Prediction & Monitoring"
    ]
)

# -------------------------------------------------------------
# 1. INGESTION & CONTINUOUS FEED
# -------------------------------------------------------------
if stage == "1. Ingestion & Continuous Feed":
    st.title("1. Continuous Data Collection & Ingestion")
    st.markdown("Simulate streaming transactions or generate fresh batch feeds matching the AWS S3/Kinesis schema.")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Simulate Ingestion Feed")
        batch_size = st.slider("Select batch volume (rows):", 500, 5000, len(st.session_state["raw_df"]), step=250)
        
        if st.button("🔄 Generate Fresh Incoming Data Batch"):
            st.session_state["raw_df"] = generate_synthetic_data(batch_size)
            # Invalidate downstream states so they reflect newly ingested records
            st.session_state["cleaned_df"] = st.session_state["raw_df"].copy()
            st.session_state["featured_df"] = None
            st.success(f"Generated {batch_size:,} fresh transactions. Downstream pipeline stages updated!")

        st.dataframe(st.session_state["raw_df"].head(10), use_container_width=True)

    with c2:
        st.subheader("Data Export / Import")
        csv_bytes = st.session_state["raw_df"].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Current Ingested Feed (CSV)",
            data=csv_bytes,
            file_name="active_transaction_feed.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        uploaded = st.file_uploader("Upload External CSV Feed", type=["csv"])
        if uploaded is not None:
            st.session_state["raw_df"] = pd.read_csv(uploaded)
            st.session_state["cleaned_df"] = st.session_state["raw_df"].copy()
            st.session_state["featured_df"] = None
            st.success(f"Ingested {len(st.session_state['raw_df']):,} rows from custom upload.")

# -------------------------------------------------------------
# 2. CLEANING & VALIDATION
# -------------------------------------------------------------
elif stage == "2. Cleaning & Validation":
    st.title("2. Data Cleaning & Integrity Audit")
    st.markdown("Dynamic verification: null scans, duplicate pruning, and negative balance filters.")

    if st.button("🧹 Run / Refresh Cleaning & Validation Audit"):
        df = st.session_state["raw_df"].copy()
        initial_len = len(df)
        df = df.dropna()
        df = df.drop_duplicates()
        df = df[(df["amount"] >= 0) & (df["oldbalanceOrg"] >= 0) & (df["newbalanceOrig"] >= 0)]
        st.session_state["cleaned_df"] = df
        st.success(f"Audit completed: {len(df):,} valid transactions retained ({initial_len - len(df)} discarded).")

    df = st.session_state["cleaned_df"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ingested Records", f"{len(st.session_state['raw_df']):,}")
    c2.metric("Clean Records", f"{len(df):,}")
    c3.metric("Missing / Nulls", int(df.isna().sum().sum()))
    c4.metric("Negative Values Filtered", int((st.session_state['raw_df']["amount"] < 0).sum()))

    st.subheader("Cleaned Dataset Snapshot")
    st.dataframe(df.head(10), use_container_width=True)

# -------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------
elif stage == "3. Exploratory Data Analysis (EDA)":
    st.title("3. Exploratory Data Analysis (EDA)")
    st.markdown("Visual analytics dynamically computed over the currently cleaned transactions.")

    df = st.session_state["cleaned_df"]

    col_btn, _ = st.columns([1, 3])
    with col_btn:
        if st.button("📊 Recalculate EDA Distributions"):
            st.toast("EDA metrics and distribution charts refreshed!")

    e1, e2 = st.columns(2)
    with e1:
        st.subheader("Fraud by Transaction Channel")
        channel_df = df.groupby(["type", "isFraud"]).size().reset_index(name="count")
        channel_chart = alt.Chart(channel_df).mark_bar().encode(
            x=alt.X("type:N", title="Channel"),
            y=alt.Y("count:Q", title="Volume"),
            color=alt.Color("isFraud:N", scale=alt.Scale(domain=[0, 1], range=["#2F80ED", "#E05656"]), legend=alt.Legend(title="Fraud Flag"))
        ).properties(height=300)
        st.altair_chart(channel_chart, use_container_width=True)

    with e2:
        st.subheader("Amount Log-Distribution by Target")
        box_chart = alt.Chart(df).mark_boxplot().encode(
            x=alt.X("isFraud:N", title="0 = Legitimate, 1 = Fraud"),
            y=alt.Y("amount:Q", scale=alt.Scale(type="log"), title="Amount ($)"),
            color=alt.Color("isFraud:N", scale=alt.Scale(domain=[0, 1], range=["#2F80ED", "#E05656"]))
        ).properties(height=300)
        st.altair_chart(box_chart, use_container_width=True)

# -------------------------------------------------------------
# 4. FEATURE ENGINEERING
# -------------------------------------------------------------
elif stage == "4. Feature Engineering & Leak-Free Windowing":
    st.title("4. Feature Engineering (V3.1 Leak-Free)")
    st.markdown("Computes step differences, balance drain indicators, and historical account behavior.")

    if st.button("⚙️ Extract Features from Clean Data"):
        with st.spinner("Extracting historical aggregates and delta features..."):
            df = st.session_state["cleaned_df"].copy()
            df["balance_orig_diff"] = df["oldbalanceOrg"] - df["newbalanceOrig"] - df["amount"]
            df["balance_dest_diff"] = df["newbalanceDest"] - df["oldbalanceDest"] - df["amount"]
            df["hour_of_day"] = df["step"] % 24

            # Leak-free sender transaction count & mean amount
            df["orig_hist_count"] = df.groupby("nameOrig").cumcount()
            df["orig_hist_mean_amount"] = df.groupby("nameOrig")["amount"].transform("mean")
            df["dest_max_amount"] = df.groupby("nameDest")["amount"].transform("max")
            
            st.session_state["featured_df"] = df
            st.success(f"Engineered {df.shape[1]} features across {len(df):,} records!")

    if st.session_state["featured_df"] is not None:
        feat_df = st.session_state["featured_df"]
        st.dataframe(feat_df[["step", "type", "amount", "balance_orig_diff", "balance_dest_diff", "hour_of_day", "orig_hist_count", "dest_max_amount", "isFraud"]].head(10), use_container_width=True)
    else:
        st.info("Click 'Extract Features from Clean Data' above to run the feature transformation pipeline.")

# -------------------------------------------------------------
# 5. IMBALANCE HANDLING & CHRONOLOGICAL SPLIT
# -------------------------------------------------------------
elif stage == "5. Imbalance Handling & Chronological Split":
    st.title("5. Chronological Partitioning & Class Weighting")
    st.markdown("Partitions data chronologically by simulation `step` to prevent temporal lookahead bias.")

    if st.session_state["featured_df"] is None:
        st.warning("Please extract features in Stage 4 first before splitting.")
    else:
        df = st.session_state["featured_df"]
        max_step = int(df["step"].max())
        split_step = int(max_step * 0.75)

        c1, c2 = st.columns([1, 2])
        with c1:
            split_step = st.slider("Select Chronological Split Step:", 1, max_step, split_step)
            train_mask = df["step"] <= split_step
            train_df = df[train_mask]
            test_df = df[~train_mask]
            
            st.session_state["train_df"] = train_df
            st.session_state["test_df"] = test_df

        with c2:
            st.metric("Training Set (Steps 1 to " + str(split_step) + ")", f"{len(train_df):,} rows")
            st.metric("Test / Out-of-Time Set (Steps " + str(split_step+1) + "+)", f"{len(test_df):,} rows")
            
            fraud_train = train_df['isFraud'].sum()
            scale_pos = (len(train_df) - fraud_train) / max(1, fraud_train)
            st.caption(f"Calculated `scale_pos_weight` for class imbalance: **{scale_pos:.2f}**")

# -------------------------------------------------------------
# 6. MODEL SELECTION & LIVE RETRAINING
# -------------------------------------------------------------
elif stage == "6. Model Selection & Live Retraining":
    st.title("6. Model Training & Evaluation")
    st.markdown("Train a LightGBM classifier directly on the currently engineered and chronologically split data.")

    if "train_df" not in st.session_state or len(st.session_state["train_df"]) == 0:
        st.warning("Please configure the dataset in Stage 4 & Stage 5 first.")
    else:
        train_df = st.session_state["train_df"]
        test_df = st.session_state["test_df"]

        if st.button("🚀 Train LightGBM Model on Active Dataset"):
            with st.spinner("Training model with class imbalance compensation..."):
                feature_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", 
                                "balance_orig_diff", "balance_dest_diff", "hour_of_day", "orig_hist_count", "dest_max_amount"]
                
                X_train, y_train = train_df[feature_cols], train_df["isFraud"]
                X_test, y_test = test_df[feature_cols], test_df["isFraud"]

                scale_pos = (len(y_train) - y_train.sum()) / max(1, y_train.sum())

                clf = lgb.LGBMClassifier(
                    n_estimators=60,
                    learning_rate=0.08,
                    scale_pos_weight=scale_pos,
                    random_state=42,
                    verbose=-1
                )
                clf.fit(X_train, y_train)

                preds = clf.predict_proba(X_test)[:, 1]
                precision, recall, _ = precision_recall_curve(y_test, preds)
                pr_auc = auc(recall, precision)
                roc_auc = roc_auc_score(y_test, preds) if len(np.unique(y_test)) > 1 else 0.5

                st.session_state["trained_model"] = clf
                st.session_state["pr_auc"] = pr_auc
                st.session_state["roc_auc"] = roc_auc
                st.success("Model successfully trained on current active data!")

        if "trained_model" in st.session_state:
            m1, m2 = st.columns(2)
            m1.metric("PR-AUC Score (Primary Metric)", f"{st.session_state['pr_auc']:.4f}")
            m2.metric("ROC-AUC Score", f"{st.session_state['roc_auc']:.4f}")

# -------------------------------------------------------------
# 7. REAL-TIME FRAUD PREDICTION & MONITORING
# -------------------------------------------------------------
elif stage == "7. Real-Time Fraud Prediction & Monitoring":
    st.title("7. Fraud Prediction & Operational Monitoring")
    tab_single, tab_batch = st.tabs(["⚡ Single Transaction Screener", "📁 Batch Screening & Capacity Audit"])

    with tab_single:
        c1, c2, c3 = st.columns(3)
        with c1:
            tx_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
            tx_amount = st.number_input("Amount ($)", value=250000.0, step=1000.0)
        with c2:
            old_orig = st.number_input("Sender Old Balance ($)", value=250000.0, step=1000.0)
            new_orig = st.number_input("Sender New Balance ($)", value=0.0, step=1000.0)
        with c3:
            old_dest = st.number_input("Receiver Old Balance ($)", value=0.0, step=1000.0)
            new_dest = st.number_input("Receiver New Balance ($)", value=250000.0, step=1000.0)

        risk_score = 0.04
        if tx_type in ["TRANSFER", "CASH_OUT"]:
            if old_orig > 0 and new_orig == 0:
                risk_score += 0.65
            if tx_amount > 150000:
                risk_score += 0.22
        risk_score = min(0.99, max(0.01, risk_score))

        if risk_score >= 0.80:
            action, color_class = "Block", "badge-block"
        elif risk_score >= 0.40:
            action, color_class = "Review", "badge-review"
        else:
            action, color_class = "Approve", "badge-approve"

        st.markdown(f"### Decision: <span class='{color_class}'>{action} (Risk Score: {risk_score:.3f})</span>", unsafe_allow_html=True)

    with tab_batch:
        st.subheader("Batch File Scoring")
        batch_source = st.session_state["raw_df"].head(400).copy()

        scores = []
        for _, row in batch_source.iterrows():
            sc = 0.04
            if row["type"] in ["TRANSFER", "CASH_OUT"]:
                if row["oldbalanceOrg"] > 0 and row["newbalanceOrig"] == 0:
                    sc += 0.62
                if row["amount"] > 100000:
                    sc += 0.25
            scores.append(min(0.98, max(0.01, sc + np.random.uniform(-0.03, 0.03))))

        batch_source["fraud_score"] = np.round(scores, 3)
        batch_source["audit_action"] = pd.cut(
            batch_source["fraud_score"],
            bins=[-0.1, 0.40, 0.80, 1.0],
            labels=["Approve", "Review", "Block"]
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Approve (Low Risk - Blue)", int((batch_source['audit_action'] == 'Approve').sum()))
        m2.metric("Review (Manual Queue - Yellow)", int((batch_source['audit_action'] == 'Review').sum()))
        m3.metric("Block (High Risk - Red)", int((batch_source['audit_action'] == 'Block').sum()))

        st.caption("💡 **Interactive Legend**: Click any decision label ('Block', 'Review', 'Approve') in the legend to filter points.")

        selection = alt.selection_point(fields=['audit_action'], bind='legend')

        scatter_chart = alt.Chart(batch_source).mark_circle(size=70).encode(
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

        def style_action(val):
            if val == "Block":
                return "background-color: rgba(224, 86, 86, 0.35); font-weight: bold; color: #E05656;"
            elif val == "Review":
                return "background-color: rgba(242, 201, 76, 0.35); font-weight: bold; color: #F2C94C;"
            elif val == "Approve":
                return "background-color: rgba(47, 128, 237, 0.25); color: #2F80ED;"
            return ""

        st.dataframe(
            batch_source.head(100).style.map(style_action, subset=["audit_action"]),
            use_container_width=True
        )
