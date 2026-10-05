import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# --- PAGE CONFIGURATION & SOLAR THEME SETUP ---
st.set_page_config(
    page_title="Financial Fraud Detection System (AWS ML Architecture)",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Solar Palette)
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

# Fixed Color Palette Contract
ACTION_COLORS = {
    "Block": "#E05656",     # Red
    "Review": "#F2C94C",    # Amber / Yellow
    "Approve": "#2F80ED"    # Blue
}

action_color_scale = alt.Scale(
    domain=["Block", "Review", "Approve"],
    range=["#E05656", "#F2C94C", "#2F80ED"]
)

# --- REUSABLE SYNTHETIC DATA GENERATOR ---
@st.cache_data
def generate_sample_feed(n_rows: int = 1500) -> pd.DataFrame:
    np.random.seed(42)
    steps = np.random.randint(1, 15, size=n_rows)
    types = np.random.choice(["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"], size=n_rows, p=[0.35, 0.10, 0.35, 0.05, 0.15])
    amounts = np.round(np.random.exponential(scale=50000, size=n_rows) + 10, 2)
    orig_ids = [f"C{np.random.randint(100000, 999999)}" for _ in range(n_rows)]
    dest_ids = [f"M{np.random.randint(100000, 999999)}" if t == "PAYMENT" else f"C{np.random.randint(100000, 999999)}" for t in types]
    old_orig = np.round(np.random.exponential(scale=60000, size=n_rows), 2)
    new_orig = np.maximum(0, old_orig - amounts)
    old_dest = np.round(np.random.exponential(scale=40000, size=n_rows), 2)
    new_dest = old_dest + amounts

    # Inject fraud patterns (CASH_OUT / TRANSFER draining)
    fraud_idx = np.random.choice(n_rows, size=max(1, int(n_rows * 0.035)), replace=False)
    for i in fraud_idx:
        types[i] = "TRANSFER" if np.random.rand() > 0.5 else "CASH_OUT"
        amounts[i] = old_orig[i] = np.random.uniform(200000, 850000)
        new_orig[i] = 0.0

    return pd.DataFrame({
        "step": steps,
        "type": types,
        "amount": amounts,
        "nameOrig": orig_ids,
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "nameDest": dest_ids,
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest
    })

# --- PIPELINE NAVIGATION SIDEBAR ---
st.sidebar.title("Pipeline Navigation")
stage = st.sidebar.radio(
    "Lifecycle Stages:",
    [
        "1. Ingestion & Continuous Feed",
        "2. Cleaning & Validation",
        "3. Exploratory Data Analysis (EDA)",
        "4. Feature Engineering & Leak-Free Windowing",
        "5. Imbalance Handling & Chronological Split",
        "6. Model Selection & Ablation Benchmarks",
        "7. Real-Time Fraud Prediction & Monitoring"
    ]
)

# -------------------------------------------------------------
# STAGE 1: CONTINUOUS DATA INGESTION & TEST BATCH GENERATOR
# -------------------------------------------------------------
if stage == "1. Ingestion & Continuous Feed":
    st.title("1. Continuous Data Collection & Ingestion")
    st.markdown("Simulate continuous transaction streams or generate synthetic batch feeds matching AWS Kinesis/S3 schema.")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Synthetic Batch Generator")
        st.write("Generate a fresh transaction feed with injected fraud patterns for pipeline testing.")
        row_count = st.slider("Select batch size (records):", 500, 5000, 1500, step=250)
        feed_df = generate_sample_feed(row_count)
        st.dataframe(feed_df.head(10), use_container_width=True)

    with c2:
        st.subheader("Data Export")
        st.markdown("**Download Sample Dataset:**")
        csv_bytes = feed_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Sample Batch Feed (CSV)",
            data=csv_bytes,
            file_name="sample_fraud_batch_feed.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.info("💡 Download this CSV to run batch audits in Stage 7.")

# -------------------------------------------------------------
# STAGE 2: CLEANING & VALIDATION AUDIT
# -------------------------------------------------------------
elif stage == "2. Cleaning & Validation":
    st.title("2. Data Cleaning & Integrity Audit")
    st.markdown("Automated schema verification, missing value scans, and boundary validation.")

    v_col1, v_col2, v_col3, v_col4 = st.columns(4)
    v_col1.metric("Total Records Audited", "6,362,620")
    v_col2.metric("Missing / Null Values", "0 (0.0%)")
    v_col3.metric("Negative Amounts Filtered", "0")
    v_col4.metric("Schema Status", "100% Validated")

    st.subheader("Dataset Schema Contract")
    schema_data = {
        "Column": ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest", "oldbalanceDest", "newbalanceDest", "isFraud"],
        "Expected Type": ["int64 (Hour)", "string", "float64", "string (C...)", "float64", "float64", "string (C... / M...)", "float64", "float64", "int64 (0/1)"],
        "Status": ["PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS"]
    }
    st.table(pd.DataFrame(schema_data))

# -------------------------------------------------------------
# STAGE 3: EXPLORATORY DATA ANALYSIS (EDA)
# -------------------------------------------------------------
elif stage == "3. Exploratory Data Analysis (EDA)":
    st.title("3. Exploratory Data Analysis (EDA)")
    st.markdown("Investigating class distribution, fraudulent transaction channels, and diurnal trends.")

    e1, e2 = st.columns(2)
    with e1:
        st.subheader("Fraud by Transaction Channel")
        channel_df = pd.DataFrame({
            "Channel": ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
            "Confirmed Fraud": [4097, 4116, 0, 0, 0]
        })
        chart_channel = alt.Chart(channel_df).mark_bar(color="#F2994A").encode(
            x=alt.X("Channel:N", sort="-y"),
            y=alt.Y("Confirmed Fraud:Q", title="Fraud Count")
        ).properties(height=300)
        st.altair_chart(chart_channel, use_container_width=True)
        st.caption("Key insight: Fraud is exclusively concentrated in TRANSFER and CASH_OUT mechanisms.")

    with e2:
        st.subheader("Class Imbalance Ratio")
        imbalance_df = pd.DataFrame({
            "Category": ["Legitimate (99.87%)", "Fraud (0.13%)"],
            "Count": [6354407, 8213]
        })
        donut = alt.Chart(imbalance_df).mark_arc(innerRadius=60).encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Category:N", scale=alt.Scale(range=["#2F80ED", "#E05656"]))
        ).properties(height=300)
        st.altair_chart(donut, use_container_width=True)

# -------------------------------------------------------------
# STAGE 4: FEATURE ENGINEERING & HISTORICAL WINDOWING
# -------------------------------------------------------------
elif stage == "4. Feature Engineering & Leak-Free Windowing":
    st.title("4. Leak-Free Feature Engineering (V3.1)")
    st.markdown("Eliminating temporal leakage by updating past customer behavior buffers strictly up to $t-1$.")

    st.markdown("""
    * **No Target / Current Step Leakage**: Current transaction amounts and balances are never folded into historical lookups before scoring step $t$.
    * **Engineered Signals**:
        1. `orig_hist_count`: Cumulative transactions originating from sender.
        2. `orig_hist_mean_amount`: Rolling average transaction size.
        3. `dest_max_amount`: Historical ceiling received by destination account.
        4. `balance_orig_diff` & `balance_dest_diff`: Immediate account drain disparities.
    """)

    feat_sample = pd.DataFrame({
        "Feature": ["amount", "balance_orig_diff", "orig_hist_mean_amount", "dest_max_amount", "hour_of_day"],
        "Importance (Gain %)": [34.2, 28.6, 18.4, 11.2, 7.6]
    })
    feat_chart = alt.Chart(feat_sample).mark_bar(color="#F2C94C").encode(
        x=alt.X("Importance (Gain %):Q"),
        y=alt.Y("Feature:N", sort="-x")
    ).properties(height=260)
    st.altair_chart(feat_chart, use_container_width=True)

# -------------------------------------------------------------
# STAGE 5: IMBALANCE HANDLING & CHRONOLOGICAL SPLIT
# -------------------------------------------------------------
elif stage == "5. Imbalance Handling & Chronological Split":
    st.title("5. Imbalance Handling & Chronological Split")
    st.markdown("Preventing future lookahead bias using chronological step partitions.")

    st.table(pd.DataFrame({
        "Split": ["Train Set", "Validation Set", "Test Set"],
        "Simulation Steps": ["Steps 1 – 445 (70%)", "Steps 446 – 594 (15%)", "Steps 595 – 743 (15%)"],
        "Records": ["4,453,834", "980,416", "928,370"],
        "Imbalance Handling": ["scale_pos_weight = 773", "Natural Distribution", "Out-of-Time Verification"]
    }))

# -------------------------------------------------------------
# STAGE 6: MODEL SELECTION & ABLATION BENCHMARKS
# -------------------------------------------------------------
elif stage == "6. Model Selection & Ablation Benchmarks":
    st.title("6. Model Selection & Ablation Benchmarking")
    st.markdown("Precision-Recall AUC (PR-AUC) comparison across candidate configurations.")

    models_df = pd.DataFrame({
        "Model Architecture": ["LightGBM Historical V3.1 (Selected)", "Historical (Without Dest Max)", "Current-Only Baseline", "Logistic Regression"],
        "PR-AUC": [0.884, 0.841, 0.723, 0.412],
        "ROC-AUC": [0.998, 0.992, 0.981, 0.895],
        "Operational Recall @ 1% Alert Tier": ["48.2%", "41.5%", "25.5%", "9.8%"]
    })
    st.dataframe(models_df, use_container_width=True)

# -------------------------------------------------------------
# STAGE 7: REAL-TIME FRAUD PREDICTION & BATCH MONITORING
# -------------------------------------------------------------
elif stage == "7. Real-Time Fraud Prediction & Monitoring":
    st.title("7. Fraud Prediction & Operational Monitoring")
    st.markdown("Screen live point-of-sale transactions or audit high-throughput batch feeds.")

    tab_single, tab_batch = st.tabs(["⚡ Single Transaction Screener", "📁 Batch Screening & Capacity Audit"])

    with tab_single:
        c1, c2, c3 = st.columns(3)
        with c1:
            tx_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
            tx_amount = st.number_input("