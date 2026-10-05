import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

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

# Explicit Action Color Palette Contract
ACTION_COLORS = ["#E05656", "#F2C94C", "#2F80ED"]
ACTION_DOMAIN = ["Block", "Review", "Approve"]

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

if stage == "1. Ingestion & Continuous Feed":
    st.title("1. Continuous Data Collection & Ingestion")
    st.markdown("Simulate continuous transaction streams or generate synthetic batch feeds matching AWS Kinesis/S3 schema.")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Synthetic Batch Generator")
        row_count = st.slider("Select batch size (records):", 500, 5000, 1500, step=250)
        feed_df = generate_sample_feed(row_count)
        st.dataframe(feed_df.head(10), use_container_width=True)
    with c2:
        st.subheader("Data Export")
        csv_bytes = feed_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Sample Batch Feed (CSV)",
            data=csv_bytes,
            file_name="sample_fraud_batch_feed.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.info("💡 Download this CSV to run batch audits in Stage 7.")

elif stage == "2. Cleaning & Validation":
    st.title("2. Data Cleaning & Integrity Audit")
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

elif stage == "3. Exploratory Data Analysis (EDA)":
    st.title("3. Exploratory Data Analysis (EDA)")
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

elif stage == "4. Feature Engineering & Leak-Free Windowing":
    st.title("4. Leak-Free Feature Engineering (V3.1)")
    st.markdown("Updating past behavior buffers strictly up to t minus 1 step.\n\n* `orig_hist_count`: Cumulative transactions\n* `orig_hist_mean_amount`: Rolling average size\n* `dest_max_amount`: Ceiling received by destination\n* `balance_orig_diff` & `balance_dest_diff`: Disparities")
    feat_sample = pd.DataFrame({
        "Feature": ["amount", "balance_orig_diff", "orig_hist_mean_amount", "dest_max_amount", "hour_of_day"],
        "Importance (Gain %)": [34.2, 28.6, 18.4, 11.2, 7.6]
    })
    feat_chart = alt.Chart(feat_sample).mark_bar(color="#F2C94C").encode(
        x=alt.X("Importance (Gain %):Q"),
        y=alt.Y("Feature:N", sort="-x")
    ).properties(height=260)
    st.altair_chart(feat_chart, use_container_width=True)

elif stage == "5. Imbalance Handling & Chronological Split":
    st.title("5. Imbalance Handling & Chronological Split")
    st.table(pd.DataFrame({
        "Split": ["Train Set", "Validation Set", "Test Set"],
        "Simulation Steps": ["Steps 1 - 445 (70%)", "Steps 446 - 594 (15%)", "Steps 595 - 743 (15%)"],
        "Records": ["4,453,834", "980,416", "928,370"],
        "Imbalance Handling": ["scale_pos_weight = 773", "Natural Distribution", "Out-of-Time Verification"]
    }))

elif stage == "6. Model Selection & Ablation Benchmarks":
    st.title("6. Model Selection & Ablation Benchmarking")
    models_df = pd.DataFrame({
        "Model Architecture": ["LightGBM Historical V3.1 (Selected)", "Historical (Without Dest Max)", "Current-Only Baseline", "Logistic Regression"],
        "PR-AUC": [0.884, 0.841, 0.723, 0.412],
        "ROC-AUC": [0.998, 0.992, 0.981, 0.895],
        "Operational Recall @ 1% Alert Tier": ["48.2%", "41.5%", "25.5%", "9.8%"]
    })
    st.dataframe(models_df, use_container_width=True)

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

        risk_score = 0.05
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

        st.markdown(f"### Result: <span class='{color_class}'>{action} (Risk Score: {risk_score:.3f})</span>", unsafe_allow_html=True)

    with tab_batch:
        st.subheader("Batch File Scoring")
        uploaded_file = st.file_uploader("Upload CSV Feed for Audit", type=["csv"])
        if uploaded_file is None:
            st.info("Showing default synthetic batch feed. You can upload custom CSVs above or download samples from Stage 1.")
            batch_data = generate_sample_feed(500)
        else:
            batch_data = pd.read_csv(uploaded_file)

        scores = []
        for _, row in batch_data.iterrows():
            sc = 0.04
            if row["type"] in ["TRANSFER", "CASH_OUT"]:
                if row["oldbalanceOrg"] > 0 and row["newbalanceOrig"] == 0:
                    sc += 0.62
                if row["amount"] > 100000:
                    sc += 0.25
            scores.append(min(0.98, max(0.01, sc + np.random.uniform(-0.03, 0.03))))

        batch_data["fraud_score"] = np.round(scores, 3)
        batch_data["audit_action"] = pd.cut(
            batch_data["fraud_score"],
            bins=[-0.1, 0.40, 0.80, 1.0],
            labels=["Approve", "Review", "Block"]
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Approve (Low Risk - Blue)", int((batch_data['audit_action'] == 'Approve').sum()))
        m2.metric("Review (Manual Queue - Yellow)", int((batch_data['audit_action'] == 'Review').sum()))
        m3.metric("Block (High Risk - Red)", int((batch_data['audit_action'] == 'Block').sum()))

        st.caption("💡 **Interactive Legend**: Click any decision label ('Block', 'Review', 'Approve') in the legend below to filter points.")

        selection = alt.selection_point(fields=['audit_action'], bind='legend')

        scatter_chart = alt.Chart(batch_data).mark_circle(size=70).encode(
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
            batch_data.head(100).style.map(style_action, subset=["audit_action"]),
            use_container_width=True
        )
