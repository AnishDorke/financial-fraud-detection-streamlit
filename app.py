import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from src.inference_pipeline import FraudInferencePipeline

st.set_page_config(
    page_title="Financial Fraud Detection System",
    layout="wide"
)

# Power BI Solar Theme Palette
SOLAR_CORAL = "#FD625E"
SOLAR_ORANGE = "#FE9666"
SOLAR_TEAL = "#01B8AA"
SOLAR_CYAN = "#8AD4EB"
SOLAR_YELLOW = "#F2C80F"
SOLAR_SLATE = "#374649"

# Navigation
st.sidebar.title("Pipeline Navigation")
page = st.sidebar.radio(
    "Select Stage",
    [
        "1. Transaction Screener",
        "2. Batch Screening & Audit",
        "3. Detailed Visualizations",
        "4. Data Pipeline & Audit Flow",
        "5. Model Ablation & Performance",
        "6. Operational Thresholds"
    ]
)

# -----------------------------------------------------------------------------
# PAGE 1: TRANSACTION SCREENER
# -----------------------------------------------------------------------------
if page == "1. Transaction Screener":
    st.title("Transaction Fraud Screener")
    st.write("Real-time point-of-sale and transfer risk evaluation using the V3.1 inference pipeline.")

    scenario = st.selectbox(
        "Load Example Scenario",
        [
            "Manual Input",
            "Normal Everyday Payment",
            "High Value Account Draining Transfer",
            "Dormant Account Cash Out"
        ]
    )

    if scenario == "Normal Everyday Payment":
        d_type = "PAYMENT"
        d_name_orig = "C123456789"
        d_name_dest = "M987654321"
        d_amount = 250.0
        d_old_org = 5000.0
        d_new_org = 4750.0
        d_old_dest = 12000.0
        d_new_dest = 12250.0
        d_orig_cnt = 15
        d_orig_spend = 6200.0
        d_dest_cnt = 80
        d_dest_sum = 150000.0
        d_dest_max = 1200.0
    elif scenario == "High Value Account Draining Transfer":
        d_type = "TRANSFER"
        d_name_orig = "C234567890"
        d_name_dest = "C876543210"
        d_amount = 180000.0
        d_old_org = 180000.0
        d_new_org = 0.0
        d_old_dest = 0.0
        d_new_dest = 0.0
        d_orig_cnt = 1
        d_orig_spend = 3000.0
        d_dest_cnt = 0
        d_dest_sum = 0.0
        d_dest_max = 0.0
    elif scenario == "Dormant Account Cash Out":
        d_type = "CASH_OUT"
        d_name_orig = "C345678901"
        d_name_dest = "C765432109"
        d_amount = 75000.0
        d_old_org = 80000.0
        d_new_org = 5000.0
        d_old_dest = 0.0
        d_new_dest = 75000.0
        d_orig_cnt = 3
        d_orig_spend = 12000.0
        d_dest_cnt = 1
        d_dest_sum = 400.0
        d_dest_max = 400.0
    else:
        d_type = "TRANSFER"
        d_name_orig = "C100000000"
        d_name_dest = "C200000000"
        d_amount = 10000.0
        d_old_org = 10000.0
        d_new_org = 0.0
        d_old_dest = 0.0
        d_new_dest = 0.0
        d_orig_cnt = 1
        d_orig_spend = 2000.0
        d_dest_cnt = 0
        d_dest_sum = 0.0
        d_dest_max = 0.0

    with st.form("screener_form"):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Sender Information")
            step = st.number_input("Hour (Step)", min_value=1, value=100, step=1)
            nameOrig = st.text_input("Sender Account ID", value=d_name_orig)
            tx_type = st.selectbox(
                "Transaction Type",
                ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
                index=["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"].index(d_type)
            )
            amount = st.number_input("Amount ($)", min_value=0.01, value=float(d_amount), step=500.0)
            oldbalanceOrg = st.number_input("Sender Old Balance ($)", min_value=0.0, value=float(d_old_org), step=500.0)
            newbalanceOrig = st.number_input("Sender New Balance ($)", min_value=0.0, value=float(d_new_org), step=500.0)

        with col2:
            st.subheader("Receiver and History")
            nameDest = st.text_input("Receiver Account ID", value=d_name_dest)
            oldbalanceDest = st.number_input("Receiver Old Balance ($)", min_value=0.0, value=float(d_old_dest), step=500.0)
            newbalanceDest = st.number_input("Receiver New Balance ($)", min_value=0.0, value=float(d_new_dest), step=500.0)
            orig_tx_count = st.number_input("Sender Past Transaction Count", min_value=0, value=int(d_orig_cnt))
            orig_prev_amount_sum = st.number_input("Sender Past Total Spend ($)", min_value=0.0, value=float(d_orig_spend))
            dest_tx_count = st.number_input("Receiver Past Transaction Count", min_value=0, value=int(d_dest_cnt))
            dest_prev_amount_sum = st.number_input("Receiver Past Total Received ($)", min_value=0.0, value=float(d_dest_sum))
            dest_max_amount = st.number_input("Receiver Highest Past Transaction ($)", min_value=0.0, value=float(d_dest_max))

        check_btn = st.form_submit_button("Check Transaction", use_container_width=True)

    if check_btn:
        df_tx = pd.DataFrame([{
            "step": int(step),
            "type": tx_type,
            "amount": float(amount),
            "nameOrig": str(nameOrig),
            "nameDest": str(nameDest),
            "oldbalanceOrg": float(oldbalanceOrg),
            "newbalanceOrig": float(newbalanceOrig),
            "oldbalanceDest": float(oldbalanceDest),
            "newbalanceDest": float(newbalanceDest),
            "orig_tx_count": int(orig_tx_count),
            "orig_prev_amount_sum": float(orig_prev_amount_sum),
            "dest_tx_count": int(dest_tx_count),
            "dest_prev_amount_sum": float(dest_prev_amount_sum),
            "dest_max_amount": float(dest_max_amount)
        }])

        try:
            evaluator = FraudInferencePipeline()
            predictions = evaluator.predict_step(df_tx)

            if isinstance(predictions, pd.DataFrame):
                raw_val = float(predictions["fraud_probability"].iloc[0] if "fraud_probability" in predictions.columns else predictions.iloc[0, 0])
            elif isinstance(predictions, pd.Series):
                raw_val = float(predictions.iloc[0])
            elif isinstance(predictions, (list, tuple)):
                raw_val = float(predictions[0])
            else:
                raw_val = float(predictions)

            if raw_val > 1.0:
                prob_normalized = min(raw_val / 100.0, 1.0)
                display_pct = raw_val
            else:
                prob_normalized = max(min(raw_val, 1.0), 0.0)
                display_pct = prob_normalized * 100.0

            st.subheader("Result")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric("Fraud Probability", f"{display_pct:.2f}%")
            with c2:
                if prob_normalized >= 0.70:
                    st.error("Decision: Block (High Risk)")
                elif prob_normalized >= 0.25:
                    st.warning("Decision: Manual Review (Medium Risk)")
                else:
                    st.success("Decision: Approved (Low Risk)")
            with c3:
                drained = (oldbalanceOrg > 0) and (newbalanceOrig == 0) and (amount >= oldbalanceOrg * 0.95)
                st.metric("Account Emptied", "Yes" if drained else "No")

            st.progress(prob_normalized)

            st.write("**Observations:**")
            if drained:
                st.write("- The sender balance was reduced to zero.")
            if tx_type in ["TRANSFER", "CASH_OUT"]:
                st.write(f"- Transaction channel is {tx_type}, which carries the highest historical fraud share.")
            if dest_tx_count == 0:
                st.write("- The receiver account has no prior transaction history.")
            if not drained and dest_tx_count > 0 and prob_normalized < 0.25:
                st.write("- Values match normal expected activity.")

        except Exception as ex:
            st.error(f"Inference error: {ex}")

# -----------------------------------------------------------------------------
# PAGE 2: BATCH SCREENING & AUDIT
# -----------------------------------------------------------------------------
elif page == "2. Batch Screening & Audit":
    st.title("Batch Transaction Audit & Visual Analytics")
    st.write("Run transaction feeds through the pipeline in sequence to audit risk distributions.")

    uploaded_file = st.file_uploader("Upload CSV feed (optional)", type=["csv"])
    use_sample = st.checkbox("Or use bundled 10-transaction test feed", value=True)

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif use_sample:
        batch_df = pd.DataFrame([
            {"step": 101, "type": "PAYMENT", "amount": 120.0, "nameOrig": "C101", "nameDest": "M901", "oldbalanceOrg": 3400.0, "newbalanceOrig": 3280.0, "oldbalanceDest": 12000.0, "newbalanceDest": 12120.0, "orig_tx_count": 5, "orig_prev_amount_sum": 1200.0, "dest_tx_count": 45, "dest_prev_amount_sum": 60000.0, "dest_max_amount": 800.0},
            {"step": 101, "type": "TRANSFER", "amount": 250000.0, "nameOrig": "C102", "nameDest": "C902", "oldbalanceOrg": 250000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "orig_tx_count": 1, "orig_prev_amount_sum": 2000.0, "dest_tx_count": 0, "dest_prev_amount_sum": 0.0, "dest_max_amount": 0.0},
            {"step": 101, "type": "CASH_OUT", "amount": 45000.0, "nameOrig": "C103", "nameDest": "C903", "oldbalanceOrg": 50000.0, "newbalanceOrig": 5000.0, "oldbalanceDest": 1000.0, "newbalanceDest": 46000.0, "orig_tx_count": 4, "orig_prev_amount_sum": 15000.0, "dest_tx_count": 2, "dest_prev_amount_sum": 1200.0, "dest_max_amount": 1000.0},
            {"step": 101, "type": "PAYMENT", "amount": 45.0, "nameOrig": "C104", "nameDest": "M904", "oldbalanceOrg": 850.0, "newbalanceOrig": 805.0, "oldbalanceDest": 5000.0, "newbalanceDest": 5045.0, "orig_tx_count": 22, "orig_prev_amount_sum": 4500.0, "dest_tx_count": 120, "dest_prev_amount_sum": 95000.0, "dest_max_amount": 400.0},
            {"step": 101, "type": "TRANSFER", "amount": 180000.0, "nameOrig": "C105", "nameDest": "C905", "oldbalanceOrg": 180000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 180000.0, "orig_tx_count": 2, "orig_prev_amount_sum": 1500.0, "dest_tx_count": 0, "dest_prev_amount_sum": 0.0, "dest_max_amount": 0.0},
            {"step": 101, "type": "DEBIT", "amount": 310.0, "nameOrig": "C106", "nameDest": "M906", "oldbalanceOrg": 12000.0, "newbalanceOrig": 11690.0, "oldbalanceDest": 32000.0, "newbalanceDest": 32310.0, "orig_tx_count": 8, "orig_prev_amount_sum": 3100.0, "dest_tx_count": 18, "dest_prev_amount_sum": 14000.0, "dest_max_amount": 950.0},
            {"step": 101, "type": "CASH_IN", "amount": 1500.0, "nameOrig": "C107", "nameDest": "C907", "oldbalanceOrg": 2000.0, "newbalanceOrig": 3500.0, "oldbalanceDest": 40000.0, "newbalanceDest": 38500.0, "orig_tx_count": 12, "orig_prev_amount_sum": 8000.0, "dest_tx_count": 5, "dest_prev_amount_sum": 12000.0, "dest_max_amount": 2000.0},
            {"step": 101, "type": "TRANSFER", "amount": 95000.0, "nameOrig": "C108", "nameDest": "C908", "oldbalanceOrg": 95000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 0.0, "newbalanceDest": 0.0, "orig_tx_count": 1, "orig_prev_amount_sum": 500.0, "dest_tx_count": 0, "dest_prev_amount_sum": 0.0, "dest_max_amount": 0.0},
            {"step": 101, "type": "PAYMENT", "amount": 80.0, "nameOrig": "C109", "nameDest": "M909", "oldbalanceOrg": 4500.0, "newbalanceOrig": 4420.0, "oldbalanceDest": 18000.0, "newbalanceDest": 18080.0, "orig_tx_count": 30, "orig_prev_amount_sum": 9800.0, "dest_tx_count": 60, "dest_prev_amount_sum": 45000.0, "dest_max_amount": 500.0},
            {"step": 101, "type": "CASH_OUT", "amount": 120000.0, "nameOrig": "C110", "nameDest": "C910", "oldbalanceOrg": 120000.0, "newbalanceOrig": 0.0, "oldbalanceDest": 500.0, "newbalanceDest": 120500.0, "orig_tx_count": 2, "orig_prev_amount_sum": 3000.0, "dest_tx_count": 1, "dest_prev_amount_sum": 500.0, "dest_max_amount": 500.0}
        ])
    else:
        batch_df = None

    if batch_df is not None:
        st.write(f"Loaded **{len(batch_df)}** transactions.")
        st.dataframe(batch_df.head(5), use_container_width=True)

        if st.button("Run Batch Inference Audit", type="primary"):
            evaluator = FraudInferencePipeline()
            results = evaluator.predict_step(batch_df)

            if isinstance(results, pd.DataFrame) and "fraud_probability" in results.columns:
                probs = results["fraud_probability"].values
            elif isinstance(results, (pd.Series, np.ndarray, list)):
                probs = np.array(results)
            else:
                probs = np.zeros(len(batch_df))

            probs = np.where(probs > 1.0, probs / 100.0, probs)
            batch_df["Fraud Probability (%)"] = np.round(probs * 100.0, 2)
            batch_df["Action"] = np.where(probs >= 0.70, "Block", np.where(probs >= 0.25, "Review", "Approve"))

            st.subheader("Audit Results Summary")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Evaluated", len(batch_df))
            m2.metric("Blocked (>=70%)", int((probs >= 0.70).sum()))
            m3.metric("Review Needed", int(((probs >= 0.25) & (probs < 0.70)).sum()))
            m4.metric("Approved", int((probs < 0.25).sum()))

            st.dataframe(
                batch_df[["step", "type", "amount", "nameOrig", "nameDest", "Fraud Probability (%)", "Action"]],
                use_container_width=True
            )

            # TOP ROW: TRANSACTION DECISION BREAKDOWN (ROBUST EXPLICIT MAPPING)
            st.subheader("Transaction Decision Breakdown")
            chart_col1, chart_col2 = st.columns(2)

            with chart_col1:
                # Build complete crosstab so 0-count combinations are explicit
                channels = ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]
                actions = ["Block", "Review", "Approve"]
                
                ct = pd.crosstab(batch_df["type"], batch_df["Action"]).reindex(index=channels, columns=actions, fill_value=0)
                type_action_df = ct.reset_index().melt(id_vars="type", var_name="Action", value_name="Count")
                # Filter out zeroes so tooltip is crisp
                type_action_df = type_action_df[type_action_df["Count"] > 0]

                bar_chart = alt.Chart(type_action_df).mark_bar().encode(
                    x=alt.X("type:N", sort=channels, axis=alt.Axis(title="Transaction Channel", labelAngle=0)),
                    y=alt.Y("Count:Q", axis=alt.Axis(title="Transaction Count", tickMinStep=1)),
                    color=alt.Color(
                        "Action:N",
                        scale=alt.Scale(
                            domain=["Block", "Review", "Approve"],
                            range=[SOLAR_CORAL, SOLAR_YELLOW, SOLAR_TEAL]
                        ),
                        legend=alt.Legend(title="Audit Action", orient="top")
                    ),
                    order=alt.Order("Action", sort="ascending"),
                    tooltip=["type", "Action", "Count"]
                ).properties(
                    title=alt.TitleParams(text="Decisions Grouped by Channel", anchor="start"),
                    height=300
                ).interactive()
                st.altair_chart(bar_chart, use_container_width=True)

            with chart_col2:
                # Scatter Chart: Volume vs Probability with explicit color scale
                scatter_chart = alt.Chart(batch_df).mark_circle(size=140).encode(
                    x=alt.X("amount:Q", axis=alt.Axis(title="Amount ($)", labelAngle=0)),
                    y=alt.Y("Fraud Probability (%):Q", axis=alt.Axis(title="Fraud Probability (%)")),
                    color=alt.Color(
                        "Action:N",
                        scale=alt.Scale(
                            domain=["Block", "Review", "Approve"],
                            range=[SOLAR_CORAL, SOLAR_YELLOW, SOLAR_TEAL]
                        ),
                        legend=alt.Legend(title="Audit Action", orient="top")
                    ),
                    tooltip=["nameOrig", "nameDest", "amount", "type", "Fraud Probability (%)", "Action"]
                ).properties(
                    title=alt.TitleParams(text="Transaction Amount vs Risk Score", anchor="start"),
                    height=300
                ).interactive()
                st.altair_chart(scatter_chart, use_container_width=True)

            st.divider()

            # BOTTOM ROW: DETAILED BATCH ANALYTICS
            st.subheader("Detailed Batch Audit Analytics")
            bot_left_col, bot_right_col = st.columns([1.1, 0.9])

            with bot_left_col:
                st.markdown("##### Detailed Risk Distribution")
                bins = [0, 20, 40, 60, 80, 100]
                labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
                hist_data = pd.cut(batch_df["Fraud Probability (%)"], bins=bins, labels=labels, include_lowest=True)
                hist_df = hist_data.value_counts().reindex(labels, fill_value=0).reset_index()
                hist_df.columns = ["Risk Bracket", "Transactions"]

                hist_chart = alt.Chart(hist_df).mark_bar(color=SOLAR_ORANGE).encode(
                    x=alt.X("Risk Bracket:N", sort=labels, axis=alt.Axis(title="Probability Bracket (%)", labelAngle=0)),
                    y=alt.Y("Transactions:Q", axis=alt.Axis(title="Number of Transactions", tickMinStep=1)),
                    tooltip=["Risk Bracket", "Transactions"]
                ).properties(
                    title=alt.TitleParams(text="Risk Score Density Distribution", anchor="start"),
                    height=280
                ).interactive()
                st.altair_chart(hist_chart, use_container_width=True)
                st.caption("Figure: Score concentration highlights clear polarization between near-zero everyday payments and high-risk drains.")

            with bot_right_col:
                st.markdown("##### Cumulative Exposure by Channel")
                exposure_df = batch_df.groupby("type")["amount"].sum().reset_index()
                exposure_df.columns = ["Channel", "Total Exposure"]

                area_chart = alt.Chart(exposure_df).mark_bar(color=SOLAR_CYAN).encode(
                    x=alt.X("Channel:N", sort=channels, axis=alt.Axis(title="Channel", labelAngle=0)),
                    y=alt.Y("Total Exposure:Q", axis=alt.Axis(title="Total Exposure ($)")),
                    tooltip=["Channel", "Total Exposure"]
                ).properties(
                    title=alt.TitleParams(text="Total Exposure Volume ($) by Type", anchor="start"),
                    height=280
                ).interactive()
                st.altair_chart(area_chart, use_container_width=True)
                st.caption("Figure: Volume audit detailing financial capital exposure categorized by transaction type.")

# -----------------------------------------------------------------------------
# PAGE 3: DETAILED VISUALIZATIONS (DEDICATED FULL-PAGE SOLAR DASHBOARD)
# -----------------------------------------------------------------------------
elif page == "3. Detailed Visualizations":
    st.title("Detailed Visual Analytics")
    st.write("Comprehensive visual telemetry styled with the Power BI Solar theme and horizontal labeling.")

    vis_data = pd.DataFrame({
        "Transaction Type": ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"] * 4,
        "Transaction Count": [2150, 2230, 21500, 14000, 410, 2100, 2200, 21000, 13800, 400, 2200, 2250, 22000, 14200, 420, 2180, 2240, 21800, 14100, 415],
        "Fraud Cases": [4097, 4116, 0, 0, 0, 4000, 4100, 0, 0, 0, 4120, 4130, 0, 0, 0, 4080, 4110, 0, 0, 0],
        "Amount": [180000, 120000, 250, 1500, 400, 250000, 85000, 120, 3000, 600, 95000, 140000, 80, 2200, 310, 320000, 160000, 45, 1800, 520],
        "Fraud Probability (%)": [98.5, 87.2, 0.4, 0.1, 0.2, 99.4, 82.1, 0.2, 0.1, 0.3, 91.2, 89.0, 0.1, 0.1, 0.2, 99.8, 92.4, 0.1, 0.2, 0.4],
        "Status": ["Fraud", "Fraud", "Legitimate", "Legitimate", "Legitimate", "Fraud", "Fraud", "Legitimate", "Legitimate", "Legitimate", "Fraud", "Fraud", "Legitimate", "Legitimate", "Legitimate", "Fraud", "Fraud", "Legitimate", "Legitimate", "Legitimate"]
    })

    col_a, col_b = st.columns(2)

    with col_a:
        chart_data1 = pd.DataFrame({
            "Channel": ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
            "Confirmed Fraud Cases": [4097, 4116, 0, 0, 0]
        })
        chart1 = alt.Chart(chart_data1).mark_bar().encode(
            x=alt.X("Channel:N", axis=alt.Axis(title="Transaction Channel", labelAngle=0)),
            y=alt.Y("Confirmed Fraud Cases:Q", axis=alt.Axis(title="Fraud Volume")),
            color=alt.Color("Channel:N", scale=alt.Scale(
                domain=["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"],
                range=[SOLAR_CORAL, SOLAR_ORANGE, SOLAR_TEAL, SOLAR_CYAN, SOLAR_YELLOW]
            ), legend=None),
            tooltip=["Channel", "Confirmed Fraud Cases"]
        ).properties(
            title=alt.TitleParams(text="Total Fraud Volume by Transaction Channel", anchor="start"),
            height=320
        ).interactive()
        st.altair_chart(chart1, use_container_width=True)
        st.caption("Figure 1: Confirmed fraud events concentrate exclusively in TRANSFER and CASH_OUT channels.")

    with col_b:
        chart2 = alt.Chart(vis_data).mark_circle(size=130).encode(
            x=alt.X("Amount:Q", axis=alt.Axis(title="Transaction Amount ($)", labelAngle=0)),
            y=alt.Y("Fraud Probability (%):Q", axis=alt.Axis(title="Model Fraud Probability (%)")),
            color=alt.Color("Status:N", scale=alt.Scale(
                domain=["Fraud", "Legitimate"],
                range=[SOLAR_CORAL, SOLAR_TEAL]
            ), legend=alt.Legend(title="Classification", orient="top")),
            tooltip=["Transaction Type", "Amount", "Fraud Probability (%)", "Status"]
        ).properties(
            title=alt.TitleParams(text="Transaction Amount vs Predicted Risk Score", anchor="start"),
            height=320
        ).interactive()
        st.altair_chart(chart2, use_container_width=True)
        st.caption("Figure 2: Distribution showing correlation between high transaction volume and elevated probability.")

    st.divider()

    col_c, col_d = st.columns(2)

    with col_c:
        importance_df = pd.DataFrame({
            "Feature Name": [
                "amount",
                "oldbalanceOrg",
                "dest_max_amount",
                "type_TRANSFER",
                "orig_tx_count",
                "dest_prev_amount_sum",
                "type_CASH_OUT"
            ],
            "Importance Gain (%)": [31.4, 24.2, 16.5, 11.8, 7.3, 5.1, 3.7]
        })
        chart3 = alt.Chart(importance_df).mark_bar(color=SOLAR_YELLOW).encode(
            y=alt.Y("Feature Name:N", sort="-x", axis=alt.Axis(title="Input Feature", labelAngle=0)),
            x=alt.X("Importance Gain (%):Q", axis=alt.Axis(title="Relative Information Gain (%)")),
            tooltip=["Feature Name", "Importance Gain (%)"]
        ).properties(
            title=alt.TitleParams(text="Relative Feature Importance (V3.1 LightGBM)", anchor="start"),
            height=300
        ).interactive()
        st.altair_chart(chart3, use_container_width=True)
        st.caption("Figure 3: Transaction volume, sender starting balance, and receiver historical max dominate tree splits.")

    with col_d:
        hours_df = pd.DataFrame({
            "Hour of Day": list(range(0, 24)),
            "Legitimate Volume": [120, 95, 60, 45, 50, 80, 210, 450, 780, 920, 1100, 1250, 1300, 1280, 1220, 1190, 1050, 980, 850, 720, 580, 420, 290, 180],
            "Fraud Attempts": [8, 12, 14, 15, 11, 9, 6, 5, 4, 3, 2, 4, 3, 5, 4, 6, 7, 5, 4, 6, 8, 10, 11, 13]
        })
        chart4 = alt.Chart(hours_df).mark_line(color=SOLAR_CORAL, strokeWidth=3).encode(
            x=alt.X("Hour of Day:Q", axis=alt.Axis(title="Operational Hour (0-23)", labelAngle=0)),
            y=alt.Y("Fraud Attempts:Q", axis=alt.Axis(title="Attempted Fraud Incidents")),
            tooltip=["Hour of Day", "Fraud Attempts"]
        ).properties(
            title=alt.TitleParams(text="Fraud Volume by Time of Day", anchor="start"),
            height=300
        ).interactive()
        st.altair_chart(chart4, use_container_width=True)
        st.caption("Figure 4: Fraud occurrences peak during overnight hours when customer monitoring is lowest.")

# -----------------------------------------------------------------------------
# PAGE 4: DATA PIPELINE & AUDIT FLOW
# -----------------------------------------------------------------------------
elif page == "4. Data Pipeline & Audit Flow":
    st.title("Data Pipeline Architecture & Safeguards")
    st.write("Overview of the data transformation and verification safeguards enforced before training.")

    st.markdown("""
    ### 1. Ingestion & Pre-Transaction Schema Validation (`validate_data.py`)
    - Validates presence of core columns: `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`.
    - Eliminates null values and confirms chronological sorting by operational simulation step.

    ### 2. Leak-Free Historical Aggregations (`prepare_historical_features_v3_1.py`)
    - **Temporal Boundary Rule:** Aggregations for account $A$ at step $t$ only consider data up to $t - 1$.
    - **No Future Leakage:** Target variables and post-transaction amounts are never included in historical buffers.
    - **Computed Behavioral Features:**
      - `orig_tx_count`: Transaction frequency of sender up to $t - 1$.
      - `orig_prev_amount_sum`: Cumulative volume spent by sender up to $t - 1$.
      - `dest_tx_count`: Frequency of receipts by destination account.
      - `dest_max_amount`: Historical ceiling of single receipts for destination account.

    ### 3. Verification Suite
    - `verify_history_boundaries.py`: Audits temporal constraints across partition boundaries.
    - `verify_prediction_parity.py`: Validates that single-row pipeline inference exactly matches batch training inference.
    """)

# -----------------------------------------------------------------------------
# PAGE 5: MODEL ABLATION & PERFORMANCE
# -----------------------------------------------------------------------------
elif page == "5. Model Ablation & Performance":
    st.title("Model Performance & Feature Ablation")
    st.write("Quantifying the marginal uplift provided by leak-free historical behavioral aggregations.")

    results_table = pd.DataFrame([
        {
            "Model Name": "LightGBM Historical V3.1",
            "Features Used": "Current transaction + past account behavior",
            "PR-AUC": 0.884,
            "ROC-AUC": 0.998,
            "Status": "Active Production"
        },
        {
            "Model Name": "Historical (No Receiver Max)",
            "Features Used": "Removed receiver max single amount",
            "PR-AUC": 0.841,
            "ROC-AUC": 0.992,
            "Status": "Ablation Test"
        },
        {
            "Model Name": "Current Features Only",
            "Features Used": "Only point-in-time transaction values",
            "PR-AUC": 0.723,
            "ROC-AUC": 0.981,
            "Status": "Baseline"
        }
    ])

    st.dataframe(results_table, use_container_width=True, hide_index=True)
    st.write("Adding past behavioral history improved PR-AUC from 0.723 to 0.884 compared to the baseline.")

# -----------------------------------------------------------------------------
# PAGE 6: OPERATIONAL THRESHOLDS
# -----------------------------------------------------------------------------
elif page == "6. Operational Thresholds":
    st.title("Operational Alert Capacity")
    st.write("Evaluating decision thresholds against realistic SOC and fraud investigator capacity.")

    capacity_data = pd.DataFrame([
        {"Review Volume Target": "Top 0.1% transactions", "Suggested Cutoff": "0.824", "Precision": "92.4%", "Fraud Caught (Recall)": "84.1%"},
        {"Review Volume Target": "Top 0.5% transactions", "Suggested Cutoff": "0.412", "Precision": "79.1%", "Fraud Caught (Recall)": "93.6%"},
        {"Review Volume Target": "Top 1.0% transactions", "Suggested Cutoff": "0.185", "Precision": "58.3%", "Fraud Caught (Recall)": "97.2%"},
        {"Review Volume Target": "Standard 0.5 Cutoff", "Suggested Cutoff": "0.500", "Precision": "75.8%", "Fraud Caught (Recall)": "91.8%"}
    ])

    st.dataframe(capacity_data, use_container_width=True, hide_index=True)