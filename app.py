import streamlit as st
import pandas as pd
from src.inference_pipeline import FraudInferencePipeline

st.set_page_config(
    page_title="Financial Fraud Detection System",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Transaction Screener", "Model Comparison", "Threshold Analysis"])

# -----------------------------------------------------------------------------
# PAGE 1: TRANSACTION SCREENER
# -----------------------------------------------------------------------------
if page == "Transaction Screener":
    st.title("Transaction Fraud Screener")
    st.write("Enter transaction details below to calculate the fraud risk score.")

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
            # Create fresh instance to avoid consecutive step sequence constraints
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

            # Normalize to 0.0 - 1.0 range
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
# PAGE 2: MODEL COMPARISON
# -----------------------------------------------------------------------------
elif page == "Model Comparison":
    st.title("Model Performance Comparison")
    st.write("This table compares the models trained during feature ablation testing.")

    results_table = pd.DataFrame([
        {
            "Model Name": "LightGBM Historical V3.1",
            "Features Used": "Current transaction + past account behavior",
            "PR-AUC": 0.884,
            "ROC-AUC": 0.998,
            "Status": "Active"
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
# PAGE 3: THRESHOLD ANALYSIS
# -----------------------------------------------------------------------------
elif page == "Threshold Analysis":
    st.title("Threshold and Alert Capacity")
    st.write("Evaluating thresholds based on how many alerts an operations team can review daily.")

    capacity_data = pd.DataFrame([
        {"Review Volume Target": "Top 0.1% transactions", "Suggested Cutoff": "0.824", "Precision": "92.4%", "Fraud Caught (Recall)": "84.1%"},
        {"Review Volume Target": "Top 0.5% transactions", "Suggested Cutoff": "0.412", "Precision": "79.1%", "Fraud Caught (Recall)": "93.6%"},
        {"Review Volume Target": "Top 1.0% transactions", "Suggested Cutoff": "0.185", "Precision": "58.3%", "Fraud Caught (Recall)": "97.2%"},
        {"Review Volume Target": "Standard 0.5 Cutoff", "Suggested Cutoff": "0.500", "Precision": "75.8%", "Fraud Caught (Recall)": "91.8%"}
    ])

    st.dataframe(capacity_data, use_container_width=True, hide_index=True)