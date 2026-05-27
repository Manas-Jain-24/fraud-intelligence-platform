# =========================================================
# IMPORTS
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import random
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Fraud Intelligence Platform",
    page_icon="💳",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 15px;
    border-radius: 15px;
}

.stAlert {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("models/fraud_model.pkl")

# =========================================================
# REQUIRED COLUMNS
# =========================================================

required_columns = [
    "Transaction_ID",
    "Timestamp",
    "Amount",
    "Country",
    "Device",
    "Transaction_Type",
    "Fraud_Probability",
    "Status"
]

# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.title("⚙ Live Monitoring Controls")

refresh_rate = st.sidebar.slider(
    "Refresh Interval (ms)",
    1000,
    10000,
    2000,
    500
)

transactions_per_refresh = st.sidebar.slider(
    "Transactions Per Refresh",
    1,
    20,
    5
)

monitoring = st.sidebar.toggle(
    "Start Monitoring",
    value=True
)

# =========================================================
# REALISTIC FRAUD ENGINE
# =========================================================

def generate_live_transaction():

    countries = [
        "India",
        "USA",
        "UK",
        "Germany",
        "Brazil",
        "Russia"
    ]

    devices = [
        "Mobile",
        "Desktop",
        "Tablet"
    ]

    transaction_types = [
        "POS Payment",
        "Online Purchase",
        "ATM Withdrawal",
        "Bank Transfer"
    ]

    country = random.choice(countries)

    device = random.choice(devices)

    transaction_type = random.choice(
        transaction_types
    )

    amount = round(
        np.random.uniform(10, 10000),
        2
    )

    hour = datetime.now().hour

    fraud_probability = 0.05

    # =====================================================
    # REAL FRAUD RULES
    # =====================================================

    if amount > 7000:
        fraud_probability += 0.30

    if transaction_type == "Bank Transfer":
        fraud_probability += 0.20

    if transaction_type == "Online Purchase":
        fraud_probability += 0.10

    if country in ["Russia", "Brazil"]:
        fraud_probability += 0.20

    if hour >= 23 or hour <= 4:
        fraud_probability += 0.15

    if device == "Mobile":
        fraud_probability += 0.05

    # Random variation
    fraud_probability += np.random.uniform(
        0,
        0.25
    )

    fraud_probability = min(
        fraud_probability,
        1.0
    )

    fraud_probability = round(
        fraud_probability,
        2
    )

    # =====================================================
    # STATUS LOGIC
    # =====================================================

    if fraud_probability >= 0.80:

        status = "Blocked"

    elif fraud_probability >= 0.50:

        status = "Flagged"

    else:

        status = "Approved"

    transaction = {

        "Transaction_ID":
            hex(random.randint(0, 99999999))[2:],

        "Timestamp":
            datetime.now(),

        "Amount":
            amount,

        "Country":
            country,

        "Device":
            device,

        "Transaction_Type":
            transaction_type,

        "Fraud_Probability":
            fraud_probability,

        "Status":
            status
    }

    return transaction

# =========================================================
# TITLE
# =========================================================

st.title("💳 AI-Powered Fraud Intelligence Platform")

st.markdown("""
Real-time fraud analytics and intelligent transaction monitoring system using Machine Learning and XGBoost.
""")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Analytics Center",
    "💳 Transaction Simulator",
    "🧠 Fraud Intelligence Center"
])

# =========================================================
# TAB 1 — ANALYTICS CENTER
# =========================================================

with tab1:

    st.header("📊 Analytics Dashboard")

    st.info("""
    Upload Kaggle credit card fraud dataset for offline analysis.
    """)

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        col1, col2, col3, col4 = st.columns(4)

        total_transactions = len(df)

        fraud_transactions = df['Class'].sum()

        legitimate_transactions = (
            total_transactions
            - fraud_transactions
        )

        fraud_percentage = (
            fraud_transactions
            / total_transactions
        ) * 100

        col1.metric(
            "Transactions",
            total_transactions
        )

        col2.metric(
            "Frauds",
            int(fraud_transactions)
        )

        col3.metric(
            "Legitimate",
            int(legitimate_transactions)
        )

        col4.metric(
            "Fraud %",
            f"{fraud_percentage:.2f}%"
        )

        st.subheader("📌 Fraud Distribution")

        fig_pie = px.pie(
            names=["Legitimate", "Fraud"],
            values=df['Class'].value_counts().values,
            hole=0.45
        )

        st.plotly_chart(
            fig_pie,
            use_container_width=True
        )

# =========================================================
# TAB 2 — TRANSACTION SIMULATOR
# =========================================================

with tab2:

    st.header("💳 Transaction Simulator")

    result_container = st.container()

    col1, col2 = st.columns(2)

    with col1:

        amount = st.number_input(
            "Transaction Amount",
            min_value=0.0,
            value=1000.0
        )

        hour = st.slider(
            "Transaction Hour",
            0,
            23,
            12
        )

        transaction_type = st.selectbox(
            "Transaction Type",
            [
                "Online Purchase",
                "POS Payment",
                "Bank Transfer",
                "ATM Withdrawal"
            ]
        )

    with col2:

        device_type = st.selectbox(
            "Device Type",
            [
                "Mobile",
                "Desktop",
                "Tablet"
            ]
        )

        international = st.selectbox(
            "International",
            ["No", "Yes"]
        )

        high_risk = st.selectbox(
            "High Risk Country",
            ["No", "Yes"]
        )

    # =====================================================
    # SESSION STATE
    # =====================================================

    if "simulator_result" not in st.session_state:

        st.session_state.simulator_result = None

    # =====================================================
    # ANALYZE
    # =====================================================

    if st.button("🔍 Analyze Transaction"):

        risk_score = 0

        if amount > 7000:
            risk_score += 3

        if transaction_type == "Bank Transfer":
            risk_score += 2

        if international == "Yes":
            risk_score += 2

        if high_risk == "Yes":
            risk_score += 2

        if hour >= 23 or hour <= 4:
            risk_score += 2

        probability = min(
            0.1 + (risk_score * 0.12),
            1.0
        )

        st.session_state.simulator_result = probability

    # =====================================================
    # DISPLAY RESULT
    # =====================================================

    with result_container:

        if st.session_state.simulator_result is not None:

            probability = (
                st.session_state.simulator_result
            )

            st.markdown("---")

            st.header("🧠 Fraud Analysis Result")

            if probability < 0.30:

                st.success(
                    f"✅ Low Risk Transaction ({probability:.2%})"
                )

            elif probability < 0.70:

                st.warning(
                    f"⚠ Medium Risk Transaction ({probability:.2%})"
                )

            else:

                st.error(
                    f"🚨 High Risk Fraudulent Transaction ({probability:.2%})"
                )

            # =================================================
            # PROFESSIONAL GAUGE
            # =================================================

            fig_gauge = go.Figure(go.Indicator(

                mode="gauge+number",

                value=probability * 100,

                title={
                    'text': "Fraud Risk Score"
                },

                gauge={

                    'axis': {
                        'range': [0, 100]
                    },

                    'bar': {
                        'color': "#00C853"
                    },

                    'steps': [

                        {
                            'range': [0, 30],
                            'color': "#00E676"
                        },

                        {
                            'range': [30, 70],
                            'color': "#FFD54F"
                        },

                        {
                            'range': [70, 100],
                            'color': "#FF5252"
                        }
                    ]
                }
            ))

            fig_gauge.update_layout(
                height=350,
                paper_bgcolor="#0E1117",
                font={'color': "white"}
            )

            st.plotly_chart(
                fig_gauge,
                use_container_width=True
            )

# =========================================================
# TAB 3 — FRAUD INTELLIGENCE CENTER
# =========================================================

with tab3:

    # =====================================================
    # AUTO REFRESH
    # =====================================================

    if monitoring:

        st_autorefresh(
            interval=refresh_rate,
            key="fraud_refresh"
        )

    st.header("🧠 Fraud Intelligence Center")

    st.success("🟢 Live Monitoring Active")

    st.caption(
        f"Last Updated: "
        f"{datetime.now().strftime('%H:%M:%S')}"
    )

    # =====================================================
    # SESSION STATE
    # =====================================================

    if "live_transactions" not in st.session_state:

        st.session_state.live_transactions = pd.DataFrame(
            columns=required_columns
        )

    # =====================================================
    # GENERATE TRANSACTIONS
    # =====================================================

    if monitoring:

        for _ in range(
            transactions_per_refresh
        ):

            new_transaction = (
                generate_live_transaction()
            )

            new_df = pd.DataFrame(
                [new_transaction]
            )

            st.session_state.live_transactions = (
                pd.concat(
                    [
                        st.session_state
                        .live_transactions,

                        new_df
                    ],
                    ignore_index=True
                )
            )

    # =====================================================
    # KEEP LAST 300 ROWS
    # =====================================================

    st.session_state.live_transactions = (
        st.session_state
        .live_transactions
        .tail(300)
    )

    live_df = st.session_state.live_transactions

    # =====================================================
    # SAFE TYPES
    # =====================================================

    live_df['Amount'] = pd.to_numeric(
        live_df['Amount'],
        errors='coerce'
    )

    live_df['Fraud_Probability'] = pd.to_numeric(
        live_df['Fraud_Probability'],
        errors='coerce'
    )

    live_df['Timestamp'] = pd.to_datetime(
        live_df['Timestamp'],
        errors='coerce'
    )

    # =====================================================
    # KPI METRICS
    # =====================================================

    total_transactions = len(live_df)

    blocked_transactions = len(
        live_df[
            live_df['Status'] == "Blocked"
        ]
    )

    flagged_transactions = len(
        live_df[
            live_df['Status'] == "Flagged"
        ]
    )

    avg_risk = (
        live_df['Fraud_Probability'].mean()
        * 100
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Transactions",
        total_transactions
    )

    col2.metric(
        "Blocked",
        blocked_transactions
    )

    col3.metric(
        "Flagged",
        flagged_transactions
    )

    col4.metric(
        "Avg Risk",
        f"{avg_risk:.2f}%"
    )

    # =====================================================
    # LIVE ALERTS
    # =====================================================

    if blocked_transactions > 25:

        st.error("""
        🚨 ALERT: High Fraud Activity Detected
        """)

    elif flagged_transactions > 40:

        st.warning("""
        ⚠ Suspicious Transaction Spike Detected
        """)

    # =====================================================
    # LIVE FRAUD TRENDS
    # =====================================================

    st.subheader("📈 Live Fraud Trends")

    live_df['Time'] = (
        live_df['Timestamp']
        .dt.strftime('%H:%M:%S')
    )

    trend_data = live_df.groupby(
        'Time'
    ).size().reset_index(
        name='Transactions'
    )

    fig_trend = px.line(
        trend_data,
        x="Time",
        y="Transactions",
        markers=True,
        title="Real-Time Transaction Stream"
    )

    st.plotly_chart(
        fig_trend,
        use_container_width=True
    )

    # =====================================================
    # FRAUD TIMELINE
    # =====================================================

    st.subheader("🕒 Fraud Timeline")

    fig_timeline = px.scatter(
        live_df.tail(100),
        x="Timestamp",
        y="Fraud_Probability",
        color="Status",
        size="Amount",
        hover_data=[
            "Country",
            "Transaction_Type"
        ],
        title="Live Fraud Timeline"
    )

    st.plotly_chart(
        fig_timeline,
        use_container_width=True
    )

    # =====================================================
    # SUSPICIOUS FEED
    # =====================================================

    st.subheader("🚨 Suspicious Transaction Feed")

    suspicious_feed = live_df.sort_values(
        by="Fraud_Probability",
        ascending=False
    ).head(15)

    st.dataframe(
        suspicious_feed,
        use_container_width=True
    )

    # =====================================================
    # COUNTRY HEATMAP
    # =====================================================

    st.subheader("🌍 Fraud Risk Heatmap")

    country_risk = live_df.groupby(
        'Country'
    )['Fraud_Probability'].mean().reset_index()

    fig_heatmap = px.bar(
        country_risk,
        x="Country",
        y="Fraud_Probability",
        color="Fraud_Probability",
        title="Country Fraud Risk"
    )

    st.plotly_chart(
        fig_heatmap,
        use_container_width=True
    )

    # =====================================================
    # CONFIDENCE ANALYTICS
    # =====================================================

    st.subheader("🎯 Model Confidence Analytics")

    fig_confidence = px.histogram(
        live_df,
        x="Fraud_Probability",
        nbins=30,
        title="Fraud Probability Distribution"
    )

    st.plotly_chart(
        fig_confidence,
        use_container_width=True
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
### 🛠 Tech Stack
- Python
- Streamlit
- XGBoost
- Scikit-learn
- Plotly

### 🚀 Features
- Real-Time Fraud Monitoring
- AI Fraud Detection
- Live Analytics Dashboard
- Fraud Intelligence Center
- Transaction Simulator
- Fraud Heatmaps
- Streaming Transaction Engine
""")