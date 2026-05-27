import pandas as pd
import numpy as np
from datetime import datetime
import random
import uuid
import time
import os

# =========================================================
# CREATE CSV IF NOT EXISTS
# =========================================================

file_path = "data/live_transactions.csv"

if not os.path.exists(file_path):

    df = pd.DataFrame(columns=[
        "Transaction_ID",
        "Timestamp",
        "Amount",
        "Country",
        "Device",
        "Transaction_Type",
        "Fraud_Probability",
        "Status"
    ])

    df.to_csv(file_path, index=False)

# =========================================================
# DATA OPTIONS
# =========================================================

countries = [
    "India",
    "USA",
    "UK",
    "Russia",
    "Brazil",
    "Germany",
    "China"
]

devices = [
    "Mobile",
    "Desktop",
    "Tablet"
]

transaction_types = [
    "Online Purchase",
    "POS Payment",
    "ATM Withdrawal",
    "Bank Transfer"
]

statuses = [
    "Approved",
    "Flagged",
    "Blocked",
    "Under Review"
]

# =========================================================
# GENERATE TRANSACTIONS
# =========================================================

while True:

    transaction_id = str(uuid.uuid4())[:8]

    timestamp = datetime.now()

    amount = round(
        np.random.uniform(100, 10000),
        2
    )

    country = random.choice(countries)

    device = random.choice(devices)

    transaction_type = random.choice(
        transaction_types
    )

    # =====================================================
    # FRAUD LOGIC
    # =====================================================

    fraud_probability = np.random.uniform(
        0,
        1
    )

    if amount > 7000:
        fraud_probability += 0.2

    if country in ["Russia", "Brazil"]:
        fraud_probability += 0.2

    if transaction_type == "ATM Withdrawal":
        fraud_probability += 0.1

    fraud_probability = min(
        fraud_probability,
        1.0
    )

    # =====================================================
    # STATUS
    # =====================================================

    if fraud_probability > 0.8:

        status = "Blocked"

    elif fraud_probability > 0.6:

        status = "Under Review"

    elif fraud_probability > 0.4:

        status = "Flagged"

    else:

        status = "Approved"

    # =====================================================
    # CREATE ROW
    # =====================================================

    new_transaction = pd.DataFrame({
        "Transaction_ID": [transaction_id],
        "Timestamp": [timestamp],
        "Amount": [amount],
        "Country": [country],
        "Device": [device],
        "Transaction_Type": [transaction_type],
        "Fraud_Probability": [
            round(fraud_probability, 2)
        ],
        "Status": [status]
    })

    # =====================================================
    # APPEND TO CSV
    # =====================================================

    new_transaction.to_csv(
        file_path,
        mode='a',
        header=False,
        index=False
    )

    print(
        f"Transaction Generated: "
        f"{transaction_id}"
    )

    # =====================================================
    # WAIT
    # =====================================================

    time.sleep(2)


# What This Does

#Every 2 seconds:

#generates new transaction
#calculates fraud probability
#assigns status
#appends to CSV

#This creates:

#persistent streaming fraud data