"""
data_generator.py
=================
Realistic synthetic transaction streaming engine for the Fraud Intelligence Platform.
Generates transactions with genuine statistical PCA properties (V1-V28, Amount, Time)
along with realistic banking metadata (Merchant, Country, Channel, Device, Card Type).
"""

import time
import random
import uuid
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List, Optional
from models_engine import FraudIntelligenceEngine, engine

# Synthetic metadata categories
COUNTRIES = [
    ("United States", "USA", "North America", 0.05),
    ("United Kingdom", "GBR", "Europe", 0.04),
    ("Germany", "DEU", "Europe", 0.03),
    ("India", "IND", "Asia", 0.06),
    ("Singapore", "SGP", "Asia", 0.03),
    ("Brazil", "BRA", "South America", 0.12),
    ("Nigeria", "NGA", "Africa", 0.15),
    ("Russia", "RUS", "Europe", 0.14),
    ("Australia", "AUS", "Oceania", 0.03),
    ("Canada", "CAN", "North America", 0.04)
]

MERCHANTS = [
    ("Amazon Marketplace", "E-Commerce", 0.04),
    ("Apple App Store", "Digital Goods", 0.03),
    ("Walmart Supercenter", "Retail POS", 0.02),
    ("Binance Crypto Exchange", "Crypto / Web3", 0.22),
    ("Shell Gas & Petroleum", "Automated Fuel", 0.05),
    ("Western Union Transfer", "Remittance / Wire", 0.18),
    ("Steam Gaming", "Digital Gaming", 0.08),
    ("Uber Technologies", "Rideshare", 0.03),
    ("Luxury Watch Direct", "High-Value Goods", 0.25),
    ("Starbucks Coffee", "Food & Beverage", 0.01)
]

CHANNELS = ["E-Commerce (Web)", "Mobile In-App", "Contactless POS", "ATM Withdrawal", "Wire Transfer"]
DEVICES = ["iOS Safari", "Android Chrome", "Windows Desktop", "MacOS Firefox", "Point-of-Sale Terminal"]
CARD_TYPES = ["Visa Infinite", "Mastercard World Elite", "Amex Platinum", "Discover Cashback", "Virtual Single-Use Card"]


class LiveTransactionGenerator:
    """
    Simulates a high-throughput bank card transaction stream.
    Samples realistic statistical vectors (V1-V28, Amount) from calibrated normal vs fraud clusters.
    """
    def __init__(self, fraud_rate: float = 0.08):
        self.fraud_rate = fraud_rate
        # Calibrated distributions for normal vs fraud
        # Fraud transactions typically have strong negative shifts in V14, V10, V12, V17 and positive in V4, V11
        np.random.seed(int(time.time()) % 10000)

    def generate_single_transaction(self, force_fraud: Optional[bool] = None) -> Dict[str, Any]:
        """Generates one realistic transaction with all 29 mathematical features and rich metadata."""
        is_fraud = force_fraud if force_fraud is not None else (random.random() < self.fraud_rate)

        # Generate 28 PCA features
        if not is_fraud:
            # Genuine transaction: standard normal with subtle covariance
            v_vec = np.random.normal(loc=0.0, scale=1.0, size=28)
            amount = round(float(np.random.exponential(scale=65.0) + 2.50), 2)
            merchant_info = random.choices(MERCHANTS, weights=[1-m[2] for m in MERCHANTS])[0]
            country_info = random.choices(COUNTRIES, weights=[1-c[3] for c in COUNTRIES])[0]
            channel = random.choice(CHANNELS[:3])  # POS, E-commerce, In-app
        else:
            # Fraud attack pattern: characteristic PCA drift
            v_vec = np.random.normal(loc=0.0, scale=1.3, size=28)
            # Signature fraud eigenvector shifts
            v_vec[3] = np.random.normal(loc=3.2, scale=1.0)   # V4 high positive
            v_vec[9] = np.random.normal(loc=-3.8, scale=1.2)  # V10 negative
            v_vec[10] = np.random.normal(loc=2.8, scale=0.9)  # V11 positive
            v_vec[11] = np.random.normal(loc=-4.1, scale=1.1) # V12 negative
            v_vec[13] = np.random.normal(loc=-5.4, scale=1.3) # V14 extreme negative
            v_vec[16] = np.random.normal(loc=-3.5, scale=1.2) # V17 negative
            # Fraud amount is often either micro-carding (< $3) or high extraction ($900 - $8000)
            if random.random() < 0.65:
                amount = round(float(np.random.uniform(750.0, 4800.0)), 2)
            else:
                amount = round(float(np.random.uniform(1.0, 8.5)), 2)
            merchant_info = random.choices(MERCHANTS, weights=[m[2] for m in MERCHANTS])[0]
            country_info = random.choices(COUNTRIES, weights=[c[3] for c in COUNTRIES])[0]
            channel = random.choice([CHANNELS[0], CHANNELS[3], CHANNELS[4]])  # E-commerce, ATM, Wire

        tx_id = f"TX-{uuid.uuid4().hex[:8].upper()}"
        timestamp = datetime.now()
        card_type = random.choice(CARD_TYPES)
        device = random.choice(DEVICES)

        # Build raw feature vector: V1..V28, Amount
        raw_x = np.zeros(29)
        raw_x[:28] = v_vec
        raw_x[28] = amount

        # Execute live inference via master engine
        model_scores = engine.predict_all(raw_x)
        champion_prob = model_scores["Soft-Voting Ensemble (Scratch Champion)"]

        # Explainability
        xai_report = engine.explain_transaction(raw_x)

        # Decision Status based on champion risk score
        if champion_prob >= 0.75:
            status = "BLOCKED"
            action_code = "HARD_DECLINE"
            status_color = "#EF4444"
        elif champion_prob >= 0.40:
            status = "FLAGGED"
            action_code = "STEP_UP_MFA"
            status_color = "#F59E0B"
        else:
            status = "APPROVED"
            action_code = "INSTANT_SETTLE"
            status_color = "#10B981"

        return {
            "tx_id": tx_id,
            "timestamp": timestamp.strftime("%H:%M:%S.%f")[:-4],
            "datetime_obj": timestamp,
            "amount": amount,
            "country": country_info[0],
            "country_code": country_info[1],
            "region": country_info[2],
            "merchant": merchant_info[0],
            "merchant_category": merchant_info[1],
            "channel": channel,
            "device": device,
            "card_type": card_type,
            "is_ground_truth_fraud": is_fraud,
            "fraud_probability": champion_prob,
            "all_model_scores": model_scores,
            "status": status,
            "status_color": status_color,
            "action_code": action_code,
            "top_drivers": xai_report["top_drivers"],
            "narrative": xai_report["narrative"],
            "raw_features": raw_x
        }

    def generate_batch(self, count: int = 10) -> List[Dict[str, Any]]:
        """Generates a batch of transactions."""
        return [self.generate_single_transaction() for _ in range(count)]


# Global generator
generator = LiveTransactionGenerator(fraud_rate=0.08)

if __name__ == "__main__":
    tx = generator.generate_single_transaction()
    print("Generated live transaction:")
    print(f"  ID: {tx['tx_id']} | Amount: ${tx['amount']:.2f} | Status: {tx['status']} ({tx['fraud_probability']:.2%})")
    print(f"  Merchant: {tx['merchant']} ({tx['country']})")
    print(f"  Top Driver: {tx['top_drivers'][0] if tx['top_drivers'] else 'None'}")
    print(f"  Narrative: {tx['narrative']}")
