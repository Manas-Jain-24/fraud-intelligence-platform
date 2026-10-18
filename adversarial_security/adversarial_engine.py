"""
adversarial_security/adversarial_engine.py
==========================================
Fintech Adversarial Evasion & Red-Teaming Engine.
Simulates real-world cyber fraud syndicate attacks:
1. Fast Gradient Sign Perturbation (FGSM) on mutable financial vectors.
2. Minimal Boundary Evasion: Calculates the exact perturbation budget epsilon* required to evade detection.
3. Smurfing / Transaction Splitting Simulation: Decomposes large fraudulent sums into micro-transactions.
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models_engine import FraudIntelligenceEngine, engine


class AdversarialRedTeamEngine:
    """
    Evaluates adversarial vulnerability of machine learning fraud detection models.
    """
    # Mutable transaction features that fraudsters can realistically manipulate:
    # Amount (index 28), V4 (velocity proxy, index 3), V11 (terminal sequence, index 10)
    MUTABLE_FEATURE_INDICES = [3, 10, 28]

    def __init__(self, target_engine: FraudIntelligenceEngine = engine):
        self.engine = target_engine
        if not self.engine.is_loaded:
            self.engine.load()

    def fgsm_perturbation(
        self,
        x_raw: np.ndarray,
        epsilon: float = 0.5,
        mutable_only: bool = True
    ) -> np.ndarray:
        """
        Applies Fast Gradient Sign Method (FGSM) perturbation to decrease fraud probability:
        x_adv = x - epsilon * sign(grad_x)
        """
        x_adv = x_raw.copy()
        x_norm = self.engine.scale_features(x_raw)

        # Gradient of binary cross-entropy with respect to normalized input x
        # p = sigmoid(w^T x + b) -> dL/dx = (p - 1) * w (since target is y=0 legitimate)
        model = self.engine.models["Baseline Logistic Regression (NumPy)"]
        w = model.w
        p = model.predict_proba(x_norm)

        # Direction that reduces fraud probability
        grad_direction = - np.sign(w)

        # Convert perturbation budget epsilon back to raw scale
        std_scale = self.engine.scaler_std
        perturbation = epsilon * grad_direction * std_scale

        if mutable_only:
            # Only apply to realistic attacker-controllable features
            mask = np.zeros(len(x_raw), dtype=bool)
            mask[self.MUTABLE_FEATURE_INDICES] = True
            perturbation = np.where(mask, perturbation, 0.0)

        x_adv = x_raw + perturbation
        # Enforce non-negative transaction amount
        x_adv[28] = max(1.0, float(x_adv[28]))
        return x_adv

    def find_evasion_budget(
        self,
        x_fraud: np.ndarray,
        model_name: str = "Soft-Voting Ensemble (Scratch Champion)",
        target_threshold: float = 0.40,
        max_epsilon: float = 3.0,
        steps: int = 50
    ) -> Dict[str, Any]:
        """
        Performs binary search to find the minimal perturbation budget epsilon*
        needed to drop fraud score below target_threshold (turning BLOCKED into APPROVED).
        """
        init_scores = self.engine.predict_all(x_fraud)
        init_prob = init_scores[model_name]

        if init_prob < target_threshold:
            return {
                "initial_prob": init_prob,
                "evaded": True,
                "epsilon_star": 0.0,
                "final_prob": init_prob,
                "perturbed_vector": x_fraud
            }

        eps_grid = np.linspace(0.01, max_epsilon, steps)
        best_eps = None
        best_adv_x = None
        best_prob = init_prob

        for eps in eps_grid:
            x_adv = self.fgsm_perturbation(x_fraud, epsilon=eps, mutable_only=True)
            scores = self.engine.predict_all(x_adv)
            current_prob = scores[model_name]

            if current_prob < target_threshold:
                best_eps = float(eps)
                best_adv_x = x_adv
                best_prob = float(current_prob)
                break
            elif current_prob < best_prob:
                best_prob = float(current_prob)
                best_adv_x = x_adv

        evaded = best_eps is not None
        return {
            "initial_prob": float(init_prob),
            "evaded": evaded,
            "epsilon_star": best_eps if evaded else float(max_epsilon),
            "final_prob": float(best_prob),
            "amount_delta": float(best_adv_x[28] - x_fraud[28]) if best_adv_x is not None else 0.0,
            "perturbed_vector": best_adv_x if best_adv_x is not None else x_fraud
        }

    def simulate_smurfing_attack(
        self,
        base_fraud_vector: np.ndarray,
        split_count: int = 5
    ) -> Dict[str, Any]:
        """
        Simulates structured transaction splitting ("smurfing"):
        Takes a single large extraction ($4,500) and splits it into N micro-transactions,
        evaluating whether individual split transactions fly under the detection radar.
        """
        total_amount = float(base_fraud_vector[28])
        split_amount = round(total_amount / split_count, 2)

        # Baseline single transaction evaluation
        single_scores = self.engine.predict_all(base_fraud_vector)
        single_blocked = single_scores["Soft-Voting Ensemble (Scratch Champion)"] >= 0.75

        # Split transaction evaluations
        sub_probs = []
        evaded_count = 0

        for _ in range(split_count):
            sub_x = base_fraud_vector.copy()
            sub_x[28] = split_amount
            # Slightly lower velocity indicator due to smaller amount footprint
            sub_x[3] = max(-1.0, float(sub_x[3]) * 0.7)
            sub_score = self.engine.predict_all(sub_x)["Soft-Voting Ensemble (Scratch Champion)"]
            sub_probs.append(float(sub_score))
            if sub_score < 0.40:
                evaded_count += 1

        evasion_rate = evaded_count / split_count
        return {
            "original_amount": total_amount,
            "split_count": split_count,
            "split_amount": split_amount,
            "original_risk": float(single_scores["Soft-Voting Ensemble (Scratch Champion)"]),
            "original_blocked": single_blocked,
            "average_split_risk": float(np.mean(sub_probs)),
            "evasion_success_rate": float(evasion_rate),
            "is_smurfing_successful": bool(evasion_rate > 0.5)
        }


# Global red-team engine
red_team = AdversarialRedTeamEngine()
