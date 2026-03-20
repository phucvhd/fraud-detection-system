import logging

import numpy as np
from pandas import DataFrame
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score, confusion_matrix, \
    precision_recall_curve

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class FraudModelEvaluator:
    def __init__(self, config_loader: ConfigLoader):
        self.evaluator_config = config_loader.config["evaluation"]
        self.threshold = self.evaluator_config["threshold"]

    def evaluate(self, model, data_x, data_y):
        logger.info(f"Evaluating model")

        data_y_pred = model.predict(data_x)
        data_y_proba = model.predict_proba(data_x)[:, 1]

        precision = precision_score(data_y, data_y_pred)
        recall = recall_score(data_y, data_y_pred)
        f1 = f1_score(data_y, data_y_pred)
        pr_auc = average_precision_score(data_y, data_y_proba)

        cm = confusion_matrix(data_y, data_y_pred)
        tn, fp, fn, tp = cm.ravel()

        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        logger.info(f"PR-AUC: {pr_auc:.4f}")

        logger.info(f"Confusion Matrix:")
        logger.info(f"True Negatives: {tn}")
        logger.info(f"False Positives: {fp}")
        logger.info(f"False Negatives: {fn}")
        logger.info(f"True Positives: {tp}")

        logger.info(f"Business Metrics:")
        logger.info(f"Fraud detected: {tp}/{tp + fn} ({tp / (tp + fn) * 100:.1f}%)")
        logger.info(f"Fraud missed: {fn}/{tp + fn} ({fn / (tp + fn) * 100:.1f}%)")
        logger.info(f"False alarms: {fp}")

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "pr_auc": pr_auc,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn
        }

    def find_optimal_threshold(self, model, data_x: DataFrame, data_y: DataFrame):
        logger.info("Finding optimal threshold")

        data_y_proba = model.predict_proba(data_x)[:, 1]

        precision, recall, thresholds = precision_recall_curve(data_y, data_y_proba)

        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

        logger.info(f"Optimal threshold: {best_threshold:.4f}")

        return best_threshold