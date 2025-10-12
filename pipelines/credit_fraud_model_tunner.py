import logging
import time
from datetime import datetime
from typing import List

import pandas as pd

from pipelines.credit_fraud_detection import CreditFraudDetection

logger = logging.getLogger(__name__)

class CreditFraudModelTunner:
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path
        self.n_estimators_report = pd.DataFrame(columns=["Precision", "Recall", "MAE", "n_estimators"])
        self.fraud_ratio_report = pd.DataFrame(columns=["Precision", "Recall", "MAE", "fraud_ratio"])

    def batch_training_n_estimators(self, n_estimators_list: List[int]):
        try:
            credit_fraud_detection = CreditFraudDetection()
            credit_fraud_detection.load_train_data(self.data_file_path)

            logger.info(f"Start training {len(n_estimators_list)} models")
            start_time = time.time()

            for n_estimators in n_estimators_list:
                logger.info(f"Training with n_estimators={n_estimators}")
                credit_fraud_detection.n_estimators = n_estimators
                credit_fraud_detection.train_model()
                credit_fraud_detection.evaluate_trained_model()
                new_row = {"Precision": credit_fraud_detection.precision,
                           "Recall": credit_fraud_detection.recall,
                           "MAE": credit_fraud_detection.mae,
                           "n_estimators": n_estimators}
                self.n_estimators_report = pd.concat([self.n_estimators_report, pd.DataFrame([new_row])], ignore_index=True)

            end_time = time.time()
            logger.info(f"Finish training {len(n_estimators_list)} models. Execution time: {(end_time - start_time):.4f} seconds")
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def export_n_estimators_report(self):
        try:
            if len(self.n_estimators_report) == 0:
                logger.warning("Report is empty")
                return None
            now = datetime.now()
            time_str = now.strftime("%Y%m%d_%H%M%S")
            file_name = f"n_estimators_report_{time_str}.csv"
            self.n_estimators_report.to_csv(f"reports/{file_name}", index=False)
            return file_name
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def batch_training_fraud_ratio(self, fraud_ratio_list: List[float]):
        try:
            credit_fraud_detection = CreditFraudDetection()
            credit_fraud_detection.load_train_data(self.data_file_path)

            logger.info(f"Start training {len(fraud_ratio_list)} models")
            start_time = time.time()

            for fraud_ratio in fraud_ratio_list:
                logger.info(f"Training with fraud ratio={fraud_ratio}")
                credit_fraud_detection.fraud_ratio = fraud_ratio
                credit_fraud_detection.train_model()
                credit_fraud_detection.evaluate_trained_model()
                new_row = {"Precision": credit_fraud_detection.precision,
                           "Recall": credit_fraud_detection.recall,
                           "MAE": credit_fraud_detection.mae,
                           "fraud_ratio": fraud_ratio}
                self.fraud_ratio_report = pd.concat([self.fraud_ratio_report, pd.DataFrame([new_row])],
                                                     ignore_index=True)
            end_time = time.time()
            logger.info(f"Finish training {len(fraud_ratio_list)} models. Execution time: {(end_time - start_time):.4f} seconds")
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def export_fraud_ratio_report(self):
        try:
            if len(self.fraud_ratio_report) == 0:
                logger.warning("Report is empty")
                return None
            now = datetime.now()
            time_str = now.strftime("%Y%m%d_%H%M%S")
            file_name = f"fraud_ratio_report_{time_str}.csv"
            self.fraud_ratio_report.to_csv(f"reports/{file_name}", index=False)
            return file_name
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e