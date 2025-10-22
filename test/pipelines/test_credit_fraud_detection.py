import os

from pipelines.credit_fraud_detection import CreditFraudDetection

credit_fraud_detection = CreditFraudDetection()

def test_integration():
    credit_fraud_detection.load_train_data("../data/creditcard.csv")
    credit_fraud_detection.train_model()
    credit_fraud_detection.evaluate_trained_model()

    assert credit_fraud_detection._trained_model is not None
    assert credit_fraud_detection.precision is not None
    assert credit_fraud_detection.recall is not None

    credit_fraud_detection.save_model()

    preprocessing_pipeline = os.path.exists("models/preprocessing_pipeline.pkl")
    fraud_model = os.path.exists("models/fraud_model.pkl")
    feature_names = os.path.exists("models/feature_names.pkl")

    assert preprocessing_pipeline is True
    assert fraud_model is True
    assert feature_names is True
