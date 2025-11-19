import os

from archives.credit_fraud_model_tunner import CreditFraudModelTunner

model_tunner = CreditFraudModelTunner("../data/creditcard.csv")

def test_batch_training_n_estimators():
    n_estimators_list = list(range(50, 151, 10))
    model_tunner.batch_training_n_estimators(n_estimators_list=n_estimators_list)
    assert len(model_tunner.n_estimators_report) == len(n_estimators_list)

    file_name = model_tunner.export_n_estimators_report()
    assert file_name is not None
    assert os.path.exists(f"reports/{file_name}") is True

def test_batch_training_fraud_ratio():
    fraud_ratio_list = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    model_tunner.batch_training_fraud_ratio(fraud_ratio_list=fraud_ratio_list)
    assert len(model_tunner.fraud_ratio_report) == len(fraud_ratio_list)

    file_name = model_tunner.export_fraud_ratio_report()
    assert file_name is not None
    assert os.path.exists(f"reports/{file_name}") is True