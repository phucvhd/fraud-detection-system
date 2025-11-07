import logging

from config.config_loader import ConfigLoader
from src.models.fraud_model import FraudModel

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
config_loader = ConfigLoader()

if __name__ == '__main__':
    trainer = FraudModel(config_loader=config_loader)
    trainer.hyper_tune()
