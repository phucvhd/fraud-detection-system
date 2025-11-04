import logging

from config.config_loader import ConfigLoader
from src.trainers.fraud_model_trainer import FraudModelTrainer

logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
config_loader = ConfigLoader()

if __name__ == '__main__':
    trainer = FraudModelTrainer(config_loader=config_loader)
    trainer.run()
