import logging

from config.config_loader import ConfigLoader
from trainers.fraud_model_trainer import FraudModelTrainer

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    config_loader = ConfigLoader()
    trainer = FraudModelTrainer(config_loader=config_loader)
    trainer.run()