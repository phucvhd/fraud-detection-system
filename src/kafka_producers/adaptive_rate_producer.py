import logging

from src.kafka_producers.transaction_producer import TransactionProducer

logger = logging.getLogger(__name__)

class AdaptiveRateProducer(TransactionProducer):
    def __init__(self, *args, min_rate: float = 10, max_rate: float = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.target_lag_ms = 100

    def adjust_rate_based_on_lag(self, current_lag_ms: float):
        if current_lag_ms > self.target_lag_ms * 2:
            new_rate = max(self.min_rate, self.kafka_producer_config.transactions_per_second * 0.8)
            logger.info(f"High lag detected ({current_lag_ms}ms), reducing rate to {new_rate}")
            self.kafka_producer_config.transactions_per_second = new_rate
        elif current_lag_ms < self.target_lag_ms * 0.5:
            new_rate = min(self.max_rate, self.kafka_producer_config.transactions_per_second * 1.2)
            logger.info(f"Low lag detected ({current_lag_ms}ms), increasing rate to {new_rate}")
            self.kafka_producer_config.transactions_per_second = new_rate
