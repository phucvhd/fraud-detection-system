import json
import logging
import threading
import time
from datetime import datetime
from typing import Callable, Dict, Optional

from config.kafka_config import KafkaConfigLoader

logger = logging.getLogger(__name__)

class TransactionProducer:
    def __init__(self,
                 topic: str,
                 kafka_config_loader: KafkaConfigLoader,
                 data_generator: Callable[[float], Dict]):
        self.topic = topic
        self.kafka_producer_config = kafka_config_loader.kafka_producer_config
        self.data_generator = data_generator
        self.producer = kafka_config_loader.producer
        self.running = False
        self.stats = {
            "total_sent": 0,
            "failed": 0,
            "current_rate": 0.0
        }
        self._stats_lock = threading.Lock()

    def _delivery_callback(self, err, msg):
        if err:
            logger.error(f"Delivery failed: {err}")
            logger.error(f"Error message: {msg}")
            with self._stats_lock:
                self.stats["failed"] += 1
        else:
            with self._stats_lock:
                self.stats["total_sent"] += 1

    def _calculate_delay(self, current_time: float) -> float:
        if self.kafka_producer_config["burst_mode"]:
            seconds_in_cycle = current_time % (self.kafka_producer_config["burst_interval_seconds"] * 2)
            in_burst = seconds_in_cycle < self.kafka_producer_config["burst_interval_seconds"]

            rate = (self.kafka_producer_config["transactions_per_second"] * self.kafka_producer_config["burst_multiplier"]
                    if in_burst else self.kafka_producer_config["transactions_per_second"])
        else:
            rate = self.kafka_producer_config["transactions_per_second"]

        with self._stats_lock:
            self.stats["current_rate"] = rate

        return 1.0 / rate if rate > 0 else 0

    def start_loading(self, duration_seconds: Optional[int] = None):
        self.running = True
        start_time = time.time()
        interval_time = 0

        logger.info(f"Starting load to topic '{self.topic}' at {self.kafka_producer_config["transactions_per_second"]} TPS")

        try:
            while self.running:
                current_time = time.time()
                delay = self._calculate_delay(current_time)

                if duration_seconds and (current_time - start_time) >= duration_seconds:
                    logger.info("Duration limit reached, stopping stream")
                    break

                transaction = self.data_generator(interval_time)
                interval_time += delay

                if "timestamp" not in transaction:
                    transaction["timestamp"] = datetime.now().isoformat()

                message = json.dumps(transaction).encode("utf-8")

                self.producer.produce(
                    topic=self.topic,
                    value=message,
                    key=transaction["transaction_id"],
                    callback=self._delivery_callback
                )

                self.producer.poll(0)

                time.sleep(delay)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping")
        finally:
            self.stop_loading()

    def stop_loading(self):
        self.running = False
        logger.info("Flushing remaining messages...")
        self.producer.flush(timeout=10)
