import json
import logging
import threading
import time
from datetime import datetime
from typing import Callable, Dict, Optional

from config.kafka_config import KafkaConfigLoader

logger = logging.getLogger(__name__)


class TransactionProducer:
    def __init__(
        self,
        topic: str,
        kafka_config_loader: KafkaConfigLoader,
        data_generator: Callable[[int], Dict],
    ):
        self.topic = topic
        self.kafka_producer_config = kafka_config_loader.kafka_producer_config
        self.data_generator = data_generator
        self.producer = kafka_config_loader.producer
        self.running = False
        self.stats = {"total_sent": 0, "failed": 0, "current_rate": 0.0}
        self._stats_lock = threading.Lock()

    def _delivery_callback(self, err, msg) -> None:
        with self._stats_lock:
            if err:
                logger.error("Delivery failed: %s", err)
                self.stats["failed"] += 1
            else:
                self.stats["total_sent"] += 1

    def _calculate_delay(self, current_time: float) -> float:
        cfg = self.kafka_producer_config
        if cfg["burst_mode"]:
            in_burst = (current_time % (cfg["burst_interval_seconds"] * 2)) < cfg["burst_interval_seconds"]
            rate = cfg["transactions_per_second"] * cfg["burst_multiplier"] if in_burst else cfg["transactions_per_second"]
        else:
            rate = cfg["transactions_per_second"]

        with self._stats_lock:
            self.stats["current_rate"] = rate

        return 1.0 / rate if rate > 0 else 0.0

    def start_loading(self, duration_seconds: Optional[int] = None) -> None:
        self.running = True
        start_time = time.time()

        now = datetime.now()
        midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elapsed_seconds = (now - midnight).total_seconds()

        logger.info("Starting producer for topic='%s' at %s TPS", self.topic, self.kafka_producer_config["transactions_per_second"])

        try:
            while self.running:
                current_time = time.time()

                if duration_seconds and (current_time - start_time) >= duration_seconds:
                    logger.info("Duration limit reached, stopping producer")
                    break

                delay = self._calculate_delay(current_time)
                transaction = self.data_generator(int(elapsed_seconds))
                elapsed_seconds += delay

                if "timestamp" not in transaction:
                    transaction["timestamp"] = datetime.now().isoformat()

                self.producer.produce(
                    topic=self.topic,
                    value=json.dumps(transaction).encode("utf-8"),
                    key=transaction["transaction_id"],
                    callback=self._delivery_callback,
                )
                self.producer.poll(0)
                time.sleep(delay)

        except KeyboardInterrupt:
            logger.info("Interrupted, stopping producer")
        finally:
            self.stop_loading()

    def stop_loading(self) -> None:
        self.running = False
        logger.info("Flushing remaining messages...")
        self.producer.flush(timeout=10)
