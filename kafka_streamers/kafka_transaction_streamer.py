import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, Optional

from confluent_kafka import Producer

logger = logging.getLogger(__name__)

@dataclass
class StreamConfig:
    transactions_per_second: float
    burst_mode: bool = False
    burst_interval_seconds: int = 60
    burst_multiplier: float = 5.0

class KafkaTransactionStreamer:
    def __init__(self,
                 bootstrap_servers: str,
                 topic: str,
                 config: StreamConfig,
                 data_generator: Callable[[float], Dict]):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.config = config
        self.data_generator = data_generator

        self.producer = Producer({
            "bootstrap.servers": bootstrap_servers,
            "compression.type": "snappy",
            "linger.ms": 10,
            "batch.size": 32768,
            "acks": 1
        })

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
        if self.config.burst_mode:
            seconds_in_cycle = current_time % (self.config.burst_interval_seconds * 2)
            in_burst = seconds_in_cycle < self.config.burst_interval_seconds

            rate = (self.config.transactions_per_second * self.config.burst_multiplier
                    if in_burst else self.config.transactions_per_second)
        else:
            rate = self.config.transactions_per_second

        with self._stats_lock:
            self.stats["current_rate"] = rate

        return 1.0 / rate if rate > 0 else 0

    def start_streaming(self, duration_seconds: Optional[int] = None):
        self.running = True
        start_time = time.time()
        interval_time = 0

        logger.info(f"Starting stream to topic '{self.topic}' at {self.config.transactions_per_second} TPS")

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

                partition_key = str(uuid.uuid4())

                self.producer.produce(
                    topic=self.topic,
                    value=message,
                    key=partition_key.encode("utf-8"),
                    callback=self._delivery_callback
                )

                self.producer.poll(0)

                time.sleep(delay)

        except KeyboardInterrupt:
            logger.info("Received interrupt signal, stopping stream")
        finally:
            self.stop_streaming()

    def stop_streaming(self):
        self.running = False
        logger.info("Flushing remaining messages...")
        self.producer.flush(timeout=10)
