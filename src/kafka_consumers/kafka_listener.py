import logging
import threading

from confluent_kafka import Consumer

from config.kafka_config import KafkaConfigLoader

logger = logging.getLogger(__name__)


class KafkaListener:
    def __init__(
        self,
        topic: str,
        handler,
        kafka_config_loader: KafkaConfigLoader,
        batch_size: int = 100,
        poll_timeout: float = 1.0,
    ):
        self.topic = topic
        self.handler = handler
        self.consumer = kafka_config_loader.consumer
        self.batch_size = batch_size
        self.poll_timeout = poll_timeout
        self._stop_event = threading.Event()

    def start(self) -> None:
        self.consumer.subscribe([self.topic])
        logger.info("Kafka listener started on topic=%s (batch_size=%d)", self.topic, self.batch_size)

        try:
            while not self._stop_event.is_set():
                # consume() drains up to batch_size messages already queued locally
                # (or whatever arrives within poll_timeout), so handler() amortizes
                # its per-call cost (DataFrame construction, model inference) across
                # many transactions instead of paying it once per message.
                msgs = self.consumer.consume(num_messages=self.batch_size, timeout=self.poll_timeout)
                if not msgs:
                    continue

                valid_msgs = []
                for msg in msgs:
                    if msg.error():
                        logger.error("Consumer error on topic=%s: %s", self.topic, msg.error())
                        continue
                    valid_msgs.append(msg)

                if not valid_msgs:
                    continue

                try:
                    self.handler([m.value() for m in valid_msgs])
                    # Assumes a single partition (this topic's dev/default setup):
                    # committing the last message's offset covers the whole batch.
                    # A multi-partition deployment would need a per-partition commit.
                    self.consumer.commit(message=valid_msgs[-1])
                except Exception:
                    logger.error(
                        "Failed to process batch of %d message(s) from topic=%s",
                        len(valid_msgs),
                        self.topic,
                        exc_info=True,
                    )
        finally:
            # confluent_kafka.Consumer is NOT thread-safe: close it on the same
            # thread that polls, never from the caller of stop().
            self.consumer.close()
            logger.info("Kafka listener stopped on topic=%s", self.topic)

    def stop(self) -> None:
        # Only signal the polling thread to exit; it closes the consumer itself.
        self._stop_event.set()
