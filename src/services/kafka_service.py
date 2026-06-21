import json
import logging
import time

from confluent_kafka import KafkaError
from confluent_kafka.cimpl import NewTopic

from config.kafka_config import KafkaConfigLoader

logger = logging.getLogger(__name__)


class KafkaService:
    def __init__(self, kafka_config_loader: KafkaConfigLoader):
        self.consumer = kafka_config_loader.consumer
        self.producer = kafka_config_loader.producer
        self.admin = kafka_config_loader.admin

    def list_all_topics(self) -> list[str]:
        try:
            metadata = self.admin.list_topics(timeout=5)
            topics = list(metadata.topics.keys())
            logger.info("Listed %d Kafka topics", len(topics))
            return topics
        except Exception:
            logger.error("Failed to list Kafka topics", exc_info=True)
            raise

    def send_message(self, topic: str, key: str, message) -> None:
        try:
            if isinstance(message, dict):
                message = json.dumps(message).encode("utf-8")
            elif isinstance(message, str):
                message = message.encode("utf-8")
            self.producer.produce(topic, key=key, value=message, callback=self.delivery_report)
            self.producer.flush(timeout=10)
        except Exception:
            logger.error("Failed to send message to topic=%s", topic, exc_info=True)
            raise

    def consume_topic(self, topic: str, consume_time: int) -> list:
        messages = []
        try:
            self.consumer.subscribe([topic])
            deadline = time.time() + consume_time
            while time.time() < deadline:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    logger.error("Consumer error: %s", msg.error())
                    break
                try:
                    messages.append(json.loads(msg.value().decode("utf-8")))
                    break
                except json.JSONDecodeError:
                    logger.warning("Failed to decode message from topic=%s", topic)
        finally:
            self.consumer.close()

        logger.info("Consumed %d message(s) from topic=%s", len(messages), topic)
        return messages

    def create_topic(self, topic_name: str) -> None:
        try:
            self.admin.create_topics([NewTopic(topic_name, num_partitions=1, replication_factor=1)])
            logger.info("Created topic=%s", topic_name)
        except Exception:
            logger.error("Failed to create topic=%s", topic_name, exc_info=True)
            raise

    def delete_topic(self, topic_name: str) -> None:
        try:
            self.admin.delete_topics([topic_name], operation_timeout=10)
            logger.info("Deleted topic=%s", topic_name)
        except Exception:
            logger.error("Failed to delete topic=%s", topic_name, exc_info=True)
            raise

    def delivery_report(self, err, msg) -> None:
        if err is not None:
            logger.error("Message delivery failed: %s", err)
        else:
            logger.info(
                "Message delivered to %s partition %d offset %d",
                msg.topic(), msg.partition(), msg.offset(),
            )
