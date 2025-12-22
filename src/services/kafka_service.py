import json
import logging
import time

from confluent_kafka import KafkaError
from confluent_kafka.cimpl import NewTopic

from config.kafka_config import KafkaConfigLoader

logger = logging.getLogger(__name__)

class KafkaService:
    def __init__(self, kafka_config_loader: KafkaConfigLoader):
        self.kafka_producer_config = kafka_config_loader.kafka_producer_config
        self.kafka_consumer_config = kafka_config_loader.kafka_consumer_config
        self.consumer = kafka_config_loader.consumer
        self.producer = kafka_config_loader.producer
        self.admin = kafka_config_loader.admin

    def list_all_topics(self):
        try:
            metadata = self.admin.list_topics(timeout=5)
            logger.info("Connected to Kafka")
            logger.info("Topics:", list(metadata.topics.keys()))

            return list(metadata.topics.keys())
        except Exception as e:
            logger.error("Cannot connect to Kafka", str(e))
            
    def send_message(self, topic: str, key: str, message: str):
        try:
            self.producer.produce(
                topic,
                key=key,
                value=message,
                callback=self.delivery_report
            )
            self.producer.flush(timeout=10)
        except Exception as e:
            logger.error(f"Cannot send message to topic={topic}", str(e))

    def consume_topic(self, topic: str, consume_time: int):
        try:
            self.consumer.subscribe([topic])

            logger.info(f"Listening for messages (timeout: {consume_time} seconds)...")

            message_count = 0
            messages = []
            timeout = time.time() + consume_time

            while time.time() < timeout:
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.info("Reached end of partition")
                        continue
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        break

                try:
                    message_value = json.loads(msg.value().decode("utf-8"))
                    logger.info(f"Received: {message_value}")
                    message_count += 1
                    messages.append(str(message_value))
                    break
                except json.JSONDecodeError as e:
                    logger.info(f"Failed to decode message: {e}")
                    logger.info(f"Raw message: {msg.value()}")

            self.consumer.close()

            if message_count > 0:
                logger.info(f"Consumer test successful! Read {message_count} messages")
            else:
                logger.warning("No messages found, but consumer connected successfully")

            return messages
        except Exception as e:
            logger.error(f"Cannot consume message from topic={topic}", str(e))
            raise e

    def delivery_report(self, err, msg):
        if err is not None:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.info(f"Message delivered to {msg.topic()} partition {msg.partition()} offset {msg.offset()}")

    def create_topic(self, topic_name: str):
        try:
            self.admin.create_topics(
                [NewTopic(topic_name, num_partitions=1, replication_factor=1)]
            )
        except Exception as e:
            logger.error(f"Cannot create topic={topic_name}", str(e))
            raise e

    def delete_topic(self, topic_name: str):
        try:
            self.admin.delete_topics([topic_name], operation_timeout=10)
        except Exception as e:
            logger.error(f"Cannot delete topic={topic_name}", str(e))
            raise e