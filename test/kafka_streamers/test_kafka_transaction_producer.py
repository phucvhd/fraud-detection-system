import json
import time

import pandas as pd
from confluent_kafka import Consumer

from config.config import Config
from config.kafka_config import KafkaConfig
from services.kafka_producers.kafka_transaction_producer import ProducerConfig, KafkaTransactionProducer

from services.generators.fraud_synthetic_generator import FraudSyntheticDataGenerator

test_config = Config(profile_env="test")
test_kafka_config = KafkaConfig(test_config)

test_df = pd.read_csv("test_creditcard.csv")
fraud_generator = FraudSyntheticDataGenerator(config=test_config, df=test_df)

kafka_config = KafkaConfig(config=test_kafka_config)

consumer_conf = {
    "bootstrap.servers": kafka_config.consumer_bootstrap_servers,
    "group.id": kafka_config.consumer_group_id,
    "auto.offset.reset": kafka_config.consumer_auto_offset_reset,
    "enable.auto.commit": kafka_config.consumer_enable_auto_commit,
    "client.id": kafka_config.consumer_client_id
}
consumer = Consumer(consumer_conf)

def test_start_loading():
    transaction_producer = KafkaTransactionProducer(
        topic=test_kafka_config.topic,
        kafka_config=kafka_config,
        data_generator=fraud_generator.generate_transaction
    )

    transaction_producer.start_loading(duration_seconds=5)
    time.sleep(5)

    messages = _consume_messages(5)

    transaction_producer.stop_loading()

    assert len(messages) > 0

def _consume_messages(duration):
    consumer.subscribe([test_kafka_config.topic])

    messages = []
    timeout = time.time() + duration

    while time.time() < timeout:
        msg = consumer.poll(timeout=duration)

        if msg is None:
            continue

        messages.append(msg.value().decode('utf-8'))

    consumer.close()
    return messages
