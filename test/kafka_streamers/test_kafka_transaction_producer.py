import time

import pandas as pd
from confluent_kafka import Consumer

from config.config_loader import ConfigLoader
from config.kafka_config import KafkaConfigLoader
from src.generators.fraud_synthetic_generator import FraudSyntheticDataGenerator
from src.kafka_producers.kafka_transaction_producer import KafkaTransactionProducer

config_loader = ConfigLoader(path="application-test.yaml")
test_kafka_config_loader = KafkaConfigLoader(config_loader)

test_df = pd.read_csv("test_creditcard.csv")
fraud_generator = FraudSyntheticDataGenerator(config_loader=config_loader, df=test_df)

kafka_config_loader = KafkaConfigLoader(config_loader=config_loader)

consumer = Consumer({
    "bootstrap.servers": kafka_config_loader.kafka_producer_config["bootstrap_servers"],
    "group.id": kafka_config_loader.kafka_producer_config["group_id"],
    "auto.offset.reset": kafka_config_loader.kafka_producer_config["auto_offset_reset"],
    "enable.auto.commit": kafka_config_loader.kafka_producer_config["enable_auto_commit"],
    "client.id": kafka_config_loader.kafka_producer_config["client_id"]
})

def test_start_loading():
    transaction_producer = KafkaTransactionProducer(
        topic=kafka_config_loader.kafka_producer_config["topic"],
        kafka_config_loader=kafka_config_loader,
        data_generator=fraud_generator.generate_transaction
    )

    transaction_producer.start_loading(duration_seconds=5)
    time.sleep(5)

    messages = _consume_messages(5)

    transaction_producer.stop_loading()

    assert len(messages) > 0

def _consume_messages(duration):
    consumer.subscribe([kafka_config_loader.kafka_producer_config["topic"]])

    messages = []
    timeout = time.time() + duration

    while time.time() < timeout:
        msg = consumer.poll(timeout=duration)

        if msg is None:
            continue

        messages.append(msg.value().decode('utf-8'))

    consumer.close()
    return messages
