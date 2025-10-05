import time

import pandas as pd

from config.config import Config
from kafka_streamers.kafka_transaction_streamer import StreamConfig, KafkaTransactionStreamer

from services.generators.fraud_synthetic_generator import FraudSyntheticDataGenerator

test_config = Config(profile_env="test")
test_df = pd.read_csv("test_creditcard.csv")
fraud_generator = FraudSyntheticDataGenerator(config=test_config, df=test_df)

stream_config = StreamConfig(
        transactions_per_second=test_config.transactions_per_second,
        burst_mode=test_config.burst_mode,
        burst_interval_seconds=test_config.burst_interval_seconds,
        burst_multiplier=test_config.burst_multiplier,
        partition_key_field=test_config.partition_key_field
    )

def test_start_streaming():
    streamer = KafkaTransactionStreamer(
        bootstrap_servers=test_config.bootstrap_servers,
        topic=test_config.topic,
        config=stream_config,
        data_generator=fraud_generator.generate_transaction
    )

    streamer.start_streaming(duration_seconds=5)
    time.sleep(10)
    streamer.stop_streaming()

