import os

from config.config import Config


class KafkaConfig(Config):
    def __init__(self, config: Config):
        super().__init__(config)
        self.transactions_per_second = int(os.getenv("KAFKA_PRODUCER_TRANSACTION_PER_SECOND", 100))
        self.burst_mode = os.getenv("KAFKA_PRODUCER_BURST_MODE", "False").lower() == "true"
        self.burst_interval_seconds = int(os.getenv("KAFKA_PRODUCER_BURST_INTERVAL_SECONDS", 30))
        self.burst_multiplier = float(os.getenv("KAFKA_PRODUCER_BURST_MULTIPLIER", 1.0))
        self.producer_bootstrap_servers = os.getenv("KAFKA_PRODUCER_BOOTSTRAP_SERVERS", "localhost:9092").strip('"')
        self.topic = os.getenv("KAFKA_PRODUCER_TOPIC", "transactions").strip('"')
        self.producer_client_id = os.getenv("KAFKA_PRODUCER_CLIENT_ID", "fraud-detection-producer").strip('"')
        self.compression_type = os.getenv("KAFKA_PRODUCER_COMPRESSION_TYPE", "snappy").strip('"')
        self.linger_ms = int(os.getenv("KAFKA_PRODUCER_LINGER_MS", 10))
        self.batch_size = int(os.getenv("KAFKA_PRODUCER_BATCH_SIZE", 32768))
        self.acks = int(os.getenv("KAFKA_PRODUCER_ACKS", 1))

        self.consumer_bootstrap_servers = os.getenv("KAFKA_CONSUMER_BOOTSTRAP_SERVERS", "localhost:9092").strip('"')
        self.consumer_group_id = os.getenv("KAFKA_CONSUMER_GROUP_ID", "fraud-detection-consumer-group").strip('"')
        self.consumer_auto_offset_reset = os.getenv("KAFKA_CONSUMER_AUTO_OFFSET_RESET", "earliest").strip('"')
        self.consumer_enable_auto_commit = os.getenv("KAFKA_CONSUMER_ENABLE_AUTO_COMMIT", "False").lower() == "true"
        self.consumer_client_id = os.getenv("KAFKA_CONSUMER_CLIENT_ID", "fraud-detection-consumer").strip('"')
