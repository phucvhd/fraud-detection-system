from confluent_kafka.cimpl import Consumer, Producer
from confluent_kafka.admin import AdminClient
from config.config_loader import ConfigLoader


class KafkaConfigLoader:
    def __init__(self, config_loader: ConfigLoader):
        self.kafka_config = config_loader.config["kafka"]
        self.kafka_producer_config = config_loader.config["kafka"]["producer"]
        self.kafka_consumer_config = config_loader.config["kafka"]["consumer"]
        
        self.producer_bootstrap_servers = self.kafka_producer_config["bootstrap_servers"]
        self.producer_client_id = self.kafka_producer_config["client_id"]
        self.consumer_bootstrap_servers = self.kafka_consumer_config["bootstrap_servers"]
        self.consumer_group_id = self.kafka_consumer_config["group_id"]
        self.consumer_client_id = self.kafka_consumer_config["client_id"]
        self.consumer = Consumer({
            "bootstrap.servers": self.consumer_bootstrap_servers,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "group.id": self.consumer_group_id,
            "client.id": self.consumer_client_id
        })
        self.producer = Producer({
            "bootstrap.servers": self.kafka_producer_config["bootstrap_servers"],
            "compression.type": self.kafka_producer_config["compression_type"],
            "linger.ms": self.kafka_producer_config["linger_ms"],
            "batch.size": self.kafka_producer_config["batch_size"],
            "acks": self.kafka_producer_config["acks"]
        })
        self.admin = AdminClient({
            "bootstrap.servers": self.producer_bootstrap_servers
        })
