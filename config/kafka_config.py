from confluent_kafka import Consumer, Producer
from confluent_kafka.admin import AdminClient

from config.config_loader import ConfigLoader


class KafkaConfigLoader:
    def __init__(self, config_loader: ConfigLoader):
        self.kafka_producer_config = config_loader.config["kafka"]["producer"]
        self.kafka_consumer_config = config_loader.config["kafka"]["consumer"]

        self._producer: Producer | None = None
        self._consumer: Consumer | None = None
        self._admin: AdminClient | None = None

    @property
    def producer(self) -> Producer:
        if self._producer is None:
            cfg = self.kafka_producer_config
            self._producer = Producer({
                "bootstrap.servers": cfg["bootstrap_servers"],
                "compression.type": cfg["compression_type"],
                "linger.ms": cfg["linger_ms"],
                "batch.size": cfg["batch_size"],
                "acks": cfg["acks"],
            })
        return self._producer

    @property
    def consumer(self) -> Consumer:
        if self._consumer is None:
            cfg = self.kafka_consumer_config
            self._consumer = Consumer({
                "bootstrap.servers": cfg["bootstrap_servers"],
                "group.id": cfg["group_id"],
                "client.id": cfg["client_id"],
                "enable.auto.commit": False,
                "auto.offset.reset": "earliest",
            })
        return self._consumer

    @property
    def admin(self) -> AdminClient:
        if self._admin is None:
            self._admin = AdminClient({
                "bootstrap.servers": self.kafka_producer_config["bootstrap_servers"],
            })
        return self._admin
