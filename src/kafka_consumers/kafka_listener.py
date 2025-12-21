from confluent_kafka import Consumer

from config.kafka_config import KafkaConfigLoader


class KafkaListener:
    def __init__(self, topic, handler, kafka_config_loader: KafkaConfigLoader):
        self.topic = topic
        self.handler = handler

        self.kafka_consumer_config = kafka_config_loader.kafka_consumer_config
        self.consumer_bootstrap_servers = self.kafka_consumer_config["bootstrap_servers"]
        self.consumer_group_id = self.kafka_consumer_config["group_id"]
        self.consumer_client_id = self.kafka_consumer_config["client_id"]
        self.consumer = Consumer({
            "bootstrap.servers": self.consumer_bootstrap_servers,
            "enable.auto.commit": False,
            'auto.offset.reset': 'earliest',
            "group.id": self.consumer_group_id,
            'client.id': self.consumer_client_id
        })

    def start(self):
        self.consumer.subscribe([self.topic])

        while True:
            msg = self.consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                continue

            try:
                self.handler(msg.value())
                self.consumer.commit(msg)
            except Exception as e:
                raise e

    def stop(self):
        self.consumer.close()
