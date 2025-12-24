from confluent_kafka import Consumer

from config.kafka_config import KafkaConfigLoader


class KafkaListener:
    def __init__(self, topic, handler, kafka_config_loader: KafkaConfigLoader):
        self.topic = topic
        self.handler = handler
        self.consumer = kafka_config_loader.consumer

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
