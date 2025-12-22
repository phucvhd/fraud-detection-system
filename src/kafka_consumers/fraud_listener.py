from config.config_loader import ConfigLoader
from config.kafka_config import KafkaConfigLoader
from src.kafka_consumers.kafka_listener import KafkaListener


class FraudListener:
    def __init__(self, config_loader: ConfigLoader, handler):
        self.config_loader = config_loader
        self.fraud_detection_config = config_loader.config["api"]["fraud_detection"]

    def start_listener(self, handler):
        is_toggle_on = bool(self.fraud_detection_config["kafka"]["listener_toggle"])
        if is_toggle_on:
            kafka_config_loader = KafkaConfigLoader(self.config_loader)
            input_topic = self.fraud_detection_config["kafka"]["topic"]
            kafka_listener = KafkaListener(input_topic, handler, kafka_config_loader)
            kafka_listener.start()
