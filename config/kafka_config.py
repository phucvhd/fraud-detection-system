from config.config_loader import ConfigLoader


class KafkaConfigLoader:
    def __init__(self, config_loader: ConfigLoader):
        self.kafka_config = config_loader.config["kafka"]
        self.kafka_producer_config = config_loader.config["kafka"]["producer"]
        self.kafka_consumer_config = config_loader.config["kafka"]["consumer"]
