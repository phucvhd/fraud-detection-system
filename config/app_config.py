import os
from typing import List, Dict, Literal, Optional, Union

import yaml
from pydantic import BaseModel, Field

class KafkaProducerConfig(BaseModel):
    transaction_per_second: int
    burst_mode: bool
    burst_interval_seconds: int
    burst_multiplier: float
    bootstrap_servers: str
    topic: str
    client_id: str
    compression_type: str
    linger_ms: int
    batch_size: int
    acks: Union[int, str]


class KafkaConsumerConfig(BaseModel):
    bootstrap_servers: str
    group_id: str
    auto_offset_reset: Literal["earliest", "latest"]
    enable_auto_commit: bool
    client_id: str


class KafkaConfig(BaseModel):
    producer: KafkaProducerConfig
    consumer: KafkaConsumerConfig

class FraudGeneratorConfig(BaseModel):
    max_transaction_time: int
    fraud_rate: float

class RandomForestParams(BaseModel):
    n_estimators: int
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    max_features: str


class XGBoostParams(BaseModel):
    n_estimators: int
    max_depth: int
    learning_rate: float
    random_state: int
    n_jobs: int
    eval_metric: str


class DecisionTreeParams(BaseModel):
    criterion: str
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int
    random_state: int


class ModelParams(BaseModel):
    random_forest: RandomForestParams
    xgboost: XGBoostParams
    decision_tree: DecisionTreeParams


class ModelConfig(BaseModel):
    train_ratio: float
    val_ratio: float
    type: Literal["random_forest", "xgboost", "decision_tree"]
    params: ModelParams

class ImbalanceConfig(BaseModel):
    method: Literal["smote", "undersample", "oversample"]
    sampling_strategy: float
    random_state: int

class PreprocessorConfig(BaseModel):
    features_to_scale: List[str]
    features_to_keep: List[str]

class EvaluationConfig(BaseModel):
    primary_metric: Literal["precision", "recall", "f1", "pr_auc"]
    metrics: List[str]
    threshold: float

class GridSearchParams(BaseModel):
    n_estimators: List[int]
    max_depth: List[int]
    min_samples_split: List[int]


class RandomSearchParams(BaseModel):
    n_estimators: List[int]
    max_depth: List[int]
    min_samples_split: List[int]
    min_samples_leaf: List[int]
    max_features: List[str]


class ParamGrid(BaseModel):
    grid_search: GridSearchParams
    random_search: RandomSearchParams


class HyperparameterTuningConfig(BaseModel):
    strategy: Literal["grid_search", "random_search"]
    cv: int
    n_jobs: int
    verbose: int
    random_state: int
    n_iter: int


class TunerConfig(BaseModel):
    hyperparameter_tuning: HyperparameterTuningConfig
    param_grid: ParamGrid

class OutputConfig(BaseModel):
    model_path: str
    report_path: str

class AppConfig(BaseModel):
    kafka: KafkaConfig
    fraud_generator: FraudGeneratorConfig
    model: ModelConfig
    imbalance: ImbalanceConfig
    preprocessor: PreprocessorConfig
    evaluation: EvaluationConfig
    tuner: TunerConfig
    output: OutputConfig


    @classmethod
    def from_yaml(cls, filename: str = "application.yaml") -> "AppConfig":
        base_dir = os.path.dirname(__file__)
        config_path = os.path.join(base_dir, "..", "config", filename)
        config_path = os.path.abspath(config_path)

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)