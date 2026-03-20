import logging

from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class FraudPreprocessor:
    def __init__(self, config_loader: ConfigLoader):
        self.preprocessor_config = config_loader.config["preprocessor"]
        self.pipeline = None
        self.is_fitted = False

    def build_pipeline(self):
        try:
            logger.info("Building preprocessing pipeline")

            features_to_scale = self.preprocessor_config["features_to_scale"]
            features_to_keep = self.preprocessor_config["features_to_keep"]

            self.pipeline = ColumnTransformer(
                transformers=[
                    ("scaler", StandardScaler(), features_to_scale),
                    ("passthrough", "passthrough", features_to_keep)
                ],
                remainder="drop"
            )

            logger.info(f"Pipeline built")
            logger.info(f"Features to scale: {features_to_scale}")
            logger.info(f"Features to keep: {len(features_to_keep)}")
            return self.pipeline
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def fit(self, data_x: DataFrame):
        try:
            logger.info("Fitting preprocessing pipeline")

            if self.pipeline is None:
                self.build_pipeline()

            self.pipeline.fit(data_x)
            self.is_fitted = True

            logger.info("Pipeline fitted")

            scaler = self.pipeline.named_transformers_["scaler"]
            logger.info("Learned parameters:")
            logger.info(f"Time - Mean: {scaler.mean_[0]:.2f}, Std: {scaler.scale_[0]:.2f}")
            logger.info(f"Amount - Mean: {scaler.mean_[1]:.2f}, Std: {scaler.scale_[1]:.2f}")
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def transform(self, data_x: DataFrame):
        try:
            if not self.is_fitted:
                raise ValueError("Pipeline must be fitted before transform")
            return self.pipeline.transform(data_x)
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def fit_transform(self, data_x: DataFrame):
        try:
            self.fit(data_x)
            return self.transform(data_x)
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def get_feature_names(self):
        try:
            if not self.is_fitted:
                raise ValueError("Pipeline must be fitted first")
    
            features_to_scale = self.preprocessor_config["features_to_scale"]
            features_to_keep = self.preprocessor_config["features_to_keep"]
    
            scaled_names = [f"{f}_scaled" for f in features_to_scale]
    
            return scaled_names + features_to_keep
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e

    def clean_features(self, df: DataFrame):
        try:
            features_to_scale = self.preprocessor_config["features_to_scale"]
            features_to_keep = self.preprocessor_config["features_to_keep"]
            all_features = features_to_scale + features_to_keep

            cleaned_df = df[all_features]
            return cleaned_df
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e
