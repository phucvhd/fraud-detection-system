import logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, recall_score, precision_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)

class HyperTuner:
    def __init__(self, config_loader: ConfigLoader):
        self.model_config = config_loader.config["model"]
        self.hyperparameter_tuning = config_loader.config["tuner"]["hyperparameter_tuning"]
        self.param_grid = config_loader.config["tuner"]["param_grid"]
        self.tuner_strategy = self.hyperparameter_tuning["strategy"]
        self.grid_search_params = self.param_grid["grid_search"]
        self.random_search_params = self.param_grid["random_search"]
        self.evaluation = config_loader.config["evaluation"]
        self.scoring = self._init_scoring()

    def _init_scoring(self):
        scoring = {}
        for metric in self.evaluation["metrics"]:
            if metric == "recall":
                scoring["recall"] = make_scorer(recall_score)
            elif metric == "precision":
                scoring["precision"] = make_scorer(precision_score)
            elif metric == "pr_auc":
                scoring["pr_auc"] = "average_precision"
        return scoring

    def init_tuner(self):
        try:
            model_type = self.model_config["type"]
            logger.info(f"Initiate tuner for {model_type} model")
            match model_type:
                case "decision_tree":
                    estimator = DecisionTreeClassifier(class_weight="balanced")
                case "random_forest":
                    estimator = RandomForestClassifier()
                case "xgboost":
                    estimator = XGBClassifier()
                case _:
                    raise Exception("Model type is invalid")

            logger.info(f"Applying {self.tuner_strategy} strategy")
            search = None
            if self.tuner_strategy == "grid_search":
                search = GridSearchCV(
                    estimator=estimator,
                    param_grid=self.grid_search_params,
                    scoring=self.scoring,
                    refit=self.evaluation["primary_metric"],
                    cv=self.hyperparameter_tuning["cv"],
                    n_jobs=self.hyperparameter_tuning["n_jobs"],
                    verbose=self.hyperparameter_tuning["verbose"]
                )
            elif self.tuner_strategy == "random_search":
                search = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=self.random_search_params,
                    n_iter=self.hyperparameter_tuning["n_iter"],
                    scoring=self.scoring,
                    refit=self.evaluation["primary_metric"],
                    cv=self.hyperparameter_tuning["cv"],
                    n_jobs=self.hyperparameter_tuning["n_jobs"],
                    verbose=self.hyperparameter_tuning["verbose"],
                    random_state=self.hyperparameter_tuning["random_state"]
                )

            if search is None:
                raise Exception(f"Tuner strategy {self.tuner_strategy} is invalid")
            return search
        except Exception as e:
            logger.error("Unexpected error: ", e)
            raise e
