import logging

import lightgbm as lgb
import pandas as pd

from default_detection.data import preprocessing, CSV_DATA
from default_detection.model.classification_model import ClassificationModel
from default_detection.model.hyper_optimization import HyperOptimization

logger = logging.getLogger(__name__)


class ModelTrainer(object):
    """The wrapper class for handling the hyper-optimization."""
    def __init__(self, target="default", max_evals=100, max_time=600):
        self.target = target
        self.max_evals = max_evals
        self.max_time = max_time

    def train_model(self):
        """Starts hyper-optimization."""
        data, features, categorical_features = self._process_data()
        return self._train_model(data, features, categorical_features)

    def _process_data(self):
        logger.info("Reading data")
        raw_data = pd.read_csv(CSV_DATA, delimiter=";")

        logger.info("Preprocessing data")
        data, features, categorical_features = preprocessing.preprocess_dataset(raw_data)

        # Split data
        data["dataset"] = 'train'
        data.loc[data[self.target].isnull(), "dataset"] = "test"
        logger.info(f"Split: {data.dataset.value_counts()}")
        return data, features, categorical_features

    def _train_model(self, data, features, categorical_features):
        lgb_train = lgb.Dataset(
            data.query("dataset == 'train'")[features],
            data.query("dataset == 'train'")[self.target],
            feature_name=features,
            categorical_feature=categorical_features
        )

        params, scores = HyperOptimization(
            lgb_train,
            features,
            categorical_features,
            self.target
        ).hyper_search(max_evals=self.max_evals, timeout=self.max_time)

        classification_model = ClassificationModel(
            lgb_train,
            features,
            categorical_features,
            self.target,
        )

        logger.info("Training on optimized parameters")
        classification_model.train(lgb_train, params)

        logger.info("Storing results")
        classification_model.save_state(params, scores)

        return classification_model, params, scores


if __name__ == '__main__':
    trainer = ModelTrainer(max_evals=500, max_time=5400)
    trainer.train_model()
