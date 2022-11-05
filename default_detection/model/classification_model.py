import datetime
import json
import logging
import os
import uuid

import dill
import lightgbm as lgb

from default_detection import DATA_DIR

logger = logging.getLogger(__name__)


class ClassificationModel(object):
    def __init__(self, data=None, features=None, categorical_features=None, target="default"):
        self.data = data
        self.features = features
        self.categorical_features = categorical_features
        self.target = target
        self.model = None
        self.datetime_str = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        self.id = str(uuid.uuid4())[:8]

    @property
    def dir(self):
        """Directory where the model and metadata get stored."""
        destination_dir = os.path.join(DATA_DIR, self.datetime_str + "-" + self.id)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
        return destination_dir

    @property
    def estimator_path(self):
        """Path where the final model is stored."""
        return os.path.join(self.dir, "model.pk")

    @property
    def metadata_path(self):
        """Path where the model metadata is stored."""
        return os.path.join(self.dir, "model-metadata.json")

    def train(self, lgb_train, params):
        logger.info("Starting model training on complete data set...")
        self.model = lgb.train(
            params,
            lgb_train,
            feature_name=self.features,
            categorical_feature=self.categorical_features,
            verbose_eval=False,
        )
        return self.model

    def save_state(self, params, scores):
        """Save the model and its corresponding metadata."""
        try:
            logger.info("Saving state to %s", self.dir)
            self._save_model()
            self._save_metadata(params, scores)
        except Exception as _:
            logger.error("Failed to save object.", exc_info=True)
            raise

    def _save_model(self):
        """Save the model to a local directory."""
        with open(self.estimator_path, 'wb') as model_file:
            dill.dump(self.model, model_file)

    def _save_metadata(self, params, scores):
        """Construct and save metadata."""

        evaluation_scores = {
            "f1_score": scores["f1"],
            "auc_score": scores["auc"],
        }

        metadata = {
            "id": self.id,
            "model_params": params,
            "evaluation_result": evaluation_scores,
            "features": self.features,
            "categorical_features": self.categorical_features,
            "target": self.target
        }

        with open(self.metadata_path, 'w') as model_meta_file:
            json.dump(metadata, model_meta_file)
