import logging

import numpy as np
import pandas as pd

from default_detection.api.predictor import Predictor
from default_detection.data import CSV_DATA, preprocessing

logger = logging.getLogger(__name__)
_default_classifier = None
_data = pd.DataFrame()


def load_model():
    """Load the model if not already loaded."""
    global _default_classifier

    if not _default_classifier:
        _default_classifier = Predictor()


def load_data():
    """Load the data if not already loaded."""
    global _data

    if _data.empty:
        logger.info("Reading data")
        raw_data = pd.read_csv(CSV_DATA, delimiter=";")

        logger.info("Preprocessing data")
        _data, _, _ = preprocessing.preprocess_dataset(raw_data)


def predict_default_probability(uuids):
    """Predict probability of default for specified UUIDs."""
    load_model()
    load_data()

    ids = [i["uuid"] for i in uuids]
    to_predict = _data.query(f"uuid.isin({ids})").copy()

    logger.info(f"Predicting for {len(to_predict)} rows")
    to_predict.loc[:, 'pd'] = np.round(_default_classifier.predict(to_predict), 5)
    to_predict.loc[:, 'default'] = np.where(to_predict["pd"] < 0.5, 0, 1)

    result = []
    for i, row in to_predict.iterrows():
        result.append(
            {
                "uuid": row.uuid,
                "default": row.default,
                "pd": row.pd
            }
        )
    return result


def predict_test_set_default_probability():
    """Predict probability of default for the whole test set."""
    load_model()
    load_data()

    to_predict = _data[_data.default.isnull()].copy()

    logger.info(f"Predicting test set, {len(to_predict)} rows")
    to_predict.loc[:, 'pd'] = np.round(_default_classifier.predict(to_predict), 5)
    to_predict.loc[:, 'default'] = np.where(to_predict["pd"] < 0.5, 0, 1)

    return to_predict[["uuid", "pd", "default"]]
