import json
import logging
import os
from pathlib import Path

import boto3
import dill
from botocore.exceptions import ClientError

from default_detection import DATA_DIR, S3_BUCKET, AWS_CREDENTIALS
from default_detection.api import app

logger = logging.getLogger(__name__)


class Predictor(object):
    """Wrapper class to handle downloading of raw classification model from S3 and predictions."""

    def __init__(self):
        self.model = None
        self.metadata = None

        download_dir = os.path.join(DATA_DIR)
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Loading model into memory...")
        self._download_model(download_dir)

        metadata_path = os.path.join(download_dir, "model-metadata.json")
        with open(metadata_path, 'r') as metadata_file:
            self.metadata = json.load(metadata_file)

        logger.info(f"Done loading model")

    def predict(self, input_data):
        """Predict probability of default."""
        return self.model.predict(input_data[self.metadata["features"]])

    def _download_model(self, download_dir):
        logger.info("Downloading data from S3...")
        for filename in ["model.pk", "model-metadata.json"]:
            logger.info("Downloading %s..." % filename)
            self._download_model_data(
                bucket_name=S3_BUCKET,
                key="model/" + filename,
                target_path=os.path.join(download_dir, filename),
                credentials=AWS_CREDENTIALS
            )

        model_path = os.path.join(download_dir, "model.pk")
        with open(model_path, 'rb') as file:
            self.model = dill.load(file)

    @staticmethod
    def _download_model_data(bucket_name, key, target_path, credentials):
        s3 = boto3.resource(
            "s3",
            aws_access_key_id=credentials["aws_access_key_id"],
            aws_secret_access_key=credentials["aws_secret_access_key"]
        )

        try:
            s3.Bucket(bucket_name).download_file(key, target_path)
        except ClientError as e:
            if e.response['Error']['Code'] == "404":
                app.logger.error("The object does not exist.")
            else:
                raise
