import logging
import os
import sys

# Download folder
DATA_DIR = os.getenv("DATA_DIR", None)
if not DATA_DIR:
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "downloaded_model"))

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    os.environ["DATA_DIR"] = DATA_DIR

# Bucket to download model
S3_BUCKET = os.getenv("S3_BUCKET", "labinot-development")
RANDOM_STATE = os.getenv("RANDOM_STATE", 1337)

# AWS credentials
AWS_CREDENTIALS = {
    "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
    "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY")
}

# Logging
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
logger.setLevel(logging.INFO)
