import os

from flask import Flask

S3_BUCKET = os.getenv("S3_BUCKET", "labinot-development")
app = Flask(__name__)

from default_detection.api import endpoints
