from flask import jsonify, request, render_template
from werkzeug.exceptions import HTTPException

from default_detection.api import app
from default_detection.api.predictions import predict_default_probability, predict_test_set_default_probability


@app.route("/predictions", methods=["POST"])
def predictions():
    data = request.get_json()
    try:
        result = predict_default_probability(data["uuids"])
        return jsonify(predictions=result)
    except Exception as e:
        app.logger.error(e, exc_info=True)
        handle_error(e)


@app.route("/get-all", methods=["GET"])
def get_all_predictions():
    try:
        result = predict_test_set_default_probability()
        return render_template(
            "predictions.html",
            tables=[result.to_html(classes='data')],
        )

    except Exception as e:
        app.logger.error(e, exc_info=True)
        handle_error(e)


@app.route("/health-check", methods=["GET"])
def health_check():
    return "OK"


@app.errorhandler(Exception)
def handle_error(e):
    code = 500
    if isinstance(e, HTTPException):
        code = e.code
    return jsonify(error=str(e)), code
