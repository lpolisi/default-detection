from default_detection.api import predictions, app

if __name__ == '__main__':
    predictions.load_data()
    predictions.load_model()
    app.run()
