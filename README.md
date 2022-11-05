## General info

This project is a simple Flask application with a model for predicting probability of credit default.

## Environment

The environment is managed by [anaconda](https://www.anaconda.com/).

To create the environment use:

```
conda env create -f environment.yml
```

To import changes:

```
conda env update --file environment.yml
```

To export changes:

Make sure the version is specified and that it exists for all relevant platforms in the conda channel.

```
conda env export --from-history | grep -v "^prefix: " > environment.yml
```

## Setup

To run this project, activate the conda environment with:

```
conda activate default-detection
```

Then run the Flask application with:

```
 FLASK_ENV=development FLASK_APP=default_detection.api flask run
```

## Endpoints

There are two endpoints for this application:

```
1. http://13.53.140.245:8080/get-all
2. http://13.53.140.245:8080/predictions
```

The first one queries all predictions for the test set. Simply type in the URL in a browser to retrieve the predictions if running locally. 

The second one queries predictions for specific UUIDs, both within the test set and the training set. An example query: 

```
echo '{"uuids": [{"uuid":"0095dfb6-a886-4e2a-b056-15ef45fdb0ef"}]}' | http http://13.53.140.245:8080/predictions
```
