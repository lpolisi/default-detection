import re

import category_encoders as ce
import pandas as pd

from default_detection.data import ENCODED_FEATURES, NUMERICAL_FEATURES, BINNED_FEATURES


def preprocess_dataset(data):
    """Apply data preprocessing such as imputation and one-hot-encoding"""
    df = data.copy()

    # Convert to int
    df["has_paid"] = df["has_paid"].astype(int)

    # Bin numerical features
    df, new_features = _bin_features(df)

    # Encode data
    encoder = ce.one_hot.OneHotEncoder(
        cols=ENCODED_FEATURES,
        use_cat_names=True,
        handle_unknown="indicator",
        handle_missing="indicator"
    )
    df = encoder.fit_transform(df)

    # Rename invalid column names
    df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    # Impute nulls
    target_features = [
        "account_status",
        "account_worst_status_12_24m",
        "account_worst_status_6_12m",
        "account_worst_status_3_6m",
        "account_worst_status_0_3m"
    ]
    df.loc[:, target_features] = df[target_features].fillna(0)

    features = list(set(df.columns.to_list()) - {"uuid", "default"})
    categorical_features = list(set(features) - set(NUMERICAL_FEATURES + new_features))
    return df, features, categorical_features


def _bin_features(df):
    def _bin(column, df):
        df[f"{column}_bin"], bins = pd.qcut(df[column], q=10, labels=False, retbins=True, duplicates="drop")
        return df, f"{column}_bin"

    new_features = []

    for f in BINNED_FEATURES:
        df, new_f = _bin(f, df)
        new_features.append(new_f)

    return df, new_features
