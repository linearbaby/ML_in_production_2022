import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import click

import pickle
import os


def process_numerical_features() -> Pipeline:
    """Generate transforms for numerical columns"""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    return numeric_transformer


def process_categorical_features() -> Pipeline:
    """Generate transforms for categorical columns"""
    categoric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="M")),
            ("scaler", OneHotEncoder()),
        ]
    )
    return categoric_transformer


def preprocess_pipeline(dataset: pd.DataFrame) -> tuple:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                process_numerical_features(),
                dataset.select_dtypes(include=["int64", "float64"]).columns,
            ),
            (
                "cat",
                process_categorical_features(),
                dataset.select_dtypes(include=["object"]).columns,
            ),
        ]
    )
    return preprocessor


@click.command()
@click.option('--ds', envvar='DAY_STAMP', required=True)
def process(ds: str):
    data = pd.read_csv(f"/data/raw/{ds}/data.csv")
    target = pd.read_csv(f"/data/raw/{ds}/target.csv")

    preprocessor = preprocess_pipeline(data)
    preprocessor.fit(data)

    # dump preprocessor
    preprocess_dir = f"/data/models/{ds}/"
    os.makedirs(preprocess_dir, exist_ok=True)
    with open(preprocess_dir + "preprocess.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    # save preprocessed data
    data = pd.DataFrame(preprocessor.transform(data))
    data_files_dir = f"/data/processed/{ds}/"
    os.makedirs(data_files_dir, exist_ok=True)
    data.to_csv(data_files_dir + "data.csv", index=False)
    target.to_csv(data_files_dir + "target.csv", index=False)


if __name__ == "__main__":
    process()
