import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)


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


def preprocess_pipeline(config: DictConfig, dataset: pd.DataFrame, eval=False) -> tuple:
    if not eval:
        log.debug('splitting target and dataset')
        target = dataset.to_numpy()[:, -1]
        dataset = dataset.iloc[:, :-1:]

    log.debug('creating preprocessor pipeline')
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

    # check if eval, then there is no target in dataset,
    # and not need to train test_split
    if not eval:
        log.debug('preprocessing dataset, and splitting it for train')
        return train_test_split(
            preprocessor.fit_transform(dataset),
            target,
            test_size=config.test_size,
            random_state=config.random_state,
        )
    else:
        log.debug('preprocessing dataset for evaluation')
        return preprocessor.fit_transform(dataset)
