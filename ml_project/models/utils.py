from typing import Union
import logging
import pickle
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import sklearn
import pandas as pd
from omegaconf import DictConfig

from utils.workflow import create_subdirs


log = logging.getLogger(__name__)

model_configure_dict = {"LogReg": LogisticRegression, "GaussianNB": GaussianNB}


def get_model(model_config: DictConfig) -> sklearn.base.BaseEstimator:
    """Generate model from config"""
    log.debug("trying to create model instance")
    try:
        model = model_configure_dict[model_config.model_type]
        model = model(**model_config.params)
    except:
        log.critical("model object was not created")
        model = None
    else:
        log.info(f"successfully created model object of class{model_config.model_type}")

    return model


def save_model(model: sklearn.base.BaseEstimator, model_config: DictConfig):
    log.debug("trying to save model")
    try:
        create_subdirs(model_config.artifact_path)
        with open(model_config.artifact_path, "wb") as art_file:
            pickle.dump(model, art_file)
    except:
        log.error("cannot save model to pcl file")
    else:
        log.info("successfully saved model to artifact file")


def save_metrics(metrics: str, model_config: DictConfig):
    log.debug("trying to save metrics")
    try:
        create_subdirs(model_config.metrics_path)
        with open(model_config.metrics_path, "w") as metrics_file:
            metrics_file.write(metrics)
    except:
        log.error("cannot save metrics")
    else:
        log.info("successfully saved metrics")


def load_recent_model(model_config: DictConfig):
    """Loads most recent model from hydra outputs"""
    model = None

    if not os.path.isdir(model_config.model_pkl_path):
        log.error("No model have been fit yet")
    else:
        log.debug("searching for most recent trained model")
        latest_date = sorted(os.listdir(model_config.model_pkl_path), reverse=True)[0]
        latest_time = sorted(
            os.listdir(model_config.model_pkl_path + "/" + latest_date), reverse=True
        )[0]
        model_path = (
            model_config.model_pkl_path
            + "/"
            + latest_date
            + "/"
            + latest_time
            + "/"
            + model_config.artifact_path
        )

        log.info(f"found most recent model in: {model_path}")
        with open(model_path, "rb") as art_file:
            model = pickle.load(art_file)

    log.info(f"successfully loaded model")
    return model


def save_eval(model_config: DictConfig, ds: Union[pd.DataFrame, np.ndarray]):
    dataset = pd.DataFrame(ds)
    log.info(f"saving results to results dir")
    dataset.to_csv(model_config.model_results_path)
