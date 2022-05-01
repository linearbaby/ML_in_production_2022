from data.utils import get_dataset
from features.processing import preprocess_pipeline
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression


def train(model, X, y):
    epochs = 20
    for epoch in range(epochs):
        model.fit(X, y)

        
@hydra.main(config_path="configs", config_name="main_conf")
def main(cfg: OmegaConf):
    ds = get_dataset()
    X_train, X_test, y_train, y_test = preprocess_pipeline(ds)
    model = LogisticRegression()
    result = model.fit(X_train, y_train)



