from data.utils import get_dataset
from models.utils import (get_model, save_metrics, save_model)
from features.processing import preprocess_pipeline
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score
from utils.workflow import init_rootpath
import logging


@hydra.main(config_path="configs", config_name="main_conf_train")
def main(cfg: OmegaConf):
    init_rootpath(cfg)

    model = get_model(cfg.model)
    if model is None:
        return

    ds = get_dataset(config=cfg.data)
    X_train, X_test, y_train, y_test = preprocess_pipeline(cfg.pipeline, ds)

    model.fit(X_train, y_train)

    save_model(model, cfg.model)
    save_metrics(f"accuracy score of model: {accuracy_score(model.predict(X_test), y_test)}", cfg.model)

if __name__ == "__main__":
    main()