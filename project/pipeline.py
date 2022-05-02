from data.utils import get_dataset
from features.processing import preprocess_pipeline
import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from hydra.utils import get_original_cwd
import logging

log = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="main_conf")
def main(cfg: OmegaConf):
    # instantiate root_path for propriate file management
    log.debug('instantiate root_path for propriate file management')
    cfg.core.root_path = get_original_cwd()

    ds = get_dataset(config=cfg.data)
    X_train, X_test, y_train, y_test = preprocess_pipeline(cfg.pipeline, ds)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    print(accuracy_score(model.predict(X_test), y_test))
    pass

if __name__ == "__main__":
    main()