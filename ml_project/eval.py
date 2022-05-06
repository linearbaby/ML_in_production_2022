import hydra
from omegaconf import OmegaConf

from models.utils import load_recent_model, save_eval
from data.utils import get_dataset
from features.processing import preprocess_pipeline
from utils.workflow import init_rootpath


@hydra.main(config_path="configs", config_name="main_conf_eval")
def main(cfg: OmegaConf):
    init_rootpath(cfg)

    model = load_recent_model(cfg.model)
    if model is None:
        return

    ds = get_dataset(cfg.data)
    X = preprocess_pipeline(cfg.pipeline, ds, eval=True)

    save_eval(cfg.model, model.predict(X))


if __name__ == "__main__":
    main()
