import logging
import pickle
import os
from hydra import compose, initialize
from typing import List, Union
from google_drive_downloader import GoogleDriveDownloader as gdd
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conlist


logger = logging.getLogger("uvicorn")

cfg = None
predict_pipeline = None

app = FastAPI()


class PredictionData(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=31, max_items=31)]
    features: List[str]


class Response(BaseModel):
    result: int


@app.get("/")
def main():
    return "entry point of framework"


@app.on_event("startup")
def load_model():
    global cfg, predict_pipeline

    logger.info("initializing hydra configs")

    # hydra config processing.
    initialize(config_path="configs")
    cfg = compose(config_name="main_conf_eval")
    cfg.core.root_path = os.path.dirname(os.path.abspath(__file__))

    logger.info("parsing environment variables")
    serialization_path = os.getenv("PATH_TO_SERIALIZATION")
    model_location = os.getenv("MODEL_LOCATION")
    if serialization_path is None:
        if model_location == "Google.disk":
            serialization_path = cfg.pipeline.default_model_sha
        elif model_location == "local":
            serialization_path = cfg.pipeline.default_prediction_pipeline
        else:
            logger.critical(
                'Wrong place to load model. Shoud be either '
                '"local" or "Google.disk"'
            )
            raise Exception("no model location specified")

    # model loading part
    if model_location != "Google.disk":
        logger.info(f"Loading model from disk, location: {serialization_path}")

        with open(serialization_path, "rb") as art_file:
            predict_pipeline = pickle.load(art_file)
    else:
        logger.info(
            "Loading model from google disk, "
            f"location: {serialization_path}")

        gdd.download_file_from_google_drive(
            file_id=serialization_path, 
            dest_path=cfg.pipeline.path_to_load_model)
        with open(cfg.pipeline.path_to_load_model, "rb") as art_file:
            predict_pipeline = pickle.load(art_file)
        os.remove(cfg.pipeline.path_to_load_model)

    # check if model loaded
    if predict_pipeline is None:
        logger.critical("Can't load model, exiting")
        raise Exception("Model loading error")


@app.get("/health")
def health():
    return predict_pipeline is not None


@app.get("/predict", response_model=List[Response])
def predict(request: PredictionData):
    pred_data = pd.DataFrame(request.data, columns=request.features)
    return [Response(result=int(i)) for i in predict_pipeline.predict(pred_data)]


if __name__ == "__main__":
    uvicorn.run("app:app")
