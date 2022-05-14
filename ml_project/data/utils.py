import logging
import wget
from os import mkdir
from os.path import split
from pathlib import Path
from omegaconf import DictConfig
import pandas as pd

from utils.workflow import create_subdirs

log = logging.getLogger(__name__)


def load_file(url: str, file_path: str, override=False):
    """Nothrow loads file from specified url"""
    # creating file
    path = Path(file_path)
    if override and path.is_file() or not path.is_file():
        # checking directory existance or creating it
        log.debug(f"trying to mkdir for data file ({split(file_path)[0]})")
        try:
            create_subdirs(file_path)
        except OSError:
            log.debug("data folder already exists")
        else:
            log.debug("made folder for data file")

        log.debug("loading dataset file")
        wget.download(url, file_path)
    else:
        log.debug("dataset file is already on local machine")


def get_dataset(config: DictConfig) -> pd.DataFrame:
    load_file(config.data_url, config.file_location)

    # possible place for extintions checking

    log.info(f"reading DataFrame object from {config.file_location}")
    df = pd.read_csv(config.file_location)
    log.info(f"DataFrame object instantiated")
    return df
