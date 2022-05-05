import os
import errno
import logging
from omegaconf import OmegaConf
from hydra.utils import get_original_cwd

log = logging.getLogger(__name__)

def create_subdirs(file_path:str):
    '''function for creating files with path'''
    if not os.path.exists(os.path.dirname(file_path)):
        try:
            os.makedirs(os.path.dirname(file_path))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                log.critical("cannot create directories according to file path")
                raise


def init_rootpath(cfg: OmegaConf):
    # instantiate root_path for propriate file management
    log.debug('instantiates root_path for appropriate file management')
    cfg.core.root_path = get_original_cwd()