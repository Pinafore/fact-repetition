import subprocess
import logging
import os
import torch


try:
    CUDA = torch.cuda.is_available()
    DEVICE = torch.device('cuda' if CUDA else 'cpu')
except:  # pylint: disable=bare-except
    CUDA = False
    DEVICE = 'cpu'


def shell(command):
    return subprocess.run(command, check=True, shell=True, stderr=subprocess.STDOUT)


def safe_path(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_logger(name):
    log = logging.getLogger(name)

    if not log.hasHandlers():
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        sh = logging.StreamHandler()
        sh.setLevel(logging.INFO)
        sh.setFormatter(formatter)
        log.addHandler(sh)
        log.setLevel(logging.INFO)
    return log
