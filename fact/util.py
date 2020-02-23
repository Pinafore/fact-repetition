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


class Flashcard(BaseModel):
    text: str
    user_id: Optional[str]
    question_id: Optional[str]
    user_accuracy: Optional[float]
    user_buzzratio: Optional[float]
    user_count: Optional[float]
    question_accuracy: Optional[float]
    question_buzzratio: Optional[float]
    question_count: Optional[float]
    times_seen: Optional[float]
    times_correct: Optional[float]
    times_wrong: Optional[float]
    label: Optional[str]
    answer: Optional[str]
    category: Optional[str]
