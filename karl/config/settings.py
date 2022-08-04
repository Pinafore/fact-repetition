import os
from dotenv import load_dotenv

load_dotenv()

CODE_DIR = os.environ.get('CODE_DIR')
DATA_DIR = f'{CODE_DIR}/data'
API_URL = os.environ.get('API_URL')
MODEL_API_URL = os.environ.get('MODEL_API_URL')
SQLALCHEMY_DATABASE_URL = os.environ.get('SQLALCHEMY_DATABASE_URL')
USE_MULTIPROCESSING = os.environ.get('USE_MULTIPROCESSING')
MP_CONTEXT = os.environ.get('MP_CONTEXT')
