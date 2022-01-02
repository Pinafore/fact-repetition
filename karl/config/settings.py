CODE_DIR = '/home/shi/scheduler'
DATA_DIR = f'{CODE_DIR}/data'
SQLALCHEMY_DATABASE_URL = 'postgresql+psycopg2://shifeng@localhost:5433/karl-prod'
STABLE_DATABASE_URL = 'postgresql+psycopg2://shifeng@localhost:5433/karl-prod'
USE_MULTIPROCESSING = True
USER_STATS_CACHE = True
MP_CONTEXT = 'fork'
HOST = 'http://172.17.0.1:4000/api/karl'
