CODE_DIR = '/fs/clip-quiz/shifeng/karl'
DATA_DIR = f'{CODE_DIR}/data'
SQLALCHEMY_DATABASE_URL = 'postgresql+psycopg2://shifeng@localhost:5433/karl-prod'
STABLE_DATABASE_URL = 'postgresql+psycopg2://shifeng@localhost:5434/karl-prod'
USE_MULTIPROCESSING = True
USER_STATS_CACHE = True
MP_CONTEXT = 'fork'
