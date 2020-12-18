CODE_DIR = '/fs/clip-quiz/shifeng/karl-dev'
# CODE_DIR = '/Users/shifeng/workspace/fact-repetition'
DATA_DIR = f'{CODE_DIR}/data'
DB_HOST = '/scratch0/shifeng/karl-db/run'
# DB_HOST = '/Users/shifeng/workspace/karl-db/run'
SQLALCHEMY_DATABASE_URL = f'postgresql+psycopg2://shifeng@localhost:5433/karl-prod?host={DB_HOST}'
# STABLE_DATABASE_URL = 'postgresql+psycopg2://shifeng@4.tcp.ngrok.io:16017/karl-prod'
STABLE_DATABASE_URL = 'postgresql+psycopg2://shifeng@localhost:5434/karl-prod'
USE_MULTIPROCESSING = False
