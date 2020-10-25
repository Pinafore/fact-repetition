# KARL scheduler

Make sure to use `git-lfs` to pull the model checkpoints too alongside the code.

## Install python dependencies
1. If you are using `conda`, consider creating a new environment, and make sure
   to run `conda install pip` so that the following dependencies are installed
   for your environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Install Spacy module with `python -m spacy download en_core_web_lg`.

## PostgreSQL server
1. Start the server `postgres -D /fs/clip-quiz/shifeng/postgres -p 5433`
2. Creat dump `pg_dump karl-prod -h /fs/clip-quiz/shifeng/postgres/run -p 5433 | gzip > /fs/clip-quiz/shifeng/karl/backup/karl-prod_20201017.gz`

## Run the scheduler
1. `uvicorn karl.web:app --log-level debug`

