# KARL scheduler

Make sure to use `git-lfs` to pull the model checkpoints too alongside the code.

## Install python dependencies
1. If you are using `conda`, consider creating a new environment, and make sure
   to run `conda install pip` so that the following dependencies are installed
   for your environment.
2. It's recommended that you use python 3.11.4
3. Install dependencies with `poetry install`.
4. [Optional] Install Spacy module with `python -m spacy download en_core_web_lg`.
5. Start the poetry shell `poetry shell`.
6. If you see an error related to `psycopg2-binary`, the easiest solution is probably to install it via pip.

## Start PostgreSQL server and load the dev database
1. Use brew to install PostgreSQL 12: `brew install postgresql@12`.
2. The server should automatically start. You can use brew services to manage it, e.g., `brew services stop postgresql@12`.
3. Create DB cluster `initdb`, then create DB `createdb karl-prod`.
4. Restore from dump `gunzip -c data/karl-dev.gz | psql karl-prod`.

## Start PostgreSQL server on UMIACS
The default PostgreSQL runtime directory is not available on UMIACS machines, so extra steps are required to redirect it.
1. Create DB cluster `initdb -D /fs/clip-quiz/shifeng/postgres`
2. Create the runtime directory `/fs/clip-quiz/shifeng/postgres/run`
3. Open `/fs/clip-quiz/shifeng/postgres/postgresql.conf`, find `unix_socket_directories` and point it to the runtime directory created above. 
4. Start the server `postgres -D /fs/clip-quiz/shifeng/postgres -p 5433`
5. Create DB `createdb -h /fs/clip-quiz/shifeng/postgres/run -p 5433 karl-prod`
6. Create dump `pg_dump karl-prod -h /fs/clip-quiz/shifeng/postgres/run -p 5433 | gzip > /fs/clip-quiz/shifeng/db-karl-backup/karl-prod_20210309.gz`
7. Restore from dump `gunzip -c /fs/clip-quiz/shifeng/db-karl-backup/karl-prod_20210309.gz | psql -p 5433 karl-prod`. Need to drop existing database first.

## Run the scheduler
1. Start the retention model: `uvicorn karl.retention_hf.web:app --log-level info --port 8001`
2. Start the scheduler itself: `uvicorn karl.web:app --log-level debug`

## DB migration
For example, to keep the development branch DB up to date with master branch.
1. Create dump: `pg_dump karl-prod -h /fs/clip-quiz/shifeng/postgres/run -p 5433 | gzip > /fs/clip-quiz/shifeng/karl/backup/karl-prod_20201017.gz`
2. On the development machine, delete the existing database: `psql -p 5433 karl-dev` then `DROP DATABASE "karl-prod";`
3. Load the dump `gunzip -c /fs/clip-quiz/shifeng/db-karl-backup/karl-prod_20210309.gz | psql -p 5433 karl-prod`

## Figures
1. For figures you might need `vega-lite`. Install with conda: `conda install -c conda-forge vega vega-lite vega-embed vega-lite-cli`.
2. Note, can also be done by doing `npm install -g` for vega, vega-lite, vega-embed, and vega-cli.
