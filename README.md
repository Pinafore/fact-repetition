# KARL scheduler

Make sure to use `git-lfs` to pull the model checkpoints too alongside the code.

## Content-Aware Data

Our data is available on huggingface [here](https://huggingface.co/datasets/nbalepur/KARL)!

## Install python dependencies
1. If you are using `conda`, consider creating a new environment, and make sure
   to run `conda install pip` so that the following dependencies are installed
   for your environment.
2. It's recommended that you use python 3.11.4
3. Install dependencies with `poetry install`.
4. [Optional] Install Spacy module with `python -m spacy download en_core_web_lg`.
5. Start the poetry shell `poetry shell`.
6. If you see an error related to `psycopg2-binary`, the easiest solution is probably to install it via pip.

## Start PostgreSQL server
1. Use brew to install PostgreSQL 12: `brew install postgresql@12`.
2. The server should automatically start. You can use brew services to manage it, e.g., `brew services stop postgresql@12`.
3. Create DB cluster `initdb`, then create DB `createdb karl-prod`.
4. You may need to modify `alembic.ini` to specify `sqlalchemy.url` to have your name

### load the dev database
1. Restore from dump `gunzip -c data/karl-dev.gz | psql karl-prod

### start from scratch
1. Run `alembic upgrade head`

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
1. Start the scheduler itself: `uvicorn karl.web:app --log-level debug`
2. If you are using a retention model hosted on UMIACS machine, you can likely use the following command to connect to it `ssh -NfL 8001:localhost:8001 your_name@nexusclip00.umiacs.umd.edu`. An alternative is to use an ssh config
3. If you are running a local retention model, start it with: `uvicorn karl.retention_hf.web:app --log-level info --port 8001`

### SSH Config
Add the below code to ~/.ssh/config and you can call `ssh clip` going forward to connect to the retention model
```
Host clip
  LocalForward  [your MODEL_API_URL] localhost:8001 # my end, other end
  User [add user]
  Hostname nexusclip00.umiacs.umd.edu
```
## Running a test
1. After `poetry shell`, run `python -m karl.tests.test_scheduling_with_session`.

## `dotenv` file
You need a `.env` file in the `karl` directory. Modify `CODE_DIR` as needed and change `shifeng` in `SQLALCHEMY_DATABASE_URL` to your user (check via `SELECT current_user;`). 
Change `API_URL` to match with the `INTERFACE` variable in the app. You may also need to specify a password to your database url.
```
CODE_DIR="/Users/shifeng/workspace/fact-repetition"
# Should match with port defined in INTERFACE in karl app .env 
API_URL="http://0.0.0.0:8000" 
MODEL_API_URL="http://0.0.0.0:8001"
SQLALCHEMY_DATABASE_URL="postgresql+psycopg2://shifeng@localhost:5432/karl-prod"
USE_MULTIPROCESSING=True
MP_CONTEXT="fork"
```

## DB migration
For example, to keep the development branch DB up to date with master branch.
1. Create dump: `pg_dump karl-prod -h /fs/clip-quiz/shifeng/postgres/run -p 5433 | gzip > /fs/clip-quiz/shifeng/karl/backup/karl-prod_20201017.gz`
2. On the development machine, delete the existing database: `psql -p 5433 karl-dev` then `DROP DATABASE "karl-prod";`
3. Load the dump `gunzip -c /fs/clip-quiz/shifeng/db-karl-backup/karl-prod_20210309.gz | psql -p 5433 karl-prod`

## Figures
1. For figures you might need `vega-lite`. Install with conda: `conda install -c conda-forge vega vega-lite vega-embed vega-lite-cli`.
2. Note, can also be done by doing `npm install -g` for vega, vega-lite, vega-embed, and vega-cli.
