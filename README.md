# KARL scheduler

Make sure to use `git-lfs` to pull the model checkpoints too alongside the code.

## Install python dependencies
1. If you are using `conda`, consider creating a new environment, and make sure
   to run `conda install pip` so that the following dependencies are installed
   for your environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Install Spacy module with `python -m spacy download en_core_web_lg`.

## Run the scheduler
1. `uvicorn karl.web:app --log-level debug`
