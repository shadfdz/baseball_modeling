# PythonProject

- This project was developed on a Mac M1 therefore Python 3.8 is recommended

# Setup for development:

- Setup a python 3.8 venv (usually in `.venv`)
  - You can run `./scripts/create-venv.sh` to generate one
- `pip3 install --upgrade pip`
- Install pip-tools `pip3 install pip-tools`
- Update dev requirements: `pip-compile --output-file=requirements.dev.txt requirements.dev.in`
- Update requirements: `pip-compile --output-file=requirements.txt requirements.in`
- Install dev requirements `pip3 install -r requirements.dev.txt`
- Install requirements `pip3 install -r requirements.txt`
- `pre-commit install`

## Update versions

`pip-compile --output-file=requirements.dev.txt requirements.dev.in --upgrade`

# Run `pre-commit` locally.

`pre-commit run --all-files`

# HW 6

Please enter user as root as set password in docker-compose.
These credentials will be need for run_scripts.sh and homework5.py on line 317

Outputs can be opened under index.html in the homework folder.

Temporarily disabled flake8 after pre-commit due to conflict with black

Credit to https://github.com/vishnubob/wait-for-it for wait-for-it.sh

# Final
