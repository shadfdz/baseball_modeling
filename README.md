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

# HW2 Info

- MySql Server version 8.0.26
- HW 2 dataset [can be found here](https://teaching.mrsharky.com/sdsu_fall_2020_lecture03.html#/10/1)

# HW4 Info

- Please enter respective pass key for db credentials and access
- JDBC connector for MySql Server version 8.0.26 located under dbConnectors folder
