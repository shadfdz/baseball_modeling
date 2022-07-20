# set base image using python-slim
FROM python:3.8-slim

# set the working directory in the container
WORKDIR /homework

COPY ./output /output

RUN apt-get update && apt-get install -y default-mysql-client

ADD ./requirements.txt /homework/requirements.txt

# install dependencies
RUN pip install -r requirements.txt

# copy files to the working directory
COPY . /homework

# run script to alter db i.e rolling transform
RUN chmod +x run_scripts.sh

CMD ["bash", "run_scripts.sh"]
