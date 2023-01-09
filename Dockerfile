#FROM --platform=linux/amd64 python:3.8-buster
FROM python:3.8-buster

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

# Get necessary system packages
RUN apt-get update \
#  && apt-get upgrade\
#  && apt update \
#  && apt upgrade \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev

# Get mysql
RUN apt-get install --yes default-mysql-client
#RUN #apt install libmariadb3 libmariadb-dev

# Setup python
COPY requirements.dev.txt .
COPY requirements.txt .
RUN python -m pip install -U pip
RUN python -m pip install -r requirements.dev.txt
RUN python -m pip install -r requirements.txt

# Cleanup
RUN rm -rf /var/lib/apt/lists/*

# Copy files
COPY baseball.sql .
COPY BaseballFeatures/feature-extract.sql .
COPY Docker/run.sh .
#COPY Final/connection-test.py .
COPY BaseballFeatures/analyze-features.py .
COPY midterm/* .

# Make directories
RUN mkdir $APP_HOME/output
RUN mkdir $APP_HOME/output/plots

# Run script
RUN ["chmod", "u+x", "run.sh"]
SHELL ["/bin/bash", "-c"]
CMD ./run.sh
