#FROM --platform=linux/amd64 python:3.8-buster
FROM python:3.8-buster

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

RUN echo "working"

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev

# Get mysql
RUN apt-get install --yes default-mysql-client

# Cleanup
RUN rm -rf /var/lib/apt/lists/*

# Copy files
COPY baseball.sql .
COPY Docker/100_day_rolling_calc.sql .
COPY Docker/run.sh /app

# Make directories
RUN mkdir $APP_HOME/output

# Run script
RUN ["chmod", "u+x", "run.sh"]
SHELL ["/bin/bash", "-c"]
CMD ./run.sh
