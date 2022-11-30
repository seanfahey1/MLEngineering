FROM python:3.8-buster

ENV APP_HOME /app
WORKDIR $APP_HOME
ENV PYTHONPATH /

RUN echo 'working-------------' && echo "" && echo "" && echo ""

# Get necessary system packages
RUN apt-get update \
  && apt-get install --no-install-recommends --yes \
     build-essential \
     python3 \
     python3-pip \
     python3-dev

# RUN apt-get install --yes mysql-server
RUN apt-get install --yes default-mysql-client
RUN rm -rf /var/lib/apt/lists/*

COPY baseball.sql .
COPY .venv /app/.venv
RUN mkdir $APP_HOME/output
RUN touch $APP_HOME/output/touch_check

COPY Docker/run.sh /app
RUN ["chmod", "u+x", "run.sh"]

SHELL ["/bin/bash", "-c"]
CMD ./run.sh
