FROM python:3.9-slim

EXPOSE $PORT


WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
COPY src/models/model.py model.py
COPY config/config.json config.json
COPY src/deployment/requirements.txt requirements.txt
COPY src/deployment/main.py main.py
COPY models.dvc models.dvc

RUN pip install --no-cache-dir --upgrade -r requirements.txt
RUN pip install dvc 'dvc[gs]'
RUN dvc init --no-scm
RUN dvc remote add -d storage gs://dtumlops-storage
RUN dvc pull


CMD exec uvicorn main:app --port $PORT --host 0.0.0.0 --workers 1