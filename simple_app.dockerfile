FROM python:3.9-slim

EXPOSE $PORT


WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
COPY src/models/model.py model.py
COPY models/detox_checkpoint1.pth detox_checkpoint1.pth
COPY src/deployment/requirements.txt requirements.txt
COPY src/deployment/main.py main.py
COPY src/deployment/latest_training_dict.pickle latest_training_dict.pickle

RUN pip install --no-cache-dir --upgrade -r requirements.txt


CMD exec uvicorn main:app --port $PORT --host 0.0.0.0 --workers 1