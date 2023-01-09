FROM python:3.9-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt requirements.txt
COPY ./src/ src/
copy ./wandb_tester.py wandb_tester.py

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENV WANDB_API_KEY="56904d7dacddf7e30783626c3109d11ecf03980a"

ENTRYPOINT ["true"]