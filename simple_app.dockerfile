FROM python:3.9-slim

#EXPOSE 8501


WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
COPY src/models/model.py src/models/model.py
COPY src/models/__init__.py src/models/__init__.py
COPY models/detox_checkpoint1.pth models/detox_checkpoint1.pth
COPY src/deployment deployment
COPY src/deployment/requirements.txt requirements.txt
COPY setup.py setup.py

RUN pip install -e .
RUN pip install --no-cache-dir --upgrade -r requirements.txt


#CMD exec uvicorn main:app --port $PORT --host 0.0.0.0 --workers 1
#CMD ["uvicorn", "src.deployment.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]