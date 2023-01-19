# Base image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install wandb

# Copy essential parts of application
WORKDIR /
COPY src/ src/
COPY reports/ reports/
COPY config/ config/
COPY data.dvc data.dvc
COPY models.dvc models.dvc

# Set work dir in our container and add commands that install dependencies
#For some reason, fails to install requirements if they are not copied after teh WORKDIR/
COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install dvc 'dvc[gs]'
RUN dvc init --no-scm
RUN dvc remote add -d storage gs://dtumlops-storage
RUN dvc pull data.dvc
RUN mkdir models/

# Set entrypoint
ENTRYPOINT ["python", "-u", "src/models/sweep_model.py"]
