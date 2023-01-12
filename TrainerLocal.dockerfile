# Base image
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# Copy essential parts of application
COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

# Set work dir in our container and add commands that install dependencies
WORKDIR /
#RUN pip install --upgrade pip
#RUN pip install -r requirements.txt --no-cache-dir

# Set entrypoint
#ENTRYPOINT ["python", "-u", "src/models/train_model.py", "train"]
