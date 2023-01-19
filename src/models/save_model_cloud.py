from google.cloud import storage
import json


def upload_model_to_cloud():
    with open("config/config.json") as file:
        cfg = json.load(file)

    # create a client object
    client = storage.Client()

    # create a bucket object
    bucket = client.bucket("dtumlops-storage")

    # create a blob object
    print(f"cloud storage path: models/")
    blob = bucket.blob(f"models/")

    print(f' path to file: {cfg["paths"]["path_checkpoint"]}')
    # upload the file
    blob.upload_from_filename(filename=cfg["paths"]["path_checkpoint"])
    print("File uploaded successfully.")


if __name__ == "__main__":
    upload_model_to_cloud()
