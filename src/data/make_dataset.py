# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import numpy as np
import torch


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    for tt in ['train','test']:
        content = [np.load(f'{input_filepath}/{path}') for path in os.listdir(input_filepath) if tt in path]
        images = [data['images'] for data in content]
        labels = [data['labels'] for data in content]
        images, labels = np.concatenate(images), np.concatenate(labels).reshape(-1,1)

        # To tensor
        images = torch.from_numpy(images.astype(np.float32)) 
        labels = torch.from_numpy(labels.astype(np.float32)).type(torch.LongTensor)

        # Normalize
        mean, std = torch.mean(images), torch.std(images)
        images = (images-mean)/std

        # Save
        torch.save(images, f'{output_filepath}/{tt}_images.pt')
        torch.save(labels, f'{output_filepath}/{tt}_labels.pt')
if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()