#!/usr/bin/env python
import logging
import click
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.data.make_dataset import make_dataset
from src.utils.prediction_utils import sample, transfer, sentransfer
from src.utils.eval_utils import evaluate


@click.group()
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    pass


main.add_command(make_dataset)
main.add_command(sample)
main.add_command(transfer)
main.add_command(sentransfer)
main.add_command(evaluate)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logging.getLogger('allennlp').setLevel(logging.WARNING)
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
