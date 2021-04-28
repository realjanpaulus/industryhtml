import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import clean_html_boilerplate


def get_code(code_list, identifier):
    """ Get code by code list and identifier."""
    name = ""
    for entry in code_list:
        if entry["Code"] == identifier:
            name = entry["Description"]
            break
    return name


def get_execution_duration(duration):
    """ Calculate program execution. """
    hours = int(duration // 3600)
    duration = duration - 3600 * hours
    minutes = int(duration // 60)
    seconds = int(duration - 60 * minutes)

    def zero_digit(element):
        if element < 10:
            if element == 0:
                return "00"
            else:
                return f"0{element}"
        else:
            return element

    return f"Total duration: {zero_digit(hours)}:{zero_digit(minutes)}:{zero_digit(seconds)}"


def parse_arguments():
    """ Initialize argument parser and return arguments."""

    parser = argparse.ArgumentParser(
        prog="dataset_pipeline", description="Pipeline for creating datasets."
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default="../data/",
        help="Path to dataset ndjson file.",
    )
    parser.add_argument(
        "--clean_html",
        "-ch",
        action="store_true",
        help="Indicates if HTML boilerplate removal should be applied.",
    )
    parser.add_argument(
        "--ignore_country",
        "-ic",
        action="store_true",
        help="Ignore addition of ISO2 country code to shorten the runtime.",
    )
    parser.add_argument(
        "--specific_country",
        "-sc",
        type=str,
        default="",
        help="Limit dataset to given country, specified by ISO2 country code.",
    )
    parser.add_argument(
        "--test_size", "-ts", type=float, default=0.2, help="Set test size."
    )
    return parser.parse_args()


def main(args):
    ### hyperparameters ###
    START_TIME = time.time()
    START_DATE = f"{datetime.now():%d.%m.%y}_{datetime.now():%H:%M:%S}"

    if len(args.specific_country) >= 1:
        LANG = "_" + args.specific_country
    else:
        LANG = ""

    CLASS_COL = "group_representative"
    CLASS_COL_LABEL = "group_representative_label"

    DATA_DIR_PATH = args.path
    DATA_PATH_JSON = DATA_DIR_PATH + "data.ndjson"
    INDUSTRY_CODES_PATH = DATA_DIR_PATH + "industries"

    TRAIN_PATH_CSV = DATA_DIR_PATH + "train" + LANG + ".csv"
    TEST_PATH_CSV = DATA_DIR_PATH + "test" + LANG + ".csv"
    TEST_URL_TXT = args.path + "testurls" + LANG + ".txt"
    TEST_SIZE = args.test_size

    ### logger ###
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging_filename = f"logs/dataset_pipeline{LANG} ({START_DATE}).log"
    logging.basicConfig(level=logging.DEBUG, filename=logging_filename, filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    logging.info("Loading ndjson.")
    data_records = map(json.loads, open(DATA_PATH_JSON))
    data = pd.DataFrame.from_records(data_records)

    ### add country information ###
    if not args.ignore_country:
        logging.info("Extract and appending country information.")
        from langdetect import detect

        tmp_data = data.copy()
        tmp_data["country"] = tmp_data.apply(
            lambda row: detect(row.text).upper(), axis=1
        )
        data = tmp_data.copy()

    codes = pd.read_csv(INDUSTRY_CODES_PATH + ".csv")
    data[CLASS_COL_LABEL] = data.apply(
        lambda row: list(codes[codes["industry"] == row[CLASS_COL]].industry_label)[0],
        axis=1,
    )

    ### reduce dataset to specific country ###
    if args.specific_country != "":
        if args.specific_country in data["country"].unique():
            logging.info(
                f"NOTE: Only instances of the country {args.specific_country} will be kept."
            )
            data = data[data["country"] == args.specific_country]
        else:
            logging.info(
                f"WARNING: Country code {args.specific_country} is not in the dataset, \
                all countries will be kept!"
            )

    ### remove boilerplate html code ###
    if args.clean_html:
        logging.info("Cleaning HTML boilerplate...")
        data["html"] = data.apply(lambda row: clean_html_boilerplate(row), axis=1)

    logging.info("Splitting data.")
    train, test = train_test_split(
        data,
        test_size=TEST_SIZE,
        stratify=data[CLASS_COL],
        random_state=42,
    )

    test_urls = ",".join(test.url.tolist())

    ### save to csv ###
    logging.info("Saving data.")
    train.to_csv(TRAIN_PATH_CSV, index=False)
    test.to_csv(TEST_PATH_CSV, index=False)

    with open(TEST_URL_TXT, "w") as f:
        f.write(test_urls)

    ### program duration ###
    DURATION = float(time.time() - START_TIME)
    logging.info(f"Run-time: {int(DURATION)/60} minute(s).")


if __name__ == "__main__":

    args = parse_arguments()
    main(args)
