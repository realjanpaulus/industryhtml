import argparse
from datetime import datetime
import logging
from pathlib import Path
import time

import justext
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from stop_words import get_stop_words


def parse_arguments():
    """Initialize argument parser and return arguments."""

    parser = argparse.ArgumentParser(
        prog="clf_pipeline", description="Pipeline for ML classification."
    )
    parser.add_argument(
        "--path", "-p", type=str, default="../data/", help="Path to dataset csv files."
    )
    parser.add_argument(
        "--clean_html",
        "-ch",
        action="store_true",
        help="Indicates if HTML boilerplate removal should be applied.",
    )
    parser.add_argument(
        "--specific_country",
        "-sc",
        type=str,
        default="",
        help="Load dataset with only given ISO2 country code.",
    )
    parser.add_argument(
        "--text_col",
        "-tc",
        type=str,
        default="html",
        help="Indicating the column with text.",
    )

    return parser.parse_args()


def main():

    ### HYPERPARAMETERS ###
    START_TIME = time.time()
    START_DATE = f"{datetime.now():%d.%m.%y}_{datetime.now():%H:%M:%S}"

    if len(args.specific_country) >= 1:
        LANG = "_" + args.specific_country
    else:
        LANG = ""
    DATA_DIR_PATH = args.path
    TRAIN_PATH_CSV = DATA_DIR_PATH + "train" + LANG + ".csv"
    TEST_PATH_CSV = DATA_DIR_PATH + "test" + LANG + ".csv"

    # "text" or "html"
    TEXT_COL = args.text_col

    # "group_representative", "group_representative_label", "industry", "industry_label", "group"
    CLASS_COL = "group_representative"
    CLASS_NAMES = "group_representative_label"

    # vectorizer settings
    MAX_DOCUMENT_FREQUENCY = 1.0
    MAX_FEATURES = None
    LOWERCASE = False
    STOP_WORDS = get_stop_words("de")

    # HTML boilerplate removal
    if TEXT_COL == "html":
        USE_CLEAN_HTML = args.clean_html

    ### LOGGER ###
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging_filename = f"logs/clf_pipeline{LANG} ({START_DATE}).log"
    logging.basicConfig(level=logging.DEBUG, filename=logging_filename, filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    ### PROGRAM DURATION ###
    DURATION = float(time.time() - START_TIME)
    logging.info(f"Run-time: {int(DURATION)/60} minute(s).")


if __name__ == "__main__":

    args = parse_arguments()
    main(args)
