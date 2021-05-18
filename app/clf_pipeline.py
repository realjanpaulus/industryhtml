import argparse
from datetime import datetime
import logging
from pathlib import Path
import time

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


# TODO: alle arguments nützlich?
# TODO: methode arg oder sowas einführen
def parse_arguments():
    """ Initialize argument parser and return arguments."""

    parser = argparse.ArgumentParser(
        prog="clf_pipeline", description="Pipeline for ML classification."
    )
    parser.add_argument(
        "--path", "-p", type=str, default="../data/", help="Path to dataset csv files."
    )
    parser.add_argument(
        "--clean_boilerplate",
        "-cb",
        action="store_true",
        help="Indicates if HTML, XHTML or XML boilerplate removal should be applied.",
    )
    parser.add_argument(
        "--max_rows",
        "-mr",
        type=int,
        default=None,
        help="Sets maximum number of rows (default: None).",
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
        default="text",
        help="Indicating the column with text.",
    )
    parser.add_argument(
        "--testing",
        "-t",
        action="store_true",
        help="Starts testing mode with a small subset of the corpus \
						and no tunable parameters.",
    )

    return parser.parse_args()


def main(args):

    # ===== #
    # Setup #
    # ===== #

    ### HYPERPARAMETERS ###
    START_TIME = time.time()
    START_DATE = f"{datetime.now():%d.%m.%y}_{datetime.now():%H:%M:%S}"


    ROWS = ""
    if args.max_rows:
        ROWS = "_" + str(args.max_rows)

    LANG = ""
    if len(args.specific_country) >= 1:
        LANG = "_" + args.specific_country

    CLEAN = ""
    if args.clean_boilerplate:
        CLEAN = "c"

    DATA_DIR_PATH = args.path
    TRAIN_PATH_CSV = DATA_DIR_PATH + CLEAN + "train" + LANG + ROWS + ".csv"
    TEST_PATH_CSV = DATA_DIR_PATH + CLEAN + "test" + LANG + ROWS + ".csv"

    ### LOGGER ###
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging_filename = f"logs/clf_pipeline{LANG} ({START_DATE}).log"
    logging.basicConfig(level=logging.DEBUG, filename=logging_filename, filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    # TEXT ("text", "html", "chtml") and CLASS columns
    TEXT_COL = args.text_col
    CLASS_COL = "group_representative_label"
    CLASS_NAMES = "group_representative_label"

    # load dataframes
    logging.info("Loading train and test dataframes.")
    # TODO: welche columsn laden?
    USECOLS = ["url", "group_representative_label", "html", "text", "meta"]
    if args.testing:
        train = pd.read_csv(TRAIN_PATH_CSV, nrows=100, usecols=USECOLS).fillna("")
        test = pd.read_csv(TEST_PATH_CSV, nrows=100, usecols=USECOLS).fillna("")
    else:
        train = pd.read_csv(TRAIN_PATH_CSV, usecols=USECOLS).fillna("")
        test = pd.read_csv(TEST_PATH_CSV, usecols=USECOLS).fillna("")


    X_train = train[TEXT_COL]
    y_train = train[CLASS_COL]
    y_train_labels = train[CLASS_NAMES]

    X_test = test[TEXT_COL]
    y_test = test[CLASS_COL]
    y_test_labels = test[CLASS_NAMES]

    # vectorizer settings
    MAX_DOCUMENT_FREQUENCY = 1.0
    MAX_FEATURES = None
    LOWERCASE = False
    STOP_WORDS = None


    # ======== #
    # Training #
    # ======== #


    # TODO: Einstellungen
    vectorizer = TfidfVectorizer()

    pipe = Pipeline(steps=[("vect", vectorizer), ("clf", LinearSVC())])



    ### PROGRAM DURATION ###
    DURATION = float(time.time() - START_TIME)
    logging.info(f"Run-time: {int(DURATION)/60} minute(s).")


if __name__ == "__main__":

    args = parse_arguments()
    main(args)
