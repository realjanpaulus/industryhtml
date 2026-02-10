import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import time

from langdetect import detect
import numpy as np
import pandas as pd

pd.set_option("mode.chained_assignment", None)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

tqdm.pandas()

from utils import clean_boilerplate, extract_meta_informations, extract_tagtexts


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
        "--clean_boilerplate",
        "-cb",
        action="store_true",
        help="Indicates if HTML, XHTML or XML boilerplate removal should be applied.",
    )
    parser.add_argument(
        "--ignore_country",
        "-ic",
        action="store_true",
        help="Ignore addition of ISO2 country code to shorten the runtime.",
    )
    parser.add_argument(
        "--keep_html_col",
        "-khc",
        action="store_true",
        help="Keep the original html column.",
    )
    parser.add_argument(
        "--max_rows",
        "-mr",
        type=int,
        default=None,
        help="Sets maximum number of rows (default: None).",
    )
    parser.add_argument(
        "--pos_tagging", "-pt", action="store_true", help="Extracts nouns from text."
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
    parser.add_argument(
        "--use_test_txt",
        "-utt",
        action="store_true",
        help=(
            "Indicates if the train test split should be performed \
            on the basis of an existing txt file."
        ),
    )
    return parser.parse_args()


def main(args):
    ### hyperparameters ###
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

    POS_TAGGING = args.pos_tagging
    POS_NAME = ""
    if POS_TAGGING:
        POS_NAME = "p"

    KEEP_HTML_COL = args.keep_html_col

    CLASS_COL = "group_representative"
    CLASS_COL_LABEL = "group_representative_label"

    DATA_DIR_PATH = args.path
    DATA_PATH_JSON = DATA_DIR_PATH + "data.ndjson"
    INDUSTRY_CODES_PATH = DATA_DIR_PATH + "industries"

    TRAIN_PATH_CSV = DATA_DIR_PATH + CLEAN + POS_NAME + "train" + LANG + ROWS + ".csv"
    TEST_PATH_CSV = DATA_DIR_PATH + CLEAN + POS_NAME + "test" + LANG + ROWS + ".csv"
    TEST_URL_TXT = args.path + CLEAN + POS_NAME + "testurls" + LANG + ROWS + ".txt"
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
    data = pd.DataFrame.from_records(data_records, nrows=args.max_rows)
    data = data.drop_duplicates()

    ### add country information ###
    if not args.ignore_country:
        logging.info("Extract and appending country information.")
        data["country"] = data.progress_apply(
            lambda row: detect(row.text).upper(), axis=1
        )

    ### add industry codes ###
    codes = pd.read_csv(INDUSTRY_CODES_PATH + ".csv")
    data[CLASS_COL_LABEL] = data.progress_apply(
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
    HTML_COL = "html"
    if args.clean_boilerplate:
        logging.info("Cleaning HTML/XHTML/XML boilerplate...")
        HTML_COL = "chtml"
        data[HTML_COL] = data.progress_apply(
            lambda row: clean_boilerplate(row.html, row.url), axis=1
        )

    # ============================== #
    # Add element content as columns #
    # ============================== #

    ### extract meta element ###
    logging.info("Extract informations from <meta> elements.")
    data["<meta>_title"] = data.progress_apply(
        lambda row: extract_meta_informations(row.html, "title"),
        axis=1,
    )
    data["<meta>_keywords"] = data.progress_apply(
        lambda row: extract_meta_informations(row.html, "keywords"),
        axis=1,
    )
    data["<meta>_description"] = data.progress_apply(
        lambda row: extract_meta_informations(row.html, "description"),
        axis=1,
    )
    ### remove meta title from text ###
    data["text"] = data.progress_apply(
        lambda row: row.text.replace(row["<meta>_title"], ""),
        axis=1,
    )

    ### extract more element content ###
    logging.info("Extract infos from other elements.")
    data["<title>"] = data.progress_apply(
        lambda row: extract_tagtexts(row[HTML_COL], "title"), axis=1
    )
    data["<h1>"] = data.progress_apply(
        lambda row: extract_tagtexts(row[HTML_COL], "h1"), axis=1
    )
    data["<h2>"] = data.progress_apply(
        lambda row: extract_tagtexts(row[HTML_COL], "h2"), axis=1
    )
    data["<h3>"] = data.progress_apply(
        lambda row: extract_tagtexts(row[HTML_COL], "h3"), axis=1
    )
    data["<h4>"] = data.progress_apply(
        lambda row: extract_tagtexts(row[HTML_COL], "h4"), axis=1
    )
    data["<h5>"] = data.progress_apply(
        lambda row: extract_tagtexts(row[HTML_COL], "h5"), axis=1
    )
    data["<h6>"] = data.progress_apply(
        lambda row: extract_tagtexts(row[HTML_COL], "h6"), axis=1
    )
    data["<b>"] = data.progress_apply(
        lambda row: extract_tagtexts(row[HTML_COL], "b"), axis=1
    )
    data["<strong>"] = data.progress_apply(
        lambda row: extract_tagtexts(row[HTML_COL], "strong"), axis=1
    )
    data["<em>"] = data.progress_apply(
        lambda row: extract_tagtexts(row[HTML_COL], "em"), axis=1
    )
    data["<i>"] = data.progress_apply(
        lambda row: extract_tagtexts(row[HTML_COL], "i"), axis=1
    )
    data["<p>"] = data.progress_apply(
        lambda row: extract_tagtexts(row[HTML_COL], "p"), axis=1
    )
    data["<a>"] = data.progress_apply(
        lambda row: extract_tagtexts(row[HTML_COL], "a"), axis=1
    )
    data["<li>"] = data.progress_apply(
        lambda row: extract_tagtexts(row[HTML_COL], "li"), axis=1
    )
    data["<div>"] = data.progress_apply(
        lambda row: extract_tagtexts(row[HTML_COL], "div", no_inner=True), axis=1
    )

    ### remove useless columns ###
    possible_columns = [
        "level_0",
        "index",
        "industry",
        "industry_label",
        "group",
        "source",
    ]
    if not KEEP_HTML_COL:
        possible_columns.append("html")

    for col in possible_columns:
        if col in data:
            data = data.drop([col], axis=1)

    columns = [
        "url",
        "group_representative",
        "group_representative_label",
        "text",
        HTML_COL,
        "<meta>_title",
        "<meta>_keywords",
        "<meta>_description",
        "<title>",
        "<h1>",
        "<h2>",
        "<h3>",
        "<h4>",
        "<h5>",
        "<h6>",
        "<b>",
        "<strong>",
        "<em>",
        "<i>",
        "<p>",
        "<a>",
        "<li>",
        "<div>",
    ]
    if not args.ignore_country:
        columns.append("country")

    if KEEP_HTML_COL:
        columns.append("html")

    data = data.reindex(columns=columns)

    logging.info("Splitting data.")

    ### load tests from testurl txt file ###
    testurls_file = Path(TEST_URL_TXT)
    if args.use_test_txt and testurls_file.is_file():
        logging.info(
            f"The testurl txt-file '{TEST_URL_TXT}' will be used for train-test split."
        )
        with open(testurls_file) as f:
            testurls_str = f.read()
            testurls = testurls_str.split(",")
        train = data[~data["url"].isin(testurls)]
        test = data[data["url"].isin(testurls)]
    else:
        if args.use_test_txt:
            logging.info(
                f"The testurl txt-file '{TEST_URL_TXT}' doesn't exist. A new split will be made."
            )
        train, test = train_test_split(
            data,
            test_size=TEST_SIZE,
            # stratify=data[CLASS_COL],
            random_state=42,
        )

    test_urls = ",".join(test.url.tolist())

    ### use pos tagging ###
    if POS_TAGGING:
        logging.info("Extracting nouns from text columns.")

        from utils import filter_nouns

        logging.getLogger("flair").setLevel(logging.ERROR)

        train["text"] = train.progress_apply(
            lambda row: filter_nouns(row["text"]), axis=1
        )
        train["<meta>_title"] = train.progress_apply(
            lambda row: filter_nouns(row["<meta>_title"]), axis=1
        )
        train["<meta>_keywords"] = train.progress_apply(
            lambda row: filter_nouns(row["<meta>_keywords"]), axis=1
        )
        train["<meta>_description"] = train.progress_apply(
            lambda row: filter_nouns(row["<meta>_description"]), axis=1
        )
        train["<title>"] = train.progress_apply(
            lambda row: filter_nouns(row["<title>"]), axis=1
        )
        train["<h1>"] = train.progress_apply(
            lambda row: filter_nouns(row["<h1>"]), axis=1
        )
        train["<h2>"] = train.progress_apply(
            lambda row: filter_nouns(row["<h2>"]), axis=1
        )
        train["<h3>"] = train.progress_apply(
            lambda row: filter_nouns(row["<h3>"]), axis=1
        )
        train["<h4>"] = train.progress_apply(
            lambda row: filter_nouns(row["<h4>"]), axis=1
        )
        train["<h5>"] = train.progress_apply(
            lambda row: filter_nouns(row["<h5>"]), axis=1
        )
        train["<h6>"] = train.progress_apply(
            lambda row: filter_nouns(row["<h6>"]), axis=1
        )
        train["<b>"] = train.progress_apply(
            lambda row: filter_nouns(row["<b>"]), axis=1
        )
        train["<strong>"] = train.progress_apply(
            lambda row: filter_nouns(row["<strong>"]), axis=1
        )
        train["<em>"] = train.progress_apply(
            lambda row: filter_nouns(row["<em>"]), axis=1
        )
        train["<i>"] = train.progress_apply(
            lambda row: filter_nouns(row["<i>"]), axis=1
        )
        train["<p>"] = train.progress_apply(
            lambda row: filter_nouns(row["<p>"]), axis=1
        )
        train["<a>"] = train.progress_apply(
            lambda row: filter_nouns(row["<a>"]), axis=1
        )
        train["<li>"] = train.progress_apply(
            lambda row: filter_nouns(row["<li>"]), axis=1
        )
        train["<div>"] = train.progress_apply(
            lambda row: filter_nouns(row["<div>"]), axis=1
        )

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
