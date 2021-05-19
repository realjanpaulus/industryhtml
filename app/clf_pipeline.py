import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    cross_val_predict,
    cross_val_score,
    cross_validate,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from xgboost import XGBClassifier


### TODO ###
# welche columns laden?
# TODO: preprocessing direkt in Vectorizer implementieren
# TODO: Einstellungen
# TODO: max features?
# TODO: scoring
#       https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# TODO: XGBClf implementieren
# TODO: params updaten


def parse_arguments():
    """ Initialize argument parser and return arguments."""
    parser = argparse.ArgumentParser(
        prog="clf_pipeline", description="Pipeline for ML classification."
    )
    parser.add_argument(
        "--path", "-p", type=str, default="../data/", help="Path to dataset csv files."
    )
    parser.add_argument(
        "--cross_validation",
        "-cv",
        type=int,
        default=3,
        help="Sets the cross validation value (default: 3).",
    )
    parser.add_argument(
        "--data_clean",
        "-dc",
        action="store_true",
        help="Indicates if datatset with HTML, XHTML or XML boilerplate removal should be loaded.",
    )
    parser.add_argument(
        "--data_shortened",
        "-ds",
        type=int,
        default=None,
        help="Indicates if dataset with specific number of rows should be loaded (default: None).",
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=int,
        default=0,
        help="Indicates number of experiment.",
    )
    parser.add_argument(
        "--n_jobs",
        "-nj",
        type=int,
        default=1,
        help="Indicates the number of processors used for computation (default: 1).",
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
        help="Indicating the column with text (default: 'text').",
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

    with open("experiments.json") as f:
        experiments = json.load(f)

    EXPERIMENT = experiments[str(args.experiment)]
    TESTING = args.testing

    Path("../results").mkdir(parents=True, exist_ok=True)
    RESULTS_PATH = f"../results/{EXPERIMENT['name']}"
    Path(f"{RESULTS_PATH}").mkdir(parents=True, exist_ok=True)

    ### Time management ###
    START_TIME = time.time()
    START_DATE = f"{datetime.now():%d.%m.%y}_{datetime.now():%H:%M:%S}"

    CLEAN = ""
    if args.data_clean:
        CLEAN = "c"

    LANG = ""
    if len(args.specific_country) >= 1:
        LANG = "_" + args.specific_country

    ROWS = ""
    if args.data_shortened:
        ROWS = "_" + str(args.max_rows)

    ### Global variables ###
    DATA_DIR_PATH = args.path
    TRAIN_PATH_CSV = DATA_DIR_PATH + CLEAN + "train" + LANG + ROWS + ".csv"
    TEST_PATH_CSV = DATA_DIR_PATH + CLEAN + "test" + LANG + ROWS + ".csv"

    USECOLS = ["url", "group_representative_label", "html", "text", "meta"]
    TEXT_COL = args.text_col
    CLASS_COL = "group_representative_label"
    CLASS_NAMES = "group_representative_label"

    N_JOBS = args.n_jobs
    CV = args.cross_validation
    SCORING = {"f1": "f1_macro", "precision": "precision_macro"}

    ### LOGGER ###
    Path("logs").mkdir(parents=True, exist_ok=True)
    logging_filename = f"logs/clf_pipeline{LANG} ({START_DATE}).log"
    logging.basicConfig(level=logging.DEBUG, filename=logging_filename, filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    ### Load dataframes ###
    logging.info(f"Loading train and test dataframes.")
    if TESTING:
        train = pd.read_csv(TRAIN_PATH_CSV, nrows=100, usecols=USECOLS).fillna("")
        test = pd.read_csv(TEST_PATH_CSV, nrows=100, usecols=USECOLS).fillna("")
    else:
        train = pd.read_csv(TRAIN_PATH_CSV, usecols=USECOLS).fillna("")
        test = pd.read_csv(TEST_PATH_CSV, usecols=USECOLS).fillna("")
    logging.info(f"Loading took {int(float(time.time() - START_TIME))/60} minute(s).")

    X_train = train[TEXT_COL]
    y_train = train[CLASS_COL]
    y_train_labels = train[CLASS_NAMES]

    X_test = test[TEXT_COL]
    y_test = test[CLASS_COL]
    y_test_labels = test[CLASS_NAMES]

    # ======== #
    # Training #
    # ======== #

    models = [
        ("log", LogisticRegression()),
        ("svm", LinearSVC()),
        ("xgb", XGBClassifier()),
    ]

    if TESTING:
        models = [("svm", LinearSVC(), {"svm__C": [1]})]

    for model_name, model_obj, model_params in models:

        OUTPUT_NAME = f"_{EXPERIMENT['name']}_{model_name}"

        pipe = Pipeline(steps=[("vect", TfidfVectorizer()), (model_name, model_obj)])

        parameters = {
            "vect__lowercase": [True],
            "vect__preprocessor": [None],
            "vect__stop_words": [None],
            "vect__max_df": [1.0],
            "vect__min_df": [1],
            "vect__max_features": [None],
            "vect__norm": ["l2"],
            "vect__sublinear_tf": [True],
        }

        parameters.update(model_params)

        if TESTING:
            parameters = {
                "vect__lowercase": [True],
                "vect__preprocessor": [None],
                "vect__stop_words": [None],
                "vect__max_df": [1.0],
                "vect__min_df": [1],
                "vect__max_features": [100],
                "vect__norm": ["l2"],
                "vect__sublinear_tf": [True],
            }
            # todo
            # SCORING = {"F1": "f1_macro"}

        logging.info(f"Begin training of {model_name}.")

        ### grid search ###
        grid = GridSearchCV(
            pipe,
            parameters,
            cv=CV,
            error_score=0.0,
            n_jobs=N_JOBS,
            refit="precision",
            scoring=SCORING,
        )
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)

        # TODO: build custom classification report mit weiteren metriken
        ### classification report ###
        clf_report = classification_report(
            y_test,
            y_pred,
            target_names=np.unique(y_test_labels),
            zero_division=0,
            output_dict=True,
        )
        clf_report_df = pd.DataFrame(clf_report).transpose()
        clf_report_df.to_csv(f"{RESULTS_PATH}/clf_report_{OUTPUT_NAME}.csv")

        ### confusion matrix handling ###
        cm = confusion_matrix(y_train, y_pred)
        cm_df = pd.DataFrame(
            cm, index=np.unique(y_train_labels), columns=np.unique(y_train_labels)
        )
        cm_df.to_csv(f"{RESULTS_PATH}/cm_{OUTPUT_NAME}.csv")

        # TODO
        results = grid.cv_results_


        ### feature importance ###
        # todo: klappt coef_ für alle clf?
        feature_names = grid.best_estimator_.named_steps["vect"].get_feature_names()
        coefs_df = pd.DataFrame(
            grid.best_estimator_.named_steps[model_name].coef_,
            index=grid.best_estimator_.named_steps[model_name].classes_,
            columns=feature_names,
        )
        coefs_df.index.name = "class"
        coefs_df.to_csv(f"{RESULTS_PATH}/coefs_{OUTPUT_NAME}.csv")

        # TODO: RFE implementieren

    ### PROGRAM DURATION ###
    logging.info(f"Run-time: {int(float(time.time() - START_TIME))/60} minute(s).")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
