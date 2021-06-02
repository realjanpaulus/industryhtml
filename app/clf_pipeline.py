import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import time

import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from xgboost import XGBClassifier


### TODO ###
# welche columns laden?
# TODO: Einstellungen
# TODO: scoring
#       https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# TODO: params updaten von tfidf
# TODO: build custom classification report mit weiteren metriken
#################################################################################
# TODO: RFE implementieren
# TODO: Permutation? klappt das überhaupt?
# https://scikit-learn.org/stable/modules/generated/sklearn.inspection.permutation_importance.html
#################################################################################


class DataFrameColumnExtracter(TransformerMixin):
    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]


class FNPipeline(Pipeline):
    def get_feature_names(self):
        for name, step in self.steps:
            if isinstance(step, TfidfVectorizer):
                return step.get_feature_names()

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
        help="Load dataset with only given ISO2 country code (default: '' = all).",
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

    ### Time management ###
    START_TIME = time.time()
    START_DATE = f"{datetime.now():%d.%m.%y}_{datetime.now():%H:%M:%S}"

    with open("experiments.json") as f:
        experiments = json.load(f)

    EXPERIMENT_N = args.experiment
    EXPERIMENT = experiments[str(EXPERIMENT_N)]
    USECOLS = EXPERIMENT["cols"]
    TESTING = args.testing

    Path("../results").mkdir(parents=True, exist_ok=True)
    RESULTS_PATH = f"../results/{EXPERIMENT['name']}"
    Path(f"{RESULTS_PATH}").mkdir(parents=True, exist_ok=True)

    ### Data variant selection ###
    CLEAN = ""
    if args.data_clean:
        CLEAN = "c"

    LANG = ""
    if len(args.specific_country) >= 1:
        LANG = "_" + args.specific_country

    ROWS = ""
    if args.data_shortened:
        ROWS = "_" + str(args.data_shortened)

    ### Global variables ###
    DATA_DIR_PATH = args.path
    TRAIN_PATH_CSV = DATA_DIR_PATH + CLEAN + "train" + LANG + ROWS + ".csv"
    TEST_PATH_CSV = DATA_DIR_PATH + CLEAN + "test" + LANG + ROWS + ".csv"

    TEXT_COL = args.text_col
    CLASS_COL = "group_representative_label"
    CLASS_NAMES = "group_representative_label"

    N_JOBS = args.n_jobs
    CV = args.cross_validation
    SCORING = {
        "f1": make_scorer(f1_score, average="macro", zero_division=0),
        "precision": make_scorer(precision_score, average="macro", zero_division=0),
    }

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

    X_train = train
    y_train = train[CLASS_COL]
    y_train_labels = train[CLASS_NAMES]

    X_test = test
    y_test = test[CLASS_COL]
    y_test_labels = test[CLASS_NAMES]

    # ======== #
    # Training #
    # ======== #

    models = [
        ("logreg", LogisticRegression(n_jobs=N_JOBS)),
        ("svm", LinearSVC()),
        ("xgb_tree", XGBClassifier(booster="gbtree", n_jobs=N_JOBS)),
        ("xgb_linear", XGBClassifier(booster="gblinear", n_jobs=N_JOBS)),
    ]

    if TESTING:
        models = [
            ("logreg", LogisticRegression(n_jobs=N_JOBS)),
            ("svm", LinearSVC()),
            ("xgb_tree", XGBClassifier(booster="gbtree", n_jobs=N_JOBS)),
            ("xgb_linear", XGBClassifier(booster="gblinear", n_jobs=N_JOBS)),
        ]

    # ========== #
    # Model loop #
    # ========== #
    for model_name, model_obj in models:

        if model_name.startswith("xgb"):
            import warnings
            warnings.filterwarnings("ignore")

        OUTPUT_NAME = f"_{EXPERIMENT['name']}_{model_name}"

        ### Experiment setup ###
        if EXPERIMENT_N == 0:
            logging.info("Experiment: Plain text.")
            pipe = Pipeline([
                ("features", FeatureUnion([
                    ("plain", FNPipeline([
                        ("extract_text", DataFrameColumnExtracter("text")),
                        ("plain_vect", TfidfVectorizer()),
                    ]))
                ])),
                (model_name, model_obj),
            ])
        elif EXPERIMENT_N == 1:
            logging.info("Experiment: Meta element addition.")
            pipe = Pipeline([
                ("features", FeatureUnion([
                    ("plain", Pipeline([
                        ("extract_text", DataFrameColumnExtracter("text")),
                        ("plain_vect", TfidfVectorizer()),
                    ])),
                    ("meta", Pipeline([
                        ("extract_meta", DataFrameColumnExtracter("meta")),
                        ("meta_vect", TfidfVectorizer()),
                    ]))
                ], n_jobs=N_JOBS)),
                (model_name, model_obj),
            ])


        ### Hyperparamter optimization ###
        if TESTING:
            parameters = {
                "features__plain__plain_vect__lowercase": [True],
                "features__plain__plain_vect__preprocessor": [None],
                "features__plain__plain_vect__stop_words": [None],
                "features__plain__plain_vect__max_df": [1.0],
                "features__plain__plain_vect__min_df": [1],
                "features__plain__plain_vect__max_features": [None],
                "features__plain__plain_vect__norm": ["l2"],
                "features__plain__plain_vect__sublinear_tf": [True],
            }
        else:
            parameters = {
                "features__plain__plain_vect__lowercase": [True],
                "features__plain__plain_vect__preprocessor": [None],
                "features__plain__plain_vect__stop_words": [None],
                "features__plain__plain_vect__max_df": [0.25, 0.5, 0.75, 1.0],
                "features__plain__plain_vect__min_df": [1],
                "features__plain__plain_vect__max_features": [100, 1000, 10000, None],
                "features__plain__plain_vect__norm": ["l1", "l2"],
                "features__plain__plain_vect__sublinear_tf": [True],
            }


        logging.info(f"Begin training of {model_name}.")

        ### Grid search ###
        grid = GridSearchCV(
            pipe,
            parameters,
            cv=CV,
            error_score=0.0,
            n_jobs=N_JOBS,
            refit="precision",
            scoring=SCORING,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        ### Save best params ###
        best_params = grid.best_params_
        with open(f"{RESULTS_PATH}/bestparams_{OUTPUT_NAME}.json", "w+") as f:
            json.dump(best_params, f)


        ### Prediction & Classification report ###
        y_pred = grid.predict(X_test)

        clf_report = classification_report(
            y_test,
            y_pred,
            zero_division=0,
            output_dict=True,
        )
        clf_report_df = pd.DataFrame(clf_report).transpose()
        clf_report_df.to_csv(f"{RESULTS_PATH}/clfreport_{OUTPUT_NAME}.csv")

        ### Confusion matrix handling ###
        cm = confusion_matrix(y_train, y_pred)
        cm_df = pd.DataFrame(
            cm, index=np.unique(y_train_labels), columns=np.unique(y_train_labels)
        )
        cm_df.to_csv(f"{RESULTS_PATH}/cm_{OUTPUT_NAME}.csv")

        ### Feature importance ###
        tmp = dict(grid.best_estimator_.named_steps['features'].transformer_list).get("plain")
        feature_names = tmp.get_feature_names()

        if model_name == "xgb_tree":
            features_d = dict(
                zip(
                    feature_names,
                    grid.best_estimator_.named_steps["xgb_tree"].feature_importances_,
                )
            )
            features_df = pd.DataFrame(features_d.items(), columns=["feature", "value"])
            features_df = features_df.sort_values(by="value", ascending=False)
            features_df.to_csv(f"{RESULTS_PATH}/fi_{OUTPUT_NAME}.csv", index=False)
        else:
            coefs_df = pd.DataFrame(
                grid.best_estimator_.named_steps[model_name].coef_,
                index=grid.best_estimator_.named_steps[model_name].classes_,
                columns=feature_names,
            )
            coefs_df.index.name = "class"
            coefs_df.to_csv(f"{RESULTS_PATH}/coefs_{OUTPUT_NAME}.csv")


    ### PROGRAM DURATION ###
    logging.info(f"Run-time: {int(float(time.time() - START_TIME))/60} minute(s).")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
