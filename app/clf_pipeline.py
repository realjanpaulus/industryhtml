import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score,
    make_scorer,
    precision_score,
    recall_score
)
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
        for _, step in self.steps:
            if isinstance(step, TfidfVectorizer):
                return step.get_feature_names()

def get_experiment_pipeline(n, model_name, model_obj, text_col, vectorizer):
    """ Get experiment pipeline by number."""
    pipe = []
    if n == 0:
        pipe = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "plain",
                                FNPipeline(
                                    [
                                        (
                                            "extract_text",
                                            DataFrameColumnExtracter(text_col),
                                        ),
                                        ("plain_vect", vectorizer),
                                    ]
                                ),
                            )
                        ]
                    ),
                ),
                (model_name, model_obj),
            ]
        )
    elif n == 1:
        pipe = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "plain",
                                FNPipeline(
                                    [
                                        (
                                            "extract_text",
                                            DataFrameColumnExtracter(text_col),
                                        ),
                                        ("plain_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "meta_title",
                                FNPipeline(
                                    [
                                        (
                                            "extract_meta_title",
                                            DataFrameColumnExtracter(
                                                "<meta>_title"
                                            ),
                                        ),
                                        ("meta_title_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "meta_keywords",
                                FNPipeline(
                                    [
                                        (
                                            "extract_meta_keywords",
                                            DataFrameColumnExtracter(
                                                "<meta>_keywords"
                                            ),
                                        ),
                                        ("meta_keywords_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "meta_description",
                                FNPipeline(
                                    [
                                        (
                                            "extract_meta_description",
                                            DataFrameColumnExtracter(
                                                "<meta>_description"
                                            ),
                                        ),
                                        ("meta_description_vect", vectorizer),
                                    ]
                                ),
                            ),
                        ],
                        n_jobs=N_JOBS,
                    ),
                ),
                (model_name, model_obj),
            ]
        )
    elif n == 101:
        pipe = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "meta_title",
                                FNPipeline(
                                    [
                                        (
                                            "extract_meta_title",
                                            DataFrameColumnExtracter(
                                                "<meta>_title"
                                            ),
                                        ),
                                        ("meta_title_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "meta_keywords",
                                FNPipeline(
                                    [
                                        (
                                            "extract_meta_keywords",
                                            DataFrameColumnExtracter(
                                                "<meta>_keywords"
                                            ),
                                        ),
                                        ("meta_keywords_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "meta_description",
                                FNPipeline(
                                    [
                                        (
                                            "extract_meta_description",
                                            DataFrameColumnExtracter(
                                                "<meta>_description"
                                            ),
                                        ),
                                        ("meta_description_vect", vectorizer),
                                    ]
                                ),
                            ),
                        ],
                        n_jobs=N_JOBS,
                    ),
                ),
                (model_name, model_obj),
            ]
        )
    elif n == 2:
        pipe = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "plain",
                                FNPipeline(
                                    [
                                        (
                                            "extract_text",
                                            DataFrameColumnExtracter(text_col),
                                        ),
                                        ("plain_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "meta_title",
                                FNPipeline(
                                    [
                                        (
                                            "extract_meta_title",
                                            DataFrameColumnExtracter(
                                                "<meta>_title"
                                            ),
                                        ),
                                        ("meta_title_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "meta_keywords",
                                FNPipeline(
                                    [
                                        (
                                            "extract_meta_keywords",
                                            DataFrameColumnExtracter(
                                                "<meta>_keywords"
                                            ),
                                        ),
                                        ("meta_keywords_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "meta_description",
                                FNPipeline(
                                    [
                                        (
                                            "extract_meta_description",
                                            DataFrameColumnExtracter(
                                                "<meta>_description"
                                            ),
                                        ),
                                        ("meta_description_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "title",
                                FNPipeline(
                                    [
                                        (
                                            "extract_title",
                                            DataFrameColumnExtracter(
                                                "<title>"
                                            ),
                                        ),
                                        ("title_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h1",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h1",
                                            DataFrameColumnExtracter(
                                                "<h1>"
                                            ),
                                        ),
                                        ("h1_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h2",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h2",
                                            DataFrameColumnExtracter(
                                                "<h2>"
                                            ),
                                        ),
                                        ("h2_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h3",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h3",
                                            DataFrameColumnExtracter(
                                                "<h3>"
                                            ),
                                        ),
                                        ("h3_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h4",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h4",
                                            DataFrameColumnExtracter(
                                                "<h4>"
                                            ),
                                        ),
                                        ("h4_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h5",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h5",
                                            DataFrameColumnExtracter(
                                                "<h5>"
                                            ),
                                        ),
                                        ("h5_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h6",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h6",
                                            DataFrameColumnExtracter(
                                                "<h6>"
                                            ),
                                        ),
                                        ("h6_vect", vectorizer),
                                    ]
                                ),
                            ),
                        ],
                        n_jobs=N_JOBS,
                    ),
                ),
                (model_name, model_obj),
            ]
        )
    elif n == 201:
        pipe = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "meta_title",
                                FNPipeline(
                                    [
                                        (
                                            "extract_meta_title",
                                            DataFrameColumnExtracter(
                                                "<meta>_title"
                                            ),
                                        ),
                                        ("meta_title_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "meta_keywords",
                                FNPipeline(
                                    [
                                        (
                                            "extract_meta_keywords",
                                            DataFrameColumnExtracter(
                                                "<meta>_keywords"
                                            ),
                                        ),
                                        ("meta_keywords_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "meta_description",
                                FNPipeline(
                                    [
                                        (
                                            "extract_meta_description",
                                            DataFrameColumnExtracter(
                                                "<meta>_description"
                                            ),
                                        ),
                                        ("meta_description_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "title",
                                FNPipeline(
                                    [
                                        (
                                            "extract_title",
                                            DataFrameColumnExtracter(
                                                "<title>"
                                            ),
                                        ),
                                        ("title_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h1",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h1",
                                            DataFrameColumnExtracter(
                                                "<h1>"
                                            ),
                                        ),
                                        ("h1_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h2",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h2",
                                            DataFrameColumnExtracter(
                                                "<h2>"
                                            ),
                                        ),
                                        ("h2_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h3",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h3",
                                            DataFrameColumnExtracter(
                                                "<h3>"
                                            ),
                                        ),
                                        ("h3_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h4",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h4",
                                            DataFrameColumnExtracter(
                                                "<h4>"
                                            ),
                                        ),
                                        ("h4_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h5",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h5",
                                            DataFrameColumnExtracter(
                                                "<h5>"
                                            ),
                                        ),
                                        ("h5_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h6",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h6",
                                            DataFrameColumnExtracter(
                                                "<h6>"
                                            ),
                                        ),
                                        ("h6_vect", vectorizer),
                                    ]
                                ),
                            ),
                        ],
                        n_jobs=N_JOBS,
                    ),
                ),
                (model_name, model_obj),
            ]
        )
    elif n == 3:
        pipe = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "plain",
                                FNPipeline(
                                    [
                                        (
                                            "extract_text",
                                            DataFrameColumnExtracter(text_col),
                                        ),
                                        ("plain_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "title",
                                FNPipeline(
                                    [
                                        (
                                            "extract_title",
                                            DataFrameColumnExtracter(
                                                "<title>"
                                            ),
                                        ),
                                        ("title_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h1",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h1",
                                            DataFrameColumnExtracter(
                                                "<h1>"
                                            ),
                                        ),
                                        ("h1_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h2",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h2",
                                            DataFrameColumnExtracter(
                                                "<h2>"
                                            ),
                                        ),
                                        ("h2_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h3",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h3",
                                            DataFrameColumnExtracter(
                                                "<h3>"
                                            ),
                                        ),
                                        ("h3_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "b",
                                FNPipeline(
                                    [
                                        (
                                            "extract_b",
                                            DataFrameColumnExtracter(
                                                "<b>"
                                            ),
                                        ),
                                        ("hb_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "strong",
                                FNPipeline(
                                    [
                                        (
                                            "extract_strong",
                                            DataFrameColumnExtracter(
                                                "<strong>"
                                            ),
                                        ),
                                        ("strong_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "em",
                                FNPipeline(
                                    [
                                        (
                                            "extract_em",
                                            DataFrameColumnExtracter(
                                                "<em>"
                                            ),
                                        ),
                                        ("em_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "i",
                                FNPipeline(
                                    [
                                        (
                                            "extract_i",
                                            DataFrameColumnExtracter(
                                                "<i>"
                                            ),
                                        ),
                                        ("i_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "p",
                                FNPipeline(
                                    [
                                        (
                                            "extract_p",
                                            DataFrameColumnExtracter(
                                                "<p>"
                                            ),
                                        ),
                                        ("p_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "a",
                                FNPipeline(
                                    [
                                        (
                                            "extract_a",
                                            DataFrameColumnExtracter(
                                                "<a>"
                                            ),
                                        ),
                                        ("a_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "li",
                                FNPipeline(
                                    [
                                        (
                                            "extract_li",
                                            DataFrameColumnExtracter(
                                                "<li>"
                                            ),
                                        ),
                                        ("li_vect", vectorizer),
                                    ]
                                ),
                            ),
                        ],
                        n_jobs=N_JOBS,
                    ),
                ),
                (model_name, model_obj),
            ]
        )
    elif n == 301:
        pipe = Pipeline(
            [
                (
                    "features",
                    FeatureUnion(
                        [
                            (
                                "title",
                                FNPipeline(
                                    [
                                        (
                                            "extract_title",
                                            DataFrameColumnExtracter(
                                                "<title>"
                                            ),
                                        ),
                                        ("title_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h1",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h1",
                                            DataFrameColumnExtracter(
                                                "<h1>"
                                            ),
                                        ),
                                        ("h1_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h2",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h2",
                                            DataFrameColumnExtracter(
                                                "<h2>"
                                            ),
                                        ),
                                        ("h2_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "h3",
                                FNPipeline(
                                    [
                                        (
                                            "extract_h3",
                                            DataFrameColumnExtracter(
                                                "<h3>"
                                            ),
                                        ),
                                        ("h3_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "b",
                                FNPipeline(
                                    [
                                        (
                                            "extract_b",
                                            DataFrameColumnExtracter(
                                                "<b>"
                                            ),
                                        ),
                                        ("hb_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "strong",
                                FNPipeline(
                                    [
                                        (
                                            "extract_strong",
                                            DataFrameColumnExtracter(
                                                "<strong>"
                                            ),
                                        ),
                                        ("strong_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "em",
                                FNPipeline(
                                    [
                                        (
                                            "extract_em",
                                            DataFrameColumnExtracter(
                                                "<em>"
                                            ),
                                        ),
                                        ("em_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "i",
                                FNPipeline(
                                    [
                                        (
                                            "extract_i",
                                            DataFrameColumnExtracter(
                                                "<i>"
                                            ),
                                        ),
                                        ("i_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "p",
                                FNPipeline(
                                    [
                                        (
                                            "extract_p",
                                            DataFrameColumnExtracter(
                                                "<p>"
                                            ),
                                        ),
                                        ("p_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "a",
                                FNPipeline(
                                    [
                                        (
                                            "extract_a",
                                            DataFrameColumnExtracter(
                                                "<a>"
                                            ),
                                        ),
                                        ("a_vect", vectorizer),
                                    ]
                                ),
                            ),
                            (
                                "li",
                                FNPipeline(
                                    [
                                        (
                                            "extract_li",
                                            DataFrameColumnExtracter(
                                                "<li>"
                                            ),
                                        ),
                                        ("li_vect", vectorizer),
                                    ]
                                ),
                            ),
                        ],
                        n_jobs=N_JOBS,
                    ),
                ),
                (model_name, model_obj),
            ]
        )

    return pipe


def parse_arguments():
    """ Initialize argument parser and return arguments."""
    parser = argparse.ArgumentParser(
        prog="clf_pipeline", description="Pipeline for ML classification."
    )
    parser.add_argument(
        "--path", "-p", type=str, default="../data/", help="Path to dataset csv files."
    )
    parser.add_argument(
        "--data_clean",
        "-dc",
        action="store_true",
        help="Indicates if datatset with HTML, XHTML or XML boilerplate removal should be loaded."
    )
    parser.add_argument(
        "--data_shortened",
        "-ds",
        type=int,
        default=None,
        help="Indicates if dataset with specific number of rows should be loaded (default: None)."
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=int,
        default=0,
        help="Indicates number of experiment."
    )
    parser.add_argument(
        "--n_jobs",
        "-nj",
        type=int,
        default=1,
        help="Indicates the number of processors used for computation (default: 1)."
    )
    parser.add_argument(
        "--specific_country",
        "-sc",
        type=str,
        default="",
        help="Load dataset with only given ISO2 country code (default: '' = all)."
    )
    parser.add_argument(
        "--text_col",
        "-tc",
        type=str,
        default="text",
        help="Indicating the column with text (default: 'text')."
    )
    parser.add_argument(
        "--testing",
        "-t",
        action="store_true",
        help="Starts testing mode with a small subset of the corpus \
						and no tunable parameters."
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
    EXPERIMENT_INFO = EXPERIMENT["info"]
    USECOLS = EXPERIMENT["cols"]
    TESTING = args.testing

    results_dir = "results"
    if TESTING:
        results_dir = "testresults"
    Path(f"../{results_dir}").mkdir(parents=True, exist_ok=True)
    RESULTS_PATH = f"../{results_dir}/{EXPERIMENT['name']}"
    Path(f"{RESULTS_PATH}").mkdir(parents=True, exist_ok=True)

    ### Data variant selection ###
    CLEAN = ""
    if args.data_clean:
        CLEAN = "c"

    LANG = ""
    if len(args.specific_country) >= 1:
        LANG = "_" + args.specific_country

    NROWS = None
    if args.data_shortened:
        NROWS = args.data_shortened

    ### Global variables ###
    DATA_DIR_PATH = args.path
    TRAIN_PATH_CSV = DATA_DIR_PATH + CLEAN + "train" + LANG + ".csv"
    TEST_PATH_CSV = DATA_DIR_PATH + CLEAN + "test" + LANG + ".csv"

    TEXT_COL = args.text_col
    CLASS_COL = "group_representative_label"
    CLASS_NAMES = "group_representative_label"

    N_JOBS = args.n_jobs

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
        train = pd.read_csv(
            TRAIN_PATH_CSV, nrows=1000, usecols=USECOLS, lineterminator="\n"
        ).fillna("")
        test = pd.read_csv(
            TEST_PATH_CSV, nrows=1000, usecols=USECOLS, lineterminator="\n"
        ).fillna("")
    else:
        train = pd.read_csv(
            TRAIN_PATH_CSV, nrows=NROWS, usecols=USECOLS, lineterminator="\n"
        ).fillna("")
        test = pd.read_csv(
            TEST_PATH_CSV, nrows=NROWS, usecols=USECOLS, lineterminator="\n"
        ).fillna("")
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
        (
            "svm",
            LinearSVC(loss="squared_hinge", penalty="l2")
        ),
        (
            "xgb_tree",
            XGBClassifier(
                booster="gbtree",
                n_estimators=100,
                n_jobs=N_JOBS,
                objective="multi:softmax",
                verbosity=0,
            )
        )
    ]

    # ========== #
    # Model loop #
    # ========== #

    logging.info(f"Experiment {EXPERIMENT_N}: {EXPERIMENT_INFO}")
    for model_name, model_obj in models:
        LOOP_TIME = time.time()

        OUTPUT_NAME = f"_{EXPERIMENT['name']}_{model_name}"
        VECTORIZER = TfidfVectorizer(sublinear_tf=True)

        ### Experiment setup ###
        pipe = get_experiment_pipeline(EXPERIMENT_N, model_name, model_obj, TEXT_COL, VECTORIZER)
        
        ### Training ###
        logging.info(f"Begin training of {model_name}.")
        pipe.fit(X_train, y_train)

        ### Prediction & Classification report ###
        logging.info("Predicting.")
        y_pred = pipe.predict(X_test)

        clf_report = classification_report(
            y_test,
            y_pred,
            zero_division=0,
            output_dict=True,
        )
        clf_report_df = pd.DataFrame(clf_report).transpose()
        clf_report_df.to_csv(f"{RESULTS_PATH}/clfreport_{OUTPUT_NAME}.csv")

        ### Confusion matrix handling ###
        logging.info("Computing the Confusion Matrix.")
        cm = confusion_matrix(y_test, y_pred)
        cm_df = pd.DataFrame(
            cm, index=np.unique(y_test_labels), columns=np.unique(y_train_labels)
        )
        cm_df.to_csv(f"{RESULTS_PATH}/cm_{OUTPUT_NAME}.csv")

        ### Feature importance ###
        logging.info("Computing the Feature Importance.")
        feature_pipe = dict(pipe.named_steps["features"].transformer_list)
        feature_names = []
        for _, v in feature_pipe.items():
            feature_names.extend(v.get_feature_names())

        if model_name == "xgb_tree":
            features_d = dict(
                zip(
                    feature_names,
                    pipe.named_steps["xgb_tree"].feature_importances_,
                )
            )
            features_df = pd.DataFrame(features_d.items(), columns=["feature", "value"])
            features_df = features_df.sort_values(by="value", ascending=False)
            features_df.to_csv(f"{RESULTS_PATH}/fi_{OUTPUT_NAME}.csv", index=False)
        else:
            coefs_df = pd.DataFrame(
                pipe.named_steps[model_name].coef_,
                index=pipe.named_steps[model_name].classes_,
                columns=feature_names,
            )
            coefs_df.index.name = "class"
            coefs_df.to_csv(f"{RESULTS_PATH}/coefs_{OUTPUT_NAME}.csv.zip")

        # loop time
        logging.info(
            f"Classification duration of {model_name}: {int(float(time.time() - LOOP_TIME))/60} minute(s)."
        )

    ### PROGRAM DURATION ###
    logging.info(f"Run-time: {int(float(time.time() - START_TIME))/60} minute(s).")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
