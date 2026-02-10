# industryhtml

> :exclamation: The code and experiments in this repository were developed in the middle of 2021as part of an university project and may therefore be outdated.

Automatic classification of industries by a companies start HTML page.



## Getting started

### `industryhtml/clf_pipeline`

```sh
python clf_pipeline.py -h

usage: clf_pipeline [-h] [--path PATH] [--data_clean]
                    [--data_shortened DATA_SHORTENED]
                    [--experiment EXPERIMENT] [--max_features MAX_FEATURES]
                    [--n_jobs N_JOBS] [--pos_tagging]
                    [--specific_country SPECIFIC_COUNTRY]
                    [--text_col TEXT_COL] [--testing]

Pipeline for ML classification.

optional arguments:
  -h, --help            show this help message and exit
  --path PATH, -p PATH  Path to dataset csv files.
  --data_clean, -dc     Indicates if datatset with HTML, XHTML or XML
                        boilerplate removal should be loaded.
  --data_shortened DATA_SHORTENED, -ds DATA_SHORTENED
                        Indicates if dataset with specific number of rows
                        should be loaded (default: None).
  --experiment EXPERIMENT, -e EXPERIMENT
                        Indicates number of experiment.
  --max_features MAX_FEATURES, -mf MAX_FEATURES
                        Indicates maximum of features for the vectorizer.
  --n_jobs N_JOBS, -nj N_JOBS
                        Indicates the number of processors used for
                        computation (default: 1).
  --pos_tagging, -pt    Extracts nouns from text.
  --specific_country SPECIFIC_COUNTRY, -sc SPECIFIC_COUNTRY
                        Load dataset with only given ISO2 country code
                        (default: '' = all).
  --text_col TEXT_COL, -tc TEXT_COL
                        Indicating the column with text (default: 'text').
  --testing, -t         Starts testing mode with a small subset of the
                        corpus and no tunable parameters.
```

### `industryhtml/dataset_pipeline`

```sh
python dataset_pipeline.py -h

usage: dataset_pipeline [-h] [--path PATH] [--clean_boilerplate]
                        [--ignore_country] [--keep_html_col]
                        [--max_rows MAX_ROWS] [--pos_tagging]
                        [--specific_country SPECIFIC_COUNTRY]
                        [--test_size TEST_SIZE] [--use_test_txt]

Pipeline for creating datasets.

optional arguments:
  -h, --help            show this help message and exit
  --path PATH, -p PATH  Path to dataset ndjson file.
  --clean_boilerplate, -cb
                        Indicates if HTML, XHTML or XML boilerplate removal
                        should be applied.
  --ignore_country, -ic
                        Ignore addition of ISO2 country code to shorten the
                        runtime.
  --keep_html_col, -khc
                        Keep the original html column.
  --max_rows MAX_ROWS, -mr MAX_ROWS
                        Sets maximum number of rows (default: None).
  --pos_tagging, -pt    Extracts nouns from text.
  --specific_country SPECIFIC_COUNTRY, -sc SPECIFIC_COUNTRY
                        Limit dataset to given country, specified by ISO2
                        country code.
  --test_size TEST_SIZE, -ts TEST_SIZE
                        Set test size.
  --use_test_txt, -utt  Indicates if the train test split should be
                        performed on the basis of an existing txt file.
```


## About

### Dataset

The industries dataset has to be in a folder `data` with the name: `data.ndjson`. It can be transformed to a train/test csv with `dataset_pipline.py` in the `app` directory.

### Project structure

#### `industryhtml`

- `clf_pipeline.py`: for the classification experiments
- `dataset_pipeline.py`: for the preparation of the train and test dataset
- `experiments.json`: contains the informations about the experiments. Important for `clf_pipeline.py`
- `utils.py`: contains helper functions

#### `data`

- should contain: `data.ndjson` (see section **Dataset**)
- `cptesturls.txt`: URLs for a deterministic train/test split for the POS experiments
- `ctesturls.txt`: URLs for a deterministic train/test split for the unchanged text content experiments
- `industries.csv`: contains the industries labels from LinkedIn

#### `notebooks`

- `dataset_insights.ipynb`: for the creation of dataset insight plot
- `evaluation.ipynb`: for the creation of evaluation tables and plots
