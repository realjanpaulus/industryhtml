# industryhtml
Classify industries by a companies start HTML page.

## dataset

The industries dataset has to be in a folder `data` with the name: `data.ndjson`. It can be transformed to a train/test csv with `dataset_pipline.py` in the `app` directory.

## project structure

- **app**
    - `clf_pipeline.py`: for the classification experiments
    - `dataset_pipeline.py`: for the preparation of the train and test dataset
    - `experiments.json`: contains the informations about the experiments. Important for `clf_pipeline.py`
    - `utils.py`: contains helper functions
- **data**
    - should contain: `data.ndjson` (see section **dataset**)
    - `cptesturls.txt`: URLs for a deterministic train/test split for the POS experiments
    - `ctesturls.txt`: URLs for a deterministic train/test split for the unchanged text content experiments
    - `industries.csv`: contains the industries labels from LinkedIn
- **notebooks**
    - `dataset_insights.ipynb`: for the creation of dataset insight plot
    - `evaluation.ipynb`: for the creation of evaluation tables and plots
- **results**. contains the results of the different experiments

