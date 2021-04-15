import argparse
import json
import time

import numpy as np
import pandas as pd

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
    hours = int(duration//3600)
    duration = duration - 3600*hours
    minutes = int(duration//60)
    seconds = int(duration - 60*minutes)

    def double_zero(element):
        if element == 0:
            return "00"
        else:
            return element

    return f"Total duration: {double_zero(hours)}:{double_zero(minutes)}:{double_zero(seconds)}"

def main():
    # time management
    program_start = time.time()
    
    INDUSTRY_CODES_PATH = args.path + "linkedin-industry-codes.json"
    TRAIN_PATH_JSON = args.path + "train.ndjson"
    TEST_PATH_JSON = args.path + "test.ndjson"
    TRAIN_PATH_CSV = args.path + "train.csv"
    TEST_PATH_CSV = args.path + "test.csv"


    # load into dataframes
    train_records = map(json.loads, open(TRAIN_PATH_JSON))
    train = pd.DataFrame.from_records(train_records)
    
    test_records = map(json.loads, open(TEST_PATH_JSON))
    test = pd.DataFrame.from_records(test_records)


    # add industry codes to train
    with open(INDUSTRY_CODES_PATH) as f:
        industry_codes = json.load(f)
        
    train["industry_name"] = list(map(lambda x: get_code(industry_codes, x), 
                                        dict(train["industry"]).values()))

    
    # remove empty text, html, industry and industry_name cols
    train = train.drop(train[train['text'] == ''].index)
    train = train.drop(train[train['html'] == ''].index)
    train = train.drop(train[train['industry_name'] == ''].index)

    train["country"] = train["country"].replace(np.nan, "UNKNOWN")
    train["country"] = train["country"].replace(" ", "UNKNOWN")
    train["country"] = train["country"].replace("", "UNKNOWN")

    test = test.drop(test[test['text'] == ''].index)
    test = test.drop(test[test['html'] == ''].index)


    # save to csv
    train.to_csv(TRAIN_PATH_CSV, index=False)
    test.to_csv(TEST_PATH_CSV, index=False)

    # output time
    print(get_execution_duration(float(time.time() - program_start)))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog="dataset_pipeline", 
                                    description="Pipeline for creating datasets.")
    parser.add_argument("--path", "-p", type=str, default="../data/", 
                                    help="Path to dataset ndjson files.")
    
    args = parser.parse_args()

    main()