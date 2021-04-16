import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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

    def zero_digit(element):
        if element < 10:
            if element == 0:
                return "00"
            else:
                return f"0{element}"
        else:
            return element

    return f"Total duration: {zero_digit(hours)}:{zero_digit(minutes)}:{zero_digit(seconds)}"

def main():
    # time management
    program_start = time.time()

    INDUSTRY_CODES_PATH = args.path + "industries.csv"
    DATA_PATH_JSON = args.path + "data.ndjson"
    TRAIN_PATH_CSV = args.path + "train.csv"
    TEST_PATH_CSV = args.path + "test.csv"
    TEST_SIZE = args.test_size
    
    print("Loading ndjson.")
    data_records = map(json.loads, open(DATA_PATH_JSON))
    data = pd.DataFrame.from_records(data_records)

    if not args.ignore_country:
        print("Extract and appending country information.")
        from langdetect import detect

        tmp_data = data.copy()
        tmp_data["country"] = tmp_data.apply(lambda row: detect(row.text).upper(), axis=1)
        data = tmp_data.copy()

    codes = pd.read_csv(INDUSTRY_CODES_PATH)
    data["group_representative_label"] = data.apply(lambda row: codes.iloc[row.group_representative].industry_label, axis=1)

    print("Splitting data.")
    train, test = train_test_split(data, test_size=TEST_SIZE, stratify=data.industry)
            
        
    # save to csv
    print("Saving data.")
    train.to_csv(TRAIN_PATH_CSV, index=False)
    test.to_csv(TEST_PATH_CSV, index=False)

    # output time
    print(get_execution_duration(float(time.time() - program_start)))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog="dataset_pipeline", 
                                    description="Pipeline for creating datasets.")
    parser.add_argument("--path", "-p", type=str, default="../data/", 
                        help="Path to dataset ndjson file.")
    parser.add_argument("--ignore_country", "-ic", action="store_true", 
                        help="Ignore addition of ISO2 country code to shorten the runtime.")
    parser.add_argument("--test_size", "-ts", type=float, default=0.2, help="Set test size.")
    args = parser.parse_args()

    main()