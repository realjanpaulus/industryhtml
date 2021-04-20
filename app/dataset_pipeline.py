import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import clean_html_boilerplate

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

    INDUSTRY_CODES_PATH = args.path + "industries"
    DATA_PATH_JSON = args.path + "data.ndjson"
    TRAIN_PATH_CSV = args.path + "train"
    TEST_PATH_CSV = args.path + "test"
    TEST_SIZE = args.test_size
    
    print("Loading ndjson.")
    data_records = map(json.loads, open(DATA_PATH_JSON))
    data = pd.DataFrame.from_records(data_records)

    ### add country information ###
    if not args.ignore_country:
        print("Extract and appending country information.")
        from langdetect import detect

        tmp_data = data.copy()
        tmp_data["country"] = tmp_data.apply(lambda row: detect(row.text).upper(), axis=1)
        data = tmp_data.copy()

    codes = pd.read_csv(INDUSTRY_CODES_PATH + ".csv")
    data["group_representative_label"] = data.apply(lambda row: codes.iloc[row.group_representative].industry_label, axis=1)

    ### reduce dataset to specific country ###
    if args.specific_country != "":
        if args.specific_country in data["country"].unique():
            print(f"NOTE: Only instances of the country {args.specific_country} will be kept.")
            data = data[data["country"] == args.specific_country]

            TRAIN_PATH_CSV = TRAIN_PATH_CSV + "_"
            TEST_PATH_CSV = TEST_PATH_CSV + "_"
        else:
            print(f"WARNING: Country code {args.specific_country} is not in the dataset, \
                all countries will be kept!")

    ### remove boilerplate html code ###
    if args.clean_html:
        print("Cleaning HTML boilerplate...")

        # TODO: erst in notebook testen
        # ValueError: Unicode strings with encoding declaration are not supported. 
        # # Please use bytes input or XML fragments without declaration.
        data["html"] = data.apply(lambda row: clean_html_boilerplate(row["html"]), axis=1)


    print("Splitting data.")
    train, test = train_test_split(data, 
                                    test_size=TEST_SIZE, 
                                    stratify=data["group_representative"], 
                                    random_state=42)
            
        
    # save to csv
    print("Saving data.")
    train.to_csv(TRAIN_PATH_CSV + f"{args.specific_country}.csv", index=False)
    test.to_csv(TEST_PATH_CSV + f"{args.specific_country}.csv", index=False)

    # output time
    print(get_execution_duration(float(time.time() - program_start)))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog="dataset_pipeline", 
                                    description="Pipeline for creating datasets.")
    parser.add_argument("--path", "-p", type=str, default="../data/", 
                        help="Path to dataset ndjson file.")
    parser.add_argument("--clean_html", "-ch", action="store_true", 
                        help="Indicates if HTML boilerplate removal should be applied.")
    parser.add_argument("--ignore_country", "-ic", action="store_true", 
                        help="Ignore addition of ISO2 country code to shorten the runtime.")
    parser.add_argument("--specific_country", "-sc", type=str, default="", 
                        help="Limit dataset to given country, specified by ISO2 country code.")
    parser.add_argument("--test_size", "-ts", type=float, default=0.2, help="Set test size.")
    args = parser.parse_args()

    main()
