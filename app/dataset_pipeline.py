import argparse


def main():
    

    import glob
    print(glob.glob(f"{args.path}/*.csv"))



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(prog="dataset_pipeline", 
                                    description="Pipeline for creating datasets.")
    parser.add_argument("--path", "-p", type=str, default="../data/", 
                                    help="Path to dataset ndjson files.")
    
    args = parser.parse_args()

    main()