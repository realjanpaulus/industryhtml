import argparse

def main():
    # TODO
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="clf_pipeline", 
                                    description="Pipeline for ML classification.")
    parser.add_argument("--path", "-p", type=str, default="../data/", 
                                    help="Path to dataset csv files.")

    args = parser.parse_args()

    main()