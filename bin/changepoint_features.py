import numpy as np 
import pandas as pd 
from pandarallel import pandarallel

import argparse
import os
import yaml
import datetime
import time

import ruptures as rpt 

from utils.stat_utils import calculate_cp_stats
from utils.time_utils import get_timedeltas
from utils.data_utils import *

def calculate_changepoint_features(config_file, verbose=2):
    """
    Small script for computing changepoint detection features for a number of real users.
    config_file: str, config file location for regularity statistic calculation setup
    verbose: pandarallel verbosity level
    """

    with open(config_file, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    #initialize parallel pandas
    pandarallel.initialize(nb_workers=4, progress_bar=False, verbose=verbose)

    #file paths for obtaining data and user base
    logindata_filepath = config["logindata_filepath"]
    userdata_filepath = config["userdata_filepath"]
    data_output_dir = config["data_output_dir"]

    output_dir = data_output_dir + datetime.datetime.now().strftime('%Y-%m-%d') + "/"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    login_dataframe = load_login_data(logindata_filepath=logindata_filepath)
    customer_base = load_customer_data(userdata_filepath=userdata_filepath)

    model = config["model_config"]["model"]
    ics = config["penalties"]
    cp_models = ["PELT_" + ic for ic in ics]
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    monthly_cp_names = ["cps_" + month for month in months]
    cp_delta_stats = ["cp_dt_mean", "cp_dt_median", "cp_dt_min", "cp_dt_max"]
    
    stat_types = cp_models + ["start_of_year_cps"] + monthly_cp_names + cp_delta_stats

    print(stat_types)

    #construct result dataframe
    results = pd.DataFrame(np.nan, columns=stat_types, index=customer_base.index)

    #Possible TODO: Make an option to select active range instead of contract duration
    last_log_date = login_dataframe["DATE_SAVED"].max()   

    kwds = {"customer_base": customer_base, "config": config, "last_log_date": last_log_date, "stat_names": stat_types}

    start = time.time()
    print("Starting calculation.")
    results = login_dataframe.groupby(by="CUST_CODE").parallel_apply(calculate_cp_stats, **kwds)
    end = time.time()
    elapsed = int(end - start)
    print("Calculation finished. Elapsed time: {} hrs {} mins {} secs".format(elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))
    data_ext = ".csv"
    
    filename = output_dir + "changepoint_features" + "_" + model + data_ext
    results.to_csv(filename, compression="gzip")
    
    print("Summary of changepoint detection:")
    print(results.describe())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate user binary visit pattern regularity using advanced statistics.")
    parser.add_argument("--config_file", default="cfg/cp_features.config.yml", help="yaml config file for statistical calculation")
    parser.add_argument("--verbose", type=int, default=2, help="pandarallel verbosity level (0, 1 or 2)")
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        parser.print_help()
        exit(0)

    calculate_changepoint_features(args.config_file, args.verbose)