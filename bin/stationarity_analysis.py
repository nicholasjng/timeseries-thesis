import numpy as np 
import pandas as pd 
from pandarallel import pandarallel

import argparse
import os
import yaml
import datetime

from utils.stat_utils import calculate_unittest_pvals
from utils.time_utils import get_timedeltas
from utils.data_utils import *

def calculate_stationarity(config_file, verbose=2):
    """
    Small script for computing various regularity measures for a number of real users.
    config_file: str, config file location for regularity statistic calculation setup
    verbose: pandarallel verbosity level
    """

    with open(config_file, 'r') as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)
    
    #initialize parallel pandas, progress_bar=True makes the program crash for some reason
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

    stat_names = ["kpss_c", "kpss_ct", "adf_c", "adf_ct"]

    #construct result dataframe
    results = pd.DataFrame(np.nan, columns=stat_names, index=customer_base.index)

    #Possible TODO: Make an option to select active range instead of contract duration
    last_log_date = login_dataframe["DATE_SAVED"].max()   

    kwds = {"customer_base": customer_base, "last_log_date": last_log_date, "stat_names": stat_names}

    results = login_dataframe.groupby("CUST_CODE").parallel_apply(calculate_unittest_pvals, **kwds)

    data_ext = ".csv"
    filename = "stationarity_results"
    out_path = output_dir + filename + data_ext
    results.to_csv(out_path, compression="gzip")
    
    print("Calculation finished. Summary statistics of p values:")
    print(results.describe())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate user binary visit pattern regularity using advanced statistics.")
    parser.add_argument("--config_file", default="cfg/stationarity.config.yml", help="yaml config file for statistical calculation")
    parser.add_argument("--verbose", type=int, default=2, help="pandarallel verbosity level (0, 1 or 2)")
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        parser.print_help()
        exit(0)

    calculate_stationarity(args.config_file, args.verbose)