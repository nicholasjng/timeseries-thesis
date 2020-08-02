import numpy as np 
import pandas as pd 
from pandarallel import pandarallel

import argparse
import os
import yaml
import datetime
import time

from utils.stat_utils import calculate_binary_fourier_stats, get_stat_names
from utils.time_utils import get_timedeltas
from utils.data_utils import *

def calculate_binary_fourier_regularity(config_file, verbose=2):
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
    
    mode = config["mode"]
    freq = config["freq"]

    #configurations for regularity measures
    experiments = config["experiments"]

    #total number of calculated fourier statistics
    total_fourier_stats = len(experiments) * 2

    login_dataframe = load_login_data(logindata_filepath=logindata_filepath)
    customer_base = load_customer_data(userdata_filepath=userdata_filepath)

    login_dataframe.sort_values(by=["CUST_CODE", "DATE_SAVED"], inplace=True)

    fourier_stat_names = get_stat_names(experiments, ex_type="Fourier") + get_stat_names(experiments, ex_type="Welch")

    #construct result dataframe
    results = pd.DataFrame(np.nan, columns=fourier_stat_names, index=customer_base.index)

    #Possible TODO: Make an option to select active range instead of contract duration
    last_log_date = login_dataframe["DATE_SAVED"].max()   

    kwds = {"customer_base": customer_base, "config": config, "last_log_date": last_log_date, "stat_names": fourier_stat_names}

    start = time.time()
    print("Starting calculation.")
    results = login_dataframe.groupby("CUST_CODE").parallel_apply(calculate_binary_fourier_stats, **kwds)
    end = time.time()
    elapsed = int(end - start)
    print("Calculation finished. Elapsed time: {} hrs {} mins {} secs".format(elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))
    
    data_ext = ".csv"
    
    filename = output_dir + "binary_fourier_results" +  "_" + freq + "_" + mode + data_ext
    results.to_csv(filename, compression="gzip")
    
    print("Summary statistics of binary regularity measures:")
    print(results.describe())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate user binary visit pattern regularity using advanced statistics.")
    parser.add_argument("--config_file", default="cfg/binary_fourier.config.yml", help="yaml config file for statistical calculation")
    parser.add_argument("--verbose", type=int, default=2, help="pandarallel verbosity level (0, 1 or 2)")
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        parser.print_help()
        exit(0)

    calculate_binary_fourier_regularity(args.config_file, args.verbose)