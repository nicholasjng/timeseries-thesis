import numpy as np 
import pandas as pd 
from pandarallel import pandarallel

import argparse
import os
import yaml
import datetime
import time

from utils.stat_utils import calculate_timedelta_stats
from utils.data_utils import load_login_data, load_customer_data
from utils.time_utils import get_timedeltas

def calculate_user_td_regularity(config_file, verbose=2):
    """
    Small script for computing various regularity measures on the timedelta distributions for a number of real users.
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
        
    #configurations for regularity measures
    apen_config = config["apen"]
    apen_m = apen_config["m"]
    apen_r = apen_config["r"]
    apen_detrend = apen_config["detrend"]

    sampen_config = config["sampen"]
    sampen_m = sampen_config["m"]
    sampen_r = sampen_config["r"]
    sampen_detrend = sampen_config["detrend"]

    permen_config = config["permen"]
    permen_n = permen_config["n"]
    permen_lag = permen_config["lag"]

    #total number of calculated statistics
    total_fourier_stats = len(apen_m) * len(apen_r) + len(sampen_m) * len(sampen_r) + len(permen_n) * len(permen_lag)

    login_dataframe = load_login_data(logindata_filepath=logindata_filepath)
    customer_base = load_customer_data(userdata_filepath=userdata_filepath)

    login_dataframe.sort_values(by=["CUST_CODE", "DATE_SAVED"], inplace=True)

    timedelta_apens = ["Timedelta_ApEn_m={}_r={}".format(i, j) for i in apen_m for j in apen_r]
    timedelta_sampens = ["Timedelta_SampEn_m={}_r={}".format(i, j) for i in sampen_m for j in sampen_r]
    timedelta_permens = ["Timedelta_PermEn_n={}_lag={}".format(i, j) for i in permen_n for j in permen_lag]

    stat_names = timedelta_apens + timedelta_sampens + timedelta_permens 

    #construct result dataframe
    results = pd.DataFrame(np.nan, columns=stat_names, index=customer_base.index)

    #Possible TODO: Make an option to select active range instead of contract duration
    last_log_date = login_dataframe["DATE_SAVED"].max()   

    kwds = {"customer_base": customer_base, "config": config, "last_log_date": last_log_date, "stat_names": stat_names}

    start = time.time()
    print("Starting calculation.")
    results = login_dataframe.groupby("CUST_CODE").parallel_apply(calculate_timedelta_stats, **kwds)
    end = time.time()
    elapsed = int(end - start)
    print("Calculation finished. Elapsed time: {} hrs {} mins {} secs".format(elapsed // 3600, (elapsed // 60) % 60, elapsed % 60))

    data_ext = ".csv"
    
    filename = output_dir + "td_regularity_results" + data_ext
    results.to_csv(filename, compression="gzip")
    
    print("Summary statistics of regularity measures:")
    print(results.describe())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate user pattern regularity using advanced statistics.")
    parser.add_argument("--config_file", default="cfg/timedelta_regularity.config.yml", help="yaml config file for statistical calculation")
    parser.add_argument("--verbose", type=int, default=2, help="pandarallel verbosity level (0, 1 or 2)")
    args = parser.parse_args()

    if not os.path.exists(args.config_file):
        parser.print_help()
        exit(0)

    calculate_user_td_regularity(args.config_file, args.verbose)