import numpy as np 
import pandas as pd 
import ruptures as rpt
import ruptures.metrics as rptm

from math import isclose

from regularity_analysis.feature_extraction import approximate_entropy, sample_entropy, integer_representation_entropy, permutation_entropy, lz_complexity, fourier_spectrum_entropy, multiscale_entropy, classical_periodogram_entropy, welch_entropy, lombscargle_entropy, shannon_entropy, shannon_markov_entropy
from utils.time_utils import get_timedeltas, get_periodogram, get_welch_periodogram, construct_binary_visit_series
from statsmodels.tsa.stattools import adfuller, kpss
from changepoint_detection.cp_features import * 

stat_names = {"apen": "ApEn", "sampen": "SampEn", "intrepen": "IntRepEn", "permen": "PermEn", "pse": "PowerSpectrumEntropy", "wse": "WelchEntropy", "lse": "LombScargleEntropy", "shannon": "ShannonEntropy", "shannon_markov": "ShannonMarkovEntropy"}

stat_keys = {"apen": ["m", "r"], "sampen": ["m", "r"], "intrepen": ["n", "lag"], "permen": ["n", "lag"], "pse": ["window", "window_size"], "wse": ["window", "nperseg"], "lse": [], "shannon": [], "shannon_markov": []}

stat_funcs = {"apen": approximate_entropy, "sampen": sample_entropy, "intrepen": integer_representation_entropy, "permen": permutation_entropy, "pse": classical_periodogram_entropy, "wse": welch_entropy, "lse": lombscargle_entropy, "shannon": shannon_entropy, "shannon_markov": shannon_markov_entropy}

def get_stat_names(experiment_dict, ex_type):
    statistics = []

    for ex in experiment_dict.values():
        #look up stat name and appended value keys
        stat_type = ex["stat"]
        keys  = stat_keys[stat_type]
        name = stat_names[stat_type]

        #initialize stat name by prepending ex_type (e.g. "Binary", "Timedelta", "Binary_Fourier")
        stat_name = ex_type + "_" + name
    
        if keys:
            for key in keys:
                try:
                    val = ex["params"][key]
                    if isinstance(val, tuple):
                        stat_name += "_{}={}".format(key, str(val[0]))
                    else:
                        stat_name += "_{}={}".format(key, str(val))
                except KeyError:
                    val = ex["params"]["p_config"][key]
                    #check for tuple-valued arguments, here window functions passed to get_window
                    if isinstance(val, tuple):
                        stat_name += "_{}={}".format(key, str(val[0]))
                    else:
                        stat_name += "_{}={}".format(key, str(val))
        
        statistics.append(stat_name)
        
    return statistics

def calculate_binary_stats(data, customer_base, config, last_log_date, stat_names):
    cust_code = data.iloc[0].CUST_CODE

    #arguments to construct_binary_visit_series
    freq = config["freq"]
    mode = config["mode"]

    #experiment configs from config file
    experiments = config["experiments"]

    #total number of calculated statistics
    total_binary_stats = len(stat_names)

    stats = np.full(total_binary_stats, np.nan)

    customer = None 
    try:
        customer = customer_base.loc[cust_code]
    except KeyError:
        print("Encountered an unknown customer")
        return pd.Series(stats, index=stat_names)

    #after this, visit_series contains integers 
    binary_visit_ts = construct_binary_visit_series(customer, data, last_log_date, freq=freq, mode=mode)  

    if isclose(binary_visit_ts.sum(), 0.0):
        print("Found empty time series. This is a data artifact")
        return pd.Series(stats, index=stat_names)

    idx = 0
    for ex in experiments.values():
        stat = ex["stat"]
        stats[idx] = stat_funcs.get(stat)(binary_visit_ts, **ex["params"])
        idx += 1
    
    return pd.Series(stats, index=stat_names)

def calculate_timedelta_stats(data, customer_base, config, last_log_date, stat_names):
    cust_code = data.iloc[0].CUST_CODE

    #experiment configs from config file
    experiments = config["experiments"]

    #total number of calculated statistics
    total_timedelta_stats = len(stat_names)

    stats = np.full(total_timedelta_stats, np.nan)
    timedeltas = None

    customer = None 
    try:
        customer = customer_base.loc[cust_code]
    except KeyError:
        print("Encountered an unknown customer")
        return pd.Series(stats, index=stat_names)       

    #compute FFT of timedeltas
    try:
        timedeltas = get_timedeltas(data.DATE_SAVED, return_floats=True)
    except Exception as e:
        if hasattr(e, 'message'):
            print(e.message)
            return pd.Series(stats, index=stat_names)
        else:
            print(e)
            return pd.Series(stats, index=stat_names)

    idx = 0
    for ex in experiments.values():
        stat = ex["stat"]
        stats[idx] = stat_funcs.get(stat)(timedeltas, **ex["params"])
        idx += 1
    
    return pd.Series(stats, index=stat_names)

def calculate_binary_fourier_stats(data, customer_base, config, last_log_date, stat_names):
    cust_code = data.iloc[0].CUST_CODE

    #arguments to construct_binary_visit_series
    freq = config["freq"]
    mode = config["mode"]

    #experiment configs from config file
    experiments = config["experiments"]

    #total number of calculated statistics
    total_fourier_stats = len(stat_names)

    stats = np.full(len(stat_names), np.nan)
 
    customer = None 
    try:
        customer = customer_base.loc[cust_code]
    except KeyError:
        print("Encountered an unknown customer")
        return pd.Series(stats, index=stat_names)

    #after this, visit_series contains integers 
    binary_visit_ts = construct_binary_visit_series(customer, data, last_log_date, freq=freq, mode=mode)

    if isclose(binary_visit_ts.sum(), 0.0):
        print("Found empty time series. This is a data artifact")
        return pd.Series(stats, index=stat_names)

    #compute FFT of binary
    try:
        binary_periodogram = get_periodogram(binary_visit_ts, **config["periodogram"])
        binary_welch = get_welch_periodogram(binary_visit_ts, **config["welch"])
    except Exception as e:
        if hasattr(e, 'message'):
            print(e.message)
            return pd.Series(stats, index=stat_names)
        else:
            print(e)
            return pd.Series(stats, index=stat_names)      

    idx = 0
    for ex in experiments.values():
        stat = ex["stat"]
        stats[idx] = stat_funcs.get(stat)(binary_periodogram, **ex["params"])
        idx += 1

    #repeat all experiments for the Welch Periodogram
    for ex in experiments.values():
        stat = ex["stat"]
        stats[idx] = stat_funcs.get(stat)(binary_welch, **ex["params"])
        idx += 1
    
    return pd.Series(stats, index=stat_names)

def calculate_td_fourier_stats(data, customer_base, config, last_log_date, stat_names):
    cust_code = data.iloc[0].CUST_CODE

    #experiment configs from config file
    experiments = config["experiments"]

    #total number of calculated statistics, fourier stats for both periodogram types
    total_fourier_stats = len(stat_names)

    stats = np.full(len(stat_names), np.nan)
 
    customer = None 
    try:
        customer = customer_base.loc[cust_code]
    except KeyError:
        print("Encountered an unknown customer")
        return pd.Series(stats, index=stat_names)

    #compute periodogram of timedeltas
    try:
        timedeltas = get_timedeltas(data.DATE_SAVED, return_floats=True)
        td_periodogram = get_periodogram(timedeltas, **config["periodogram"])
        td_welch = get_welch_periodogram(timedeltas, **config["welch"])
    except Exception as e:
        if hasattr(e, 'message'):
            print(e.message)
            return pd.Series(stats, index=stat_names)
        else:
            print(e)
            return pd.Series(stats, index=stat_names)      
    
    idx = 0
    for ex in experiments.values():
        stat = ex["stat"]
        stats[idx] = stat_funcs.get(stat)(td_periodogram, **ex["params"])
        idx += 1

    #repeat all experiments for the Welch Periodogram
    for ex in experiments.values():
        stat = ex["stat"]
        stats[idx] = stat_funcs.get(stat)(td_welch, **ex["params"])
        idx += 1
    
    return pd.Series(stats, index=stat_names)

def calculate_spectral_stats(data, customer_base, config, last_log_date, stat_names):
    cust_code = data.iloc[0].CUST_CODE

    #arguments to construct_binary_visit_series
    freq = config["freq"]
    mode = config["mode"]

    experiments = config["experiments"]
    bin_exs = experiments["bin"]
    td_exs = experiments["td"]

    #total number of calculated statistics
    total_stats = len(stat_names)

    stats = np.full(total_stats, np.nan)

    customer = None 
    try:
        customer = customer_base.loc[cust_code]
    except KeyError:
        print("Encountered an unknown customer")
        return pd.Series(stats, index=stat_names)  

    #after this, visit_series contains integers 
    binary_visit_ts = construct_binary_visit_series(customer, data, last_log_date, freq=freq, mode=mode)

    if isclose(binary_visit_ts.sum(), 0.0):
        print("Found empty time series. This is a data artifact")
        return pd.Series(stats, index=stat_names)

    idx = 0
    for ex in bin_exs.values():
        stat = ex["stat"]
        stats[idx] = stat_funcs.get(stat)(binary_visit_ts, **ex["params"])
        idx += 1

    #compute FFT of binary
    try:
        timedeltas = get_timedeltas(data.DATE_SAVED, return_floats=True)
    except Exception as e:
        if hasattr(e, 'message'):
            print(e.message)
            return pd.Series(stats, index=stat_names)
        else:
            print(e)
            return pd.Series(stats, index=stat_names) 

    for ex in td_exs.values():
        stat = ex["stat"]
        stats[idx] = stat_funcs.get(stat)(timedeltas, **ex["params"])
        idx += 1

    return pd.Series(stats, index=stat_names)

def calculate_unittest_pvals(data, customer_base, last_log_date, stat_names):
    data.sort_values(by="DATE_SAVED", inplace=True)

    cust_code = data.iloc[0].CUST_CODE
    
    #total number of calculated statistics
    total_stats = len(stat_names)

    stats = np.full(total_stats, np.nan)

    customer = None 
    try:
        customer = customer_base.loc[cust_code]
    except KeyError:
        print("Encountered an unknown customer")
        return pd.Series(stats, index=stat_names)  

    #after this, visit_series contains integers 
    binary_visit_ts = construct_binary_visit_series(customer, data, last_log_date)

    if isclose(binary_visit_ts.sum(), 0.0):
        print("Found empty time series. This is a data artifact")
        return pd.Series(stats, index=stat_names)

    try:
        #save p-value of kpss test for later significance level setting
        stats[0] = kpss(binary_visit_ts, regression="c", nlags="auto")[1]
        stats[1] = kpss(binary_visit_ts, regression="ct", nlags="auto")[1]

        #save p-value of ADF test 
        stats[2] = adfuller(binary_visit_ts, regression="c")[1]
        stats[3] = adfuller(binary_visit_ts, regression="ct")[1]
    except:
        return pd.Series(stats, index=stat_names)

    return pd.Series(stats, index=stat_names)

def calculate_multiscale_entropy(data, customer_base, config, last_log_date, stat_names):
    cust_code = data.iloc[0].CUST_CODE

    #arguments to construct_binary_visit_series
    freq = config["freq"]
    mode = config["mode"]

    max_scale = config["max_scale"]
    mse_freq = config["mse_freq"]

    entropy_names = ["ApEn", "SampEn", "PermEn"]

    scales = ["3H", "6H", "12H"] + list(str(i) + "D" for i in range(1, max_scale + 1))

    #fourier_scales = [(str(i) + "H" for i in range(1, 25))] + ["2D", "3D", "4D"]

    #total number of calculated statistics
    total_stats = len(stat_names)

    stats = np.full(total_stats, np.nan)

    customer = None 
    try:
        customer = customer_base.loc[cust_code]
    except KeyError:
        print("Encountered an unknown customer")
        return pd.Series(stats, index=stat_names)  

    #after this, visit_series contains integers 
    binary_visit_ts = construct_binary_visit_series(customer, data, last_log_date, freq=freq, mode=mode)

    if isclose(binary_visit_ts.sum(), 0.0):
        print("Found empty time series. This is a data artifact")
        return pd.Series(stats, index=stat_names)

    idx = 0 

    for name in entropy_names:
        mse = multiscale_entropy(binary_visit_ts, stat_name=name, scales=scales, **config[name.lower()])

        #sum as approximation to multiscale entropy integral
        stats[idx] = mse.sum()

        #last scale that produced a stat value and not nan
        stats[idx + 1] = np.argmax(np.isnan(mse.values)) + 1 if mse.isna().any() else len(mse.index)

        #print(stats[idx])
        #print(stats[idx + 1])

        idx += 2

    return pd.Series(stats, index=stat_names)

def aic_penalty(variance, length):
    """
    Gives the AIC penalty for constrained optimization change point detection.
    """
    return variance

def bic_penalty(variance, length):
    """
    Gives the BIC penalty for constrained optimization change point detection.
    """
    return variance * np.log(length)

def hqc_penalty(variance, length):
    """
    Gives the HQC penalty for constrained optimization change point detection.
    """
    return 2 * variance * np.log(np.log(length))

penalty_funcs = {"aic": aic_penalty, "bic": bic_penalty, "hqc": hqc_penalty}

def calculate_changepoints(data, customer_base, config, last_log_date, stat_names):
    global penalty_funcs
    cust_code = data.iloc[0].CUST_CODE

    #arguments to construct_binary_visit_series
    freq = config["freq"]
    mode = config["mode"]

    #total number of calculated statistics
    total_stats = len(stat_names)

    stats = np.full(total_stats, np.nan)

    customer = None 
    try:
        customer = customer_base.loc[cust_code]
    except KeyError:
        print("Encountered an unknown customer")
        return pd.Series(stats, index=stat_names)  

    #after this, visit_series contains integers 
    binary_visit_ts = construct_binary_visit_series(customer, data, last_log_date, freq=freq, mode=mode) 

    #empty TS are discarded, this happens on a few datapoints even though there are visits in the data
    if isclose(binary_visit_ts.sum(), 0.0):
        print("Found empty time series. This is a data artifact")
        return pd.Series(stats, index=stat_names)
         
    variance = binary_visit_ts.var()
    num_obs = len(binary_visit_ts.index)

    # change point detection
    model = config["model"]  #one of "l1", "l2", "rbf", "linear", "normal", "ar"
    ics = config["penalties"]

    num_ics = len(ics)
    pelt_instance = rpt.Pelt(model=model).fit(binary_visit_ts.values)
    binseg_instance = rpt.Binseg(model=model).fit(binary_visit_ts.values)

    idx = 0
    for ic in ics:
        pen = penalty_funcs.get(ic)(variance, num_obs)
        pelt_cps = pelt_instance.predict(pen=pen)
        binseg_cps = binseg_instance.predict(pen=pen)

        #number of changepoints found with PELT and BinSeg wrt criterion ic
        stats[idx] = len(pelt_cps) + 1
        stats[idx + 1] = len(binseg_cps) + 1
        try:
            stats[idx + 2] = rptm.hausdorff(pelt_cps, binseg_cps)
        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)
        try:
            stats[idx + 3] = rptm.hamming(pelt_cps, binseg_cps)
        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)    

        idx += 4
    
    print("Finished analysis for customer " + cust_code + ".")
    return pd.Series(stats, index=stat_names)

def calculate_cp_stats(data, customer_base, config, last_log_date, stat_names):
    global penalty_funcs
    cust_code = data.iloc[0].CUST_CODE

    #arguments to construct_binary_visit_series
    freq = config["freq"]
    mode = config["mode"]

    #total number of calculated statistics
    total_stats = len(stat_names)

    stats = np.full(total_stats, np.nan)

    customer = None 
    try:
        customer = customer_base.loc[cust_code]
    except KeyError:
        print("Encountered an unknown customer")
        return pd.Series(stats, index=stat_names)  

    #after this, visit_series contains integers 
    binary_visit_ts = construct_binary_visit_series(customer, data, last_log_date, freq=freq, mode=mode) 

    #empty TS are discarded, this happens on a few datapoints even though there are visits in the data
    if isclose(binary_visit_ts.sum(), 0.0):
        print("Found empty time series. This is a data artifact")
        return pd.Series(stats, index=stat_names)
         
    variance = binary_visit_ts.var()
    num_obs = len(binary_visit_ts.index)
    
    ics = config["penalties"]

    pelt_instance = rpt.Pelt(**config["model_config"]).fit(binary_visit_ts.values)

    idx = 0
    for ic in ics:
        pen = penalty_funcs.get(ic)(variance, num_obs)
        pelt_cps = np.array([1] + pelt_instance.predict(pen=pen)) - 1
        cp_dates = binary_visit_ts.index[pelt_cps]

        num_cps = len(pelt_cps)

        #number of changepoints found with PELT and BinSeg wrt criterion ic
        stats[idx] = num_cps
        idx += 1
        stats[idx] = startofyear_cps(cp_dates)
        idx += 1
        stats[idx:idx+12] = get_monthly_cps(cp_dates)
        idx += 12
        stats[idx:idx+4] = changepoint_duration_moments(cp_dates)
        idx += 4
    
    print("Finished analysis for customer " + cust_code + ".")
    return pd.Series(stats, index=stat_names)