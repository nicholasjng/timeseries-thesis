import pandas as pd
import numpy as np

from math import isclose

import scipy.special as sc
from numpy import linalg as LA

from regularity_analysis.regularity_measures import *

import os
import time

from utils.time_utils import construct_binary_visit_series, get_periodogram, get_welch_periodogram, get_lombscargle_periodogram

def get_time_of_day_visits(customer_data, login_data_sorted, verbose=False):
    """
    Subroutine to calculate the percentage of visits during a given time of day
    Assumption 1: Values of the hour parameter in the timestamp correspond to
    0-6: Night
    6-12: Morning
    12-18: Afternoon
    18-24: Evening

    Assumption 2: IMPORTANT: The input data frame is assumed to be sorted to not have to deal with complex lookup by 
    matching of customer codes in the customer data.
    """
    
    feature_list = ["night_visits", "morning_visits", "afternoon_visits", "evening_visits"]

    #TODO: Include the option of returning a new dataframe or returning the data inplace
    for feature in feature_list:
        customer_data[feature] = 0.0

    count = 0
    index = 0
    
    visit_numbers = login_data_sorted["CUST_CODE"].value_counts().astype(int)

    num_logins = len(login_data_sorted.index)
    num_users = len(visit_numbers.index)

    while index < num_logins:
        cust_code  = login_data_sorted.iloc[index].CUST_CODE
        customer_visits = visit_numbers[cust_code]
        customer = None

        try:
            customer = customer_data[cust_code]
        except KeyError:
            count += 1
            if verbose and (count % 100 == 0 or count == num_users):
                print("Checked {} customers out of {}".format(count, num_users))
            continue
            
        #select logins with the specified customer code
        customer_logins = login_data_sorted.iloc[index:index+customer_visits]

        customer_data.loc[customer, feature_list] = time_of_day_visits(customer_logins)

        count += 1 
        if verbose and (count % 100 == 0 or count == num_users):
            print("Checked {} customers out of {}".format(count, num_users))
        
        index += customer_visits

def get_quarterly_visits(customer_data, login_data_sorted, verbose=False):
    """
    Subroutine to calculate the percentage of visits during a given business quarter
    Assumption 1: Values of the hour parameter in the timestamp correspond to
    0: Visit in between January 1 and March 31
    1: Visit in between April 1 and June 30
    2: Visit in between July 1 and September 30
    3: Visit in between October 1 and December 31

    Assumption 2: IMPORTANT: The input data frame is assumed to be sorted to not have to deal with complex lookup by 
    matching of customer codes in the customer data.
    """
    
    feature_list = ["1st_quarter_visits", "2nd_quarter_visits", "3rd_quarter_visits", "4th_quarter_visits"]

    #TODO: Include the option of returning a new dataframe or returning the data inplace
    for feature in feature_list:
        customer_data[feature] = 0.0

    count = 0
    index = 0

    visit_numbers = login_data_sorted["CUST_CODE"].value_counts().astype(int)

    num_logins = len(login_data_sorted.index)
    num_users = len(visit_numbers.index)
    
    while index < num_logins:
        cust_code  = login_data_sorted.iloc[index].CUST_CODE
        customer_visits = visit_numbers[cust_code]
        customer = None

        try:
            customer = customer_data[cust_code]
        except KeyError:
            count += 1 
            if verbose and (count % 100 == 0 or count == num_users):
                print("Checked {} customers out of {}".format(count, num_users))
            continue
            
        #select logins with the specified customer code
        customer_logins = login_data_sorted.iloc[index:index+customer_visits]

        customer_data.loc[customer, feature_list] = quarterly_visits(customer_logins)

        count += 1 
        if verbose and (count % 100 == 0 or count == num_users):
            print("Checked {} customers out of {}".format(count, num_users))
        
        index += customer_visits

def get_seasonal_visits(customer_data, login_data_sorted, verbose=False):
    """
    Subroutine to calculate the percentage of visits during a given sea
    Assumption 1: Values of the hour parameter in the timestamp correspond to
    0: Visit in (meteorological) winter, between December 1 and February 28/29.
    1: Visit in (meteorological) spring, between March 1 and May 31
    2: Visit in (meteorological) summer, between June 1 and August 31
    3: Visit in (meteorological) autumn, between September 1 and November 30

    Assumption 2: IMPORTANT: The input data frame is assumed to be sorted to not have to deal with complex lookup by 
    matching of customer codes in the customer data.
    """
    
    feature_list = ["winter_visits", "spring_visits", "summer_visits", "autumn_visits"]

    count = 0
    index = 0

    visit_numbers = login_data_sorted["CUST_CODE"].value_counts().astype(int)

    num_logins = len(login_data_sorted.index)
    num_users = len(visit_numbers.index)

    #TODO: Include the option of returning a new dataframe or returning the data inplace
    for feature in feature_list:
        customer_data[feature] = 0.0
    
    while index < num_logins:
        cust_code  = login_data_sorted.iloc[index].CUST_CODE
        customer_visits = visit_numbers[cust_code]
        customer = None

        try:
            customer = customer_data[cust_code]
        except KeyError:
            count += 1 
            if verbose and (count % 100 == 0 or count == num_users):
                print("Checked {} customers out of {}".format(count, num_users))
            continue
            
        #select logins with the specified customer code
        customer_logins = login_data_sorted.iloc[index:index+customer_visits]

        customer_data.loc[customer, feature_list] = seasonal_visits(customer_logins)

        count += 1 
        if verbose and (count % 100 == 0 or count == num_users):
            print("Checked {} customers out of {}".format(count, num_users))
        
        index += customer_visits

def get_visit_entropies(customer_data, login_data_sorted, m=2, r=0.2, int_order=4, verbose=False):
    """
    Subroutine to calculate the approximate entropy for a visit series. 
    customer_data: Copy of the customer registration database. Assumed to have a visit_numbers column giving the number of visits in the data
    login_data_sorted: Data frame of the login data, assumed sorted by customers
    m: Window length, must be a positive integer
    r: Noise level, default set at 0.2 * series standard_deviation
    int_order: Order of integer for integer representation entropy, see implementation below
    """

    feature_list = ["apen_visits_m={}_r={}".format(m, r), "sampen_visits_m={}_r={}".format(m, r), "intrepen_n={}".format(int_order)]

    #TODO: Include the option of returning a new dataframe or returning the data inplace
    for feature in feature_list:
        customer_data[feature] = 0.0

    #customer counter, can be printed in verbose mode
    count = 0
    index = 0

    visit_numbers = login_data_sorted["CUST_CODE"].value_counts().astype(int)

    num_logins = len(login_data_sorted.index)
    num_users = len(visit_numbers.index)
    
    while index < num_logins:
        cust_code  = login_data_sorted.iloc[index].CUST_CODE
        customer_visits = visit_numbers[cust_code]
        customer = None
        
        try:
            customer = customer_data[cust_code]
        except KeyError:
            count += 1 
            if verbose and (count % 100 == 0 or count == num_users):
                print("Checked {} customers out of {}".format(count, num_users))
            continue
    
        #select logins with the specified customer code
        customer_logins = login_data_sorted.iloc[index:index+customer_visits]   

        #Possible TODO: Make an option to select active range instead of contract duration
        last_log_date = login_data_sorted["DATE_SAVED"].max()         
        
        #after this, visit_series contains integers 
        binary_visit_ts = construct_binary_visit_series(customer, customer_logins, last_log_date)

        #calculate entropies of the series
        visit_apen = approximate_entropy(binary_visit_ts, m, r)
        visit_sampen = sample_entropy(binary_visit_ts, m, r)
        visit_intrepen = integer_representation_entropy(binary_visit_ts, int_order)

        feature_vec = np.array([visit_apen, visit_sampen, visit_intrepen])

        count += 1 
        if verbose and (count % 100 == 0 or count == num_users):
            print("Checked {} customers out of {}".format(count, num_users))

        customer_data.loc[cust_code, feature_list] = feature_vec
        
        index += customer_visits

def time_of_day_visits(customer_logins):
    """
    Small subroutine to return time of day visits from an amount of customer login data.
    """
    visits = np.zeros(4)

    customer_visits = len(customer_logins.index)

    #6 is the size of a part of day, integer division gives a number between 0 and 3
    visit_daytimes = np.array([visit_date.hour for visit_date in customer_logins.DATE_SAVED]) // 6

    visits[visit_daytimes] += 1

    return visits / customer_visits

def seasonal_visits(customer_logins):
    """
    Small subroutine to return seasonal visits from an amount of customer login data.
    """
    visits = np.zeros(4)

    customer_visits = len(customer_logins.index)

    #6 is the size of a part of day, integer division gives a number between 0 and 3
    visit_seasonals = (np.array([visit_date.month for visit_date in customer_logins.DATE_SAVED]) % 12) // 3

    visits[visit_seasonals] += 1

    return visits / customer_visits

def quarterly_visits(customer_logins):
    """
    Small subroutine to return quarterly visits from an amount of customer login data.
    """
    visits = np.zeros(4)

    customer_visits = len(customer_logins.index)
        
    #6 is the size of a part of day, integer division gives a number between 0 and 3
    visit_quarterlies = (np.array([visit_date.month for visit_date in customer_logins.DATE_SAVED]) - 1) // 3

    visits[visit_quarterlies] += 1

    return visits / customer_visits

def approximate_entropy(timeseries, m=2, r=0.2, detrend=True, sigma_units=True):
    """
    Wrapper for the cythonized version to calculate the approximate entropy,
    as first introduced by Pincus et al, A Regularity Statistic for Medical Data Analysis, 1991

    timeseries: pd.Series of timeseries values, has to be convertible to float
    m: Window size
    r: Noise level
    detrend: Whether or not to normalize time series to mean zero and unit variance.
    """
    N = len(timeseries.index)

    if timeseries.isna().any() or m + 1 > N:
        return np.nan

    #standard deviation is saved separately because it is used to calculate the noise level parameter for ApEn
    sigma = timeseries.std()

    noise_level = r 

    if sigma_units:
        noise_level *= sigma

    #if the values are all the same, ApEn is zero 
    if isclose(sigma, 0.0):
        return 0.0

    if detrend:
        #normalization to mean 0, variance 1
        timeseries = (timeseries - timeseries.mean()) / sigma   
        noise_level = r

    return cython_approximate_entropy(timeseries.values, m, noise_level)

def sample_entropy(timeseries, m=2, r=0.2, detrend=True, sigma_units=True):
    """
    Wrapper for the cythonized version to calculate the sample entropy,
    as first introduced by Richman and Moorman, Physiological time-series analysis using approximate entropy and sample entropy, 2000

    timeseries: pd.Series of timeseries values, has to be convertible to float
    m: Window size
    r: Noise level
    detrend: Whether or not to normalize time series to mean zero and unit variance.
    """
    N = len(timeseries.index)

    if timeseries.isna().any() or m + 1 > N:
        return np.nan

    #standard deviation is saved separately because it is used to calculate the noise level parameter for ApEn
    sigma = timeseries.std()

    noise_level = r 
    if sigma_units:
        noise_level *= sigma

    #if the values are all the same, ApEn is zero 
    if isclose(sigma, 0.0):
        return 0.0

    if detrend:
        #normalization to mean 0, variance 1
        timeseries_series = (timeseries - timeseries.mean()) / sigma  
        noise_level = r  

    return cython_sample_entropy(timeseries.values, m, noise_level)
    
#TODO: Research possibilities to cythonize the "getting the unique permutations" part
def permutation_entropy(timeseries, n=5, lag=1, normalize=True):
    """
    Subroutine for calculating the permutation entropy of a timeseries, 
    as given in Bandt, Pompe: Permutation entropy — a natural complexity measure for time series (2002)

    n: Permutation order, commonly assumed between 3 and 7 
    lag: Lag parameter, determines the sliding of the computational window. Default is 1
    """

    N = len(timeseries.index)

    #for permutation orders larger than the series length, permutation entropy is undefined
    if n > N:
        return np.nan
    
    if lag == "same":
        lag = n

    #we divide by log(n!) = lngamma(n+1) to normalize everything to 1 
    #thanks to the upper bound H(n) < log(n!) where H(n) is the permutation entropy of order n
    norm_factor = (np.log(2.0) / sc.gammaln(n+1)) if normalize else 1.0
    vals = timeseries.values

    num_partitions = (N - n) // lag + 1

    perms = np.zeros((num_partitions, n)) 
        
    for i in range(num_partitions):
        perms[i,:] = vals[i*lag:i*lag+n]

    #this function returns for a given array the indices that would sort this array ascendingly by value.
    #axis=1 specifies that each row, i.e. all the windows of the timeseries should be sorted.
    permutations = np.argsort(perms, axis=1)

    #gives the amount of unique occurrences of each permutation present in the data.
    _, perm_counts = np.unique(permutations, axis=0, return_counts=True)

    #normalize everything by the counts
    probabilities = perm_counts / num_partitions

    perm_entropy = np.dot(probabilities, np.log2(probabilities))

    if isclose(perm_entropy, 0.0):
        return perm_entropy * norm_factor

    return -perm_entropy * norm_factor

def integer_representation_entropy(timeseries, n=5, lag=1, normalize=True):
    """
    Subroutine for calculating the integer entropy of a timeseries.
    Experimental concept, based on the permutation entropy by Bandt & Pompe

    n: Length of the integer representation, commonly assumed between 3 and 7 
    lag: Lag parameter, determines the sliding of the computational window. Default is 1
    """

    N = len(timeseries.index)

    #for permutation orders larger than the series length, permutation entropy is undefined
    if n > N:
        return np.nan
    
    if lag == "same":
        lag = n

    #we divide by log(n!) = lngamma(n+1) to normalize everything to 1 
    #thanks to the upper bound H(n) < log(n!) where H(n) is the permutation entropy of order n
    norm_factor = (1 / n) if normalize else 1.0

    vals = timeseries.values.astype(int)

    num_partitions = (N - n) // lag + 1

    binary_reps = np.zeros((num_partitions, n)) 
        
    for i in range(num_partitions):
        binary_reps[i,:] = vals[i*lag:i*lag+n]

    #powers of two from 0 to n-1
    integer_reps = binary_reps.dot(2 ** np.arange(n))

    #gives the amount of unique occurrences of each integer present in the data.
    _, int_counts = np.unique(integer_reps, axis=0, return_counts=True)

    #normalize everything by the counts
    probabilities = int_counts / num_partitions

    #we divide by n to normalize everything to 1 
    #thanks to the upper bound H(n) < log_2(2^n) = n where H(n) is the integer representation entropy of order n
    int_entropy = np.dot(probabilities, np.log(probabilities)) / np.log(2.0)

    if isclose(int_entropy, 0.0):
        return int_entropy * norm_factor

    return -int_entropy * norm_factor

def lz_complexity(timeseries):
    """
    Wrapper for the cythonized version to calculate the Lempel-Ziv complexity,
    as first introduced by Lempel and Ziv, On the Complexity of Finite Sequences, 1976

    timeseries: pd.Series of timeseries values, has to be convertible to int
    """

    int_series = timeseries.astype(int)

    string_rep = "".join(str(i) for i in int_series.values)

    if len(string_rep) <= 1:
        return np.nan

    return lempel_ziv_complexity(string_rep)

def fourier_spectrum_entropy(spectrum, cutoff=None, normalize=True):
    """
    Entropy of a power spectrum obtained through a periodogram routine from scipy.signal.
    Parameters:
    spectrum: pd.Series, nonnegative entries. Holds spectrum
    cutoff: float, cutoff frequency. Used to cut off a spectrum 
    normalize: bool, whether to normalize by the entropy of the discrete uniform (discrete maximum entropy distribution)
    """
    try:
        #if we put in a cutoff frequency
        if cutoff is not None:
            spectrum = spectrum[spectrum.index <= cutoff]
        N = len(spectrum.index)
        spectrum = np.abs(spectrum)

        #normalization factor
        max_entropy = (np.log(2.0) / np.log(N)) if normalize else 1.0

        #build a discrete distribution out of the spectrum
        vals = spectrum.values
        #inplace modification setting nans to zero
        np.nan_to_num(vals, copy=False, nan=0.0)
        vals = vals[vals > 0.0]
        if isclose(np.sum(vals), 0.0):
            return 0.0

        discrete_dist_vals = vals / np.sum(vals)

        #first value is close to zero in a detrended series
        spectrum_entropy = np.dot(discrete_dist_vals, np.log(discrete_dist_vals)) / np.log(2.0)

        if isclose(spectrum_entropy, 0.0):
            #if we end up here, entropy is close to zero, so no need to normalize
            return spectrum_entropy
        else:
            return -spectrum_entropy * max_entropy
    except:
        return np.nan

def classical_periodogram_entropy(timeseries, cutoff=None, normalize=True, p_config=None):
    """
    Helper routine for calculating the periodogram and then its entropy for a time series.
    Produces easier to read code in case the periodogram will not be used for anything else.
    """
    if p_config is None:
        raise ValueError("Something went wrong")

    p = get_periodogram(timeseries, **p_config)

    return fourier_spectrum_entropy(p, cutoff=cutoff, normalize=normalize)

def welch_entropy(timeseries, cutoff=None, normalize=True, p_config=None):
    """
    Helper routine for calculating the Welch periodogram and then its entropy for a time series.
    Produces easier to read code in case the periodogram will not be used for anything else.
    """
    if p_config is None:
        raise ValueError("Something went wrong")

    wp = get_welch_periodogram(timeseries, **p_config)

    return fourier_spectrum_entropy(wp, cutoff=cutoff, normalize=normalize)

def lombscargle_entropy(timeseries, cutoff=None, normalize=True, **ls_config):
    """
    Helper routine for calculating the Welch periodogram and then its entropy for a time series.
    Produces easier to read code in case the periodogram will not be used for anything else.
    """

    lsp = get_lombscargle_periodogram(timeseries, **ls_config)

    return fourier_spectrum_entropy(lsp, cutoff=cutoff, normalize=normalize)

def shannon_entropy(timeseries, normalize=True):
    """
    Helper for calculating Shannon entropy of a binary timeseries.
    """

    #estimate of the success probability
    N = len(timeseries.index)
    p = timeseries.sum() / N

    if isclose(p, 1.0) or isclose(p, 0.0):
        return 0.0

    norm_factor = -1.0

    shannon_ent = p * np.log2(p) + (1-p) * np.log2(1-p)
    return norm_factor * shannon_ent

def shannon_markov_entropy(timeseries, normalize=True):
    """
    Helper for calculating Shannon entropy of a binary timeseries.
    """

    #estimate of the success probability
    N = len(timeseries.index)
    p = timeseries.sum() / N
    conditional_probs = np.zeros(4)

    vals = timeseries.values.astype(int)

    if isclose(p, 1.0) or isclose(p, 0.0):
        return 0.0

    for i in range(N-1):
        if vals[i] == vals[i+1]:
            #0 -> 0 or 1 -> 1
            conditional_probs[vals[i]] += 1
        else:
            #0 -> 1 or 1 -> 0
            conditional_probs[vals[i] + 2] += 1

    conditional_probs /= N
    nonzero = conditional_probs > 0.0

    cond_entropy = conditional_probs[nonzero] * np.log2(conditional_probs[nonzero])
    probs = np.array([1.0 - p, p, 1.0 - p, p])
    norm_factor = -1.0

    shannon_markov_ent = np.sum(probs[nonzero] * cond_entropy)
    return norm_factor * shannon_markov_ent

def multiscale_entropy(timeseries, stat_name, scales, **kwds):
    """
    Calculates multi-scale entropy for any of the entropies defined above.
    Idea from Costa, Goldberger and Peng, "Multiscale Entropy Analysis of Complex Physiologic Time Series", Phys. Rev. Lett. 89, No.6, July 2002
    timeseries: pd.Series, data on which to perform the analysis
    stat_name: Has to be "apen", "sampen", "permen", "intrepen", entropy measure to investigate
    scales: List of time scale factors to use in MSE calculation.
    freq: string, used for resampling the pd.Series. 
    kwds: Arguments passed on to the entropy statistic.

    Returns: pd.Series of entropy corresponding to the different scaling factors
    """
    entropy_vals = np.full(len(scales), np.nan)

    stats = {"apen": approximate_entropy, "sampen": sample_entropy, "permen": permutation_entropy, "fse": fourier_spectrum_entropy}

    stat_key = stat_name.lower()

    #print separate error message for clarity, would otherwise be caught be the try-except below
    if stat_key not in list(stats.keys()):
        raise ValueError("Error: Statistic name not recognized. Available options are: \"apen\", \"sampen\", \"permen\", \"fse\"")

    for i, scale in enumerate(scales):            
        try:
            resampled_series = timeseries.resample(scale).mean()
            entropy_vals[i] = stats.get(stat_name.lower())(resampled_series, **kwds)
        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
            else:
                print(e)

    return pd.Series(entropy_vals, index=scales)

def sliding_window_calculation(timeseries, stat_name, window_size=30):
    """
    Calculates a given regularity statistic specified by stat_name 
    for a user in a sliding window of size window_size.
    """
    funcs = {"apen": approximate_entropy, "sampen": sample_entropy, "intrepen": integer_representation_entropy, "permen": permutation_entropy, "lz_complexity": lz_complexity}

    return timeseries.rolling(window_size).apply(funcs[stat_name.lower()])

def stat_evolution_calculation(timeseries, stat_name):       
    """
    Calculates a given regularity statistic specified by stat_name 
    for a user on the data in an incrementing fashion.
    """
    funcs = {"apen": approximate_entropy, "sampen": sample_entropy, "intrepen": integer_representation_entropy, "permen": permutation_entropy, "lz_complexity": lz_complexity}

    return timeseries.expanding().apply(funcs[stat_name.lower()])