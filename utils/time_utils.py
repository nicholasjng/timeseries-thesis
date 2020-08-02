import pandas as pd
import numpy as np
from scipy import signal

import os

def get_timedeltas(login_timestamps, return_floats=True):
    
    """
    Helper function that returns the time differences (delta t's) between consecutive logins for a user.
    We just input the datetime stamps as an index, hence this method will also work when called on a DataFrame of 
    customer logins.
    
    Parameters: 
    
    login_timestamps (pd.Series): DatetimeIndex from a series or dataframe with user logins. Can be used on both binary
    timeseries as returned by the method construct_binary_visit_series (see above) or from the DataFrame holding the 
    logins directly.
    
    return_floats (bool): Whether or not to return the times as timedifferences (pd.Timedelta objects) or floats.
    
    Returns: 
    timedeltas (list of objects): List of time differences, either in pd.Timedelta format or as floats.
    
    """
    if len(login_timestamps.index) <= 1:
        raise ValueError("Error: For computing time differences, the user must have more than one registered login")
    
    #get the dates on which the customer visited the gym
    timedeltas = pd.Series(login_timestamps.diff().values, index=login_timestamps.values)
    
    #realign the series so that a value on a given date represents the time in days until the next visit
    timedeltas.shift(-1)
    timedeltas.dropna(inplace=True)
    
    if return_floats:
        timedeltas = timedeltas / pd.Timedelta(days=1)
        
    return timedeltas
    
def write_timedeltas_to_file(login_data, filename, is_sorted=False, num_users=None, minimum_deltas=2, verbose=False, compression="infer"):
    """
    Function to write timedelta data to a file for HMM analysis.

    login_data: pd.DataFrame, login_data for analysis 
    filename: Output write 
    num_users: Number of sequences to write, default None (= write whole dataset)
    compression: pandas compression type
    """

    if os.path.exists(os.getcwd() + "/" + filename):
        print("The file specified already exists. It will be overwritten in the process.")
        os.remove(filename)

    #get all visits from 
    visit_numbers = login_data["CUST_CODE"].value_counts().astype(int)

    #visit number must be larger than minimum_deltas, since we need two timedeltas for HMM estimation
    eligibles = visit_numbers[visit_numbers > minimum_deltas]

    ineligibles_data = login_data[~login_data.CUST_CODE.isin(eligibles.index)]

    login_data_cleaned = login_data.drop(ineligibles_data.index)

    if not is_sorted:
        #sort the data by both customer code and date, this avoids problems with date ordering later
        login_data_cleaned.sort_values(by=["CUST_CODE", "DATE_SAVED"], inplace=True)

    num_logins = len(login_data_cleaned.index)

    if num_users is None:
        num_users = len(eligibles.index)

    #customer counter, can be printed in verbose mode
    count = 0
    index = 0
    nonsense_counts = 0

    while index < num_logins:
        cust_code  = login_data_cleaned.iloc[index].CUST_CODE
        customer_visits = eligibles[cust_code]

        count += 1  

        if verbose and (count % 100 == 0 or count == num_users):
            print("Processed {} customers out of {}".format(count, num_users))    

        #select logins with the specified customer code
        customer_logins = login_data_cleaned.iloc[index:index+customer_visits]  

        visiting_dates =  customer_logins.DATE_SAVED #pd.DatetimeIndex([visit_date for visit_date in customer_logins.DATE_SAVED])

        #extract the timedeltas
        timedeltas = get_timedeltas(visiting_dates, return_floats=True)
        
        #since timedeltas involve differencing, the first value will be NaN - we drop it
        timedeltas.dropna(inplace=True)

        #logins with timedelta under 5 minutes are dropped
        thresh = 5 * (1 / (24 * 60))
        
        #drop all timedeltas under the threshold
        eligible_tds = timedeltas[timedeltas > thresh] 

        if len(eligible_tds.index) < minimum_deltas:
            nonsense_counts += 1 
            index += customer_visits
            continue
        
        timedeltas_df = eligible_tds.to_frame().T

        #mode='a' ensures that the data are appended instead of overwritten
        timedeltas_df.to_csv(filename, mode='a', header=False, compression=compression, index=False, sep=";")

        if count >= num_users:
            break 
    
        index += customer_visits
    
    print("Found {} users with too many artefact logins".format(nonsense_counts))

def get_timedelta_sample(login_data, is_sorted=False, num_users=None, minimum_deltas=2, verbose=False):
    """
    Function to write timedelta data to a file for HMM analysis.

    login_data: pd.DataFrame, login_data for analysis 
    filename: Output write 
    num_users: Number of sequences to write, default None (= write whole dataset)

    """

    #get all visits from 
    visit_numbers = login_data["CUST_CODE"].value_counts().astype(int)

    #visit number must be larger than minimum_deltas, since we need two timedeltas for HMM estimation
    eligibles = visit_numbers[visit_numbers > minimum_deltas]

    ineligibles_data = login_data[~login_data.CUST_CODE.isin(eligibles.index)]

    login_data_cleaned = login_data.drop(ineligibles_data.index)

    if not is_sorted:
        #sort the data by both customer code and date, this avoids problems with date ordering later
        login_data_cleaned.sort_values(by=["CUST_CODE", "DATE_SAVED"], inplace=True)

    num_logins = len(login_data_cleaned.index)

    if num_users is None:
        num_users = len(eligibles.index)

    #customer counter, can be printed in verbose mode
    count = 0
    index = 0
    delta_index = 0

    num_deltas = eligibles.sum() - len(eligibles.index) 

    timedelta_sample = np.zeros(num_deltas)

    while index < num_logins:
        cust_code  = login_data_cleaned.iloc[index].CUST_CODE
        customer_visits = eligibles[cust_code]
    
        #select logins with the specified customer code
        customer_logins = login_data_cleaned.iloc[index:index+customer_visits]  

        visiting_dates =  customer_logins.DATE_SAVED

        #extract the timedeltas
        timedeltas = get_timedeltas(visiting_dates, return_floats=True)

        #since timedeltas involve differencing, the first value will be NaN - we drop it
        timedeltas.dropna(inplace=True)

        #add list
        try:
            timedelta_sample[delta_index:delta_index+customer_visits-1] = timedeltas.values
        except:
            print("#index: {}".format(index))
            print("#length of td vector: {}".format(num_deltas))

        count += 1 

        if count >= num_users:
            if verbose:
                print("Checked {} customers out of {}".format(count, num_users))
            break

        if verbose and (count % 100 == 0): 
            print("Checked {} customers out of {}".format(count, num_users))        
        index += customer_visits
        delta_index += customer_visits - 1 

    #threshold of 5 minutes to sort out artifact logins 
    thresh = 5 * (1 / (24 * 60))

    td_sample = pd.Series(timedelta_sample) 
    td_sample = td_sample[td_sample > thresh]

    return td_sample   

def get_periodogram(timeseries, fs=1.0, window="hann", window_size=14, detrend="constant", return_onesided=True, scaling="density", mode="radians"):
    """
    Wrapped method that returns the smoothed periodogram for an input time series.
    Arguments:
    timeseries: pd.Series object
    fs: Sampling or binning rate
    window: str, matches the options in scipy.signal.get_window
    window_size: int, positive. Window size for smoothing
    detrend: None, "constant" or "linear". If not None, a call to scipy.signal.detrend will be made.
    scaling: Whether or not to normalize the periodogram.
    """
    N = len(timeseries.index)

    freqs, periodogram = signal.periodogram(timeseries.values, fs=fs, detrend=detrend, return_onesided=return_onesided, scaling=scaling)

    if mode == "radians":
        #if radians mode, scale by 2 * pi 
        freqs *= 2 * np.pi 

    periodogram /= N

    #undo some of the periodogram normalization BS 
    if return_onesided:
        if N % 2 == 0:
            periodogram[1:] /= 2
        else:
            periodogram[1:-1] /= 2

    #get window function
    if window != "none":
        window = signal.get_window(window, window_size)
        window = window / np.sum(window)

        #smooth the periodogram by convolving with a window function
        smoothed_periodogram = signal.convolve(periodogram, window, mode="same")

        return pd.Series(smoothed_periodogram, index=freqs)
    
    return pd.Series(periodogram, index=freqs)

def get_welch_periodogram(timeseries, fs=1.0, window="hann", nperseg=128, noverlap=None, nfft=256, detrend="constant", return_onesided=True, scaling="density", mode="radians"):
    """
    Wrapped method that returns Welch's periodogram for an input time series.
    Arguments:
    timeseries: pd.Series object
    fs: Sampling or binning rate
    window: str, matches the options in scipy.signal.get_window
    nperseg: Values per segment
    noverlap: Overlap between segments
    nfft: Length of the FFT performed on the bits
    detrend: None, "constant" or "linear". If not None, a call to scipy.signal.detrend will be made.
    scaling: str, gives normalization of the periodogram 
    """
    freqs, welch_periodogram = signal.welch(timeseries.values, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, nfft=nfft, detrend=detrend, return_onesided=return_onesided, scaling=scaling)

    #undo some of the periodogram normalization BS 
    if return_onesided:
        if nfft % 2 == 0:
            welch_periodogram[1:] /= 2
        else:
            welch_periodogram[1:-1] /= 2

    if mode == "radians":
        #if radians mode, scale by 2 * pi 
        freqs *= 2 * np.pi 

    return pd.Series(welch_periodogram, index=freqs)

def get_lombscargle_periodogram(timedeltas, normalize=True, mode="radians"):

    N = len(timedeltas.index)
    obs_times = np.cumsum(pd.concat(pd.Series([0]),timedeltas))

    #data are ones, i.e. binary encoded visits
    data = np.ones(N + 1)
    
    #5 times 4 / day (max frequency) * observation timeframe 
    num_freqs = 5 * 12 * obs_times[-1]

    end = np.pi if mode == "radians" else 0.5

    freq_grid = np.linspace(0, 12, num=num_freqs, endpoint=True)

    pgram = signal.lombscargle(obs_times, data, freq_grid, normalize=normalize)

    return pd.Series(pgram, index=freq_grid)

def construct_binary_visit_series(customer, customer_logins, last_date, freq="1D", mode="cd"):
    """
    Small subroutine to construct a daily binary visit pattern from visit data for a given customer.
    customer: Customer database entry
    customer_logins: Logins from login data where CUST_CODE == customer.CUST_CODE
    last_date: Last day of login data, needed to limit the active range of people still active
    freq: String giving the date frequency for how to bin the timeseries data 
    mode: String, timeseries mode. "cd": Contract duration, length of the series is from membership to expiry date.
    "ac": Active duration, length of the series is from first to last visit.
    """          
    
    #get the visiting dates from the logins
    visiting_dates = pd.DatetimeIndex(customer_logins.DATE_SAVED).floor(freq)

    start, end = None, None

    if mode == "cd":
        start = customer.MEMBER_SINCE.date()
        end = (last_date if customer.ISACTIVE else customer.EXPIRE_DATE) + pd.Timedelta("1D")
    elif mode == "ad":
        start = visiting_dates[0]
        end = visiting_dates[-1]
    else:
        raise ValueError("Unrecognized binary time series mode")

    #construct the date range
    active_range = pd.date_range(start=start, end=end, freq=freq)

    res = pd.Series(0.0, index=active_range)
    try:
        eligible_dates = visiting_dates[visiting_dates.isin(active_range)]
        res.loc[eligible_dates] = 1.0
    except KeyError:
        print("Start: {}".format(start))
        print("End: {}".format(end))
        print("Expire Date: {}".format(customer.EXPIRE_DATE))
        print("Renew Date: {}".format(customer.RENEW_DATE))
        print(visiting_dates[~visiting_dates.isin(active_range)])
        return res
    #construct the binary timeseries by checking on which of the dates in the active 
    #time frame the customer went to the gym. Only applies if there are logins
    return res

def binary_user_from_td(timedeltas):
    """
    Small subroutine to construct a daily binary visit pattern from visit data for a given customer.
    timedeltas: pd.Series with 
    """   
    #get the visiting dates from the logins
    visiting_dates = pd.DatetimeIndex([d.date() for d in timedeltas.index])

    #construct the date range
    active_range = pd.date_range(start=visiting_dates[0], end=visiting_dates[-1])

    #construct the binary timeseries by checking on which of the dates in the active 
    #time frame the customer went to the gym. Only applies if there are logins
    return pd.Series(active_range.isin(visiting_dates).astype(int), index=active_range)