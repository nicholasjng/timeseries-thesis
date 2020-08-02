import numpy as np 
import pandas as pd 

def get_monthly_cps(changepoints):
    """
    Compute months for a list of detected changepoints.
    CPs have to have been previously converted to pd.Datetime objects.
    """
    #holds changepoint percentages per Season
    cps = np.zeros(12)

    num_changes = len(changepoints)

    #minus 1 to obtain eligible array indices
    cp_seasonals = (np.array([cp.month for cp in changepoints])) - 1

    for cp in cp_seasonals:
        cps[cp] += 1

    return cps

def get_seasonal_cps(changepoints):
    """
    Compute seasons for a list of detected changepoints.
    CPs have to have been previously converted to pd.Datetime objects.
    Seasonal model used is basically meteorological seasons:
    Season 0 = Winter (01.12. - 28./29.02.)
    Season 1 = Spring (01.03. - 31.05.)
    Season 2 = Summer (01.06. - 31.08.)
    Season 3 = Autumn (01.09. - 30.11.)
    """
    #holds changepoint percentages per Season
    cps = np.zeros(4)

    #modulo 12 to wrap december around, then integer division by three gives a number between 0 and 3
    cp_seasonals = (np.array([cp.month for cp in changepoints]) % 12) // 3

    for cp in cp_seasonals:
        cps[cp] += 1

    return cps

def get_quarterly_cps(changepoints):
    """
    Compute business quarters for a list of detected changepoints.
    CPs have to have been previously converted to pd.Datetime objects.
    Quarterly model used is business quarters:
    Quarter 0 = 01.01. - 31.03.
    Quarter 1 = 01.04. - 30.06.
    Quarter 2 = 01.07. - 30.09.
    Quarter 3 = 01.10. - 31.12.
    """
    cps = np.zeros(4)
        
    #6 is the size of a part of day, integer division gives a number between 0 and 3
    cp_quarterlies = (np.array([cp.month for cp in changepoints]) - 1) // 3

    for cp in cp_quarterlies:
        cps[cp] += 1

    return cps

def startofyear_cps(changepoints, thresh=5):
    """
    Determine whether the person has change points in a given model at the start of a new year, 
    specifically in the first thresh (optional argument) days of the year.
    """
    num_soy_cps = 0

    for cp in changepoints:
        if (cp.month == 1) and (cp.day <= thresh):
            num_soy_cps += 1
    
    return num_soy_cps

def changepoint_duration_moments(changepoints):
    """
    Compute moments (mean, median, min, max) for the differences in found change points.
    Aims to characterize the distribution of changepoints in time.
    """
    moments = np.zeros(4)

    # deltas between change points
    cp_deltas = changepoints.to_series().diff()
    
    cp_delta_vals = cp_deltas.values[1:] / pd.Timedelta(days=1)
    
    moments[0] = np.mean(cp_delta_vals)
    moments[1] = np.median(cp_delta_vals)
    moments[2] = np.amin(cp_delta_vals)
    moments[3] = np.amax(cp_delta_vals)

    return moments
    
def early_contract_cps(changepoints, thresh=30):
    """
    Compute changepoints closer than "thresh" (optional argument) days to the start of the contract.  
    CPs should be given as obtained by the model.predict() method from the ruptures package, 
    i.e. a list of positive integers representing days since the start of the contract,
    with the last element being the final day of the contract.
    """
    early_cps = 0

    for cp in changepoints:
        if (cp > 0) and (cp <= thresh):
            early_cps += 1
    
    return early_cps

def late_contract_cps(changepoints, thresh=60):
    """
    Compute changepoints closer than "thresh" (optional argument) days to the start of the contract.  
    CPs should be given as obtained by the model.predict() method from the ruptures package, 
    i.e. a list/numpy array of positive integers representing days since the start of the contract,
    with the last element being the final day of the contract.
    """
    late_cps = 0

    contract_end = changepoints[-1]

    for cp in changepoints:
        delta = contract_end - cp
        if (delta > 0) and (delta <= thresh):
            late_cps += 1
    
    return late_cps