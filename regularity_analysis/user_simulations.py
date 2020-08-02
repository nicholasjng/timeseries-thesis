import pandas as pd
import numpy as np

from math import isclose

from hmmlearn import hmm

def simulate_gym_user(num_days=1000, noise_frac=0.0, periodicity=3, success_prob=0.5):
    """
    Small routine returning an artificial binary visit timeseries of a very regular user for regularity analysis.
    time_window: pd.DatetimeIndex, used as index for the timeseries
    visit_dates: List, values between 0 and 6, indicating days in the week where the simulated user visits
    visits_per_week: Integer. Gives the visits per week if visit_dates is not specified. Then, exactly visits_per_week
    random visit days are drawn.
    """
    data = np.zeros(num_days)

    data[::periodicity] = 1.0
    user = pd.Series(data)

    #corrupt data by binary white noise with success probability p, which means it has variance p(1-p)
    if not isclose(noise_frac, 0.0):
        irreg_days = user.sample(frac=noise_frac)
        irreg_index = irreg_days.index
        user.loc[irreg_index] = np.random.binomial(1, success_prob, len(irreg_index))

    return user.astype(float)

def simulate_user_with_mixture(start_date, means, variances, weights, mixture_type="gaussian", num_samples=100):
    """
    Function to simulate a user's gym usage behavior by constructing a series of timestamps. 
    By design, it generates a series of gym visit dates by sampling from a Gaussian mixture model 
    P(x) = sum_i x_i N(mu_i, sigma_i^2), where the x_i are giving the mixture weights, 
    specified in the "weights" argument, and the (mu_i, sigma_i^2) are the mean and variance parameters for 
    the constituting Gaussian distributions.
    Since a fixed number of samples is drawn, the stop date is unknown a priori. 

    Arguments: 
    start_date: str, gives the start date has to be interpretable by the pd.Timestamp constructor. 
    means: np.ndarray, shape (N, ). Means of the component distributions
    variances: np.ndarray, shape (N, ). Means of the component distributions
    weights: np.ndarray, shape (N, ). Mixture weights, must sum up to 1.
    mixture: str, either "gaussian" or "gamma" depending on mixture type
    num_samples: integer, positive. Number of samples to be drawn. 

    Returns:
    sim, a simulated pd.Series of timedeltas between visits. Index is the timestamps of visits rounded to seconds
    """
    # Parameters of the mixture components
    n_components = len(means)

    if n_components != len(weights):
        raise ValueError("Error: Number of Gaussians and length of weight vector have to be equal. A number of {} Gaussians was specified, and a total of {} weights was given.".format(n_components, len(weights)))
    
    #if the weights are not normalized, normalize them 
    if not isclose(np.sum(weights), 1.0):
        weights = weights / np.sum(weights)

    start = pd.Timestamp(start_date)

    # A stream of indices from which to choose the Gaussian mixture component
    mixture_idx = np.random.choice(n_components, size=num_samples, replace=True, p=weights)

    mix_means = means[mixture_idx]
    mix_vars = variances[mixture_idx]

    td_samples = None

    if mixture_type.lower() == "gaussian":
        # mixture sample, abs applied to prevent negative times 
        td_samples = np.abs(np.random.normal(size=num_samples) * np.sqrt(mix_vars) + mix_means) 
    elif mixture_type.lower() == "gamma":
        #shape and scale parameters of the mixture gammas, scipy.stats parametrizes by those instead of mean/variance
        #scale = variance / mean, shape = mean / scale = variance / scale^2
        gamma_scales, gamma_shapes = np.zeros_like(mix_means), np.zeros_like(mix_vars)

        mix_scales = mix_vars / mix_means
        mix_shapes = mix_means / mix_scales
        td_samples = np.random.gamma(mix_shapes, mix_scales)

    #argument is a np.array of pd.Timestamps, rounded to seconds
    timestamps = pd.DatetimeIndex(pd.to_timedelta(np.cumsum(td_samples), unit="days") + start).round("1s")

    sim = pd.Series(td_samples, index=timestamps)

    return sim 

def simulate_user_with_hmm(start_date, gaussians, weights, transitions, num_samples=100):
    """
    Function to simulate a user's gym usage behavior by utilizing a predefined Hidden Markov Model (HMM). 
    It generates a series of gym visit dates by sampling a visit sequence from an HMM whose parameters 
    (Transition matrix, emission matrix, initial distribution etc.) are specified by keyword arguments. 

    Since the HMM gives an opportunity to model transitions in the mixture distributions, it is a genuine extension of
    the Gaussian / Gamma simulation above. Conversely, a Gaussian mixture simulation can be obtained by specifying the 
    transition matrix as np.ones((n, n)) / n where n is the number of mixture components or the number of states in the
    Hidden Markov model. 

    Arguments:
    """

    n_components = gaussians.shape[0]

    if n_components != len(weights):
        raise ValueError("Error: Number of Gaussians and length of weight vector have to be equal. A number of {} Gaussians was specified, and a total of {} weights was given.".format(n_components, len(weights)))

    #if the weights are not normalized, normalize them 
    if not isclose(np.sum(weights), 1.0):
        weights = weights / np.sum(weights)

    start = pd.Timestamp(start_date)

    means = gaussians[:,0].reshape((-1, 1))
    variances = gaussians[:,1]

    #initialize HMM, n_features=1 so we can use the "spherical" keyword to set a single variance 
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="spherical")

    model.startprob_ = weights
    model.transmat_ = transitions
    model.means_ = means
    model.covars_ = variances

    td_samples, Z_samples = model.sample(num_samples)

    td_samples = np.squeeze(td_samples)

    #argument is a np.array of pd.Timestamps, rounded to seconds
    timestamps = pd.DatetimeIndex(pd.to_timedelta(np.cumsum(td_samples), unit="days") + start).round("1s")

    sim = pd.Series(td_samples, index=timestamps)

    return sim 