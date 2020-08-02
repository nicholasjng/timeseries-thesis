import os
import numpy as np
import pandas as pd 

from math import isclose

import matplotlib.pyplot as plt 

import statsmodels.api as sm
import statsmodels.tsa.api as smt
import ruptures as rpt
import ruptures.metrics as rptm
from itertools import cycle

from ruptures.utils import pairwise

COLOR_CYCLE = ["#4286f4", "#f44174"]

from scipy.stats import norm, gamma
from utils.time_utils import get_periodogram, binary_user_from_td, construct_binary_visit_series, get_periodogram, get_welch_periodogram
from utils.stat_utils import aic_penalty, bic_penalty, hqc_penalty

def regularity_histogram(data, stats, plot_settings, hist_config, output_dir, count=None, single_stat=False):
    """
    Small subroutine for computing a regularity statistic histogram for multiple hyperparameters.
    data: Statistics data frame
    stats: list of strs, statistic names(match dataframe columns)
    plot_settings: dict, general plot settings like file format, dpi, figsize etc.
    hist_config: dict, histogram configuration
    hp_key: str, secondary hyperparameter key for the statistics (for column name matching)
    count: int, suffix for unique file identification
    """
    #general plot settings
    dpi = plot_settings["dpi"]
    fig_ext = plot_settings["fig_ext"]

    img_filename = output_dir

    if single_stat:
        img_filename = img_filename + stats[0] + "_regularity_histogram" 
    else:
        img_filename = img_filename + "_regularity_histogram_mixed"

    #filter the data by list of statistics, exact matching
    stat_data = data.filter(items=stats)

    if count is not None:
        img_filename = img_filename + "_" + str(count)

    ax = stat_data.plot.hist(**hist_config)
    fig = ax.get_figure()

    fig.savefig(img_filename + fig_ext, dpi=dpi)
    ax.clear()

def regularity_hexbin_plot(data, stat_x, stat_y, plot_settings, hexbin_config, output_dir):
    """
    Small subroutine for plotting two statistics on a hexbin plot (histogram + probability density estimate).
    data: Statistics data frame
    stat_x: str, statistic name on the x axis
    stat_y: str, statistic name on the y axis
    plot_settings: dict, general plot settings like file format, dpi, figsize etc.
    hexbin_config: dict, hexbin plot configuration
    """
    
    #general plot settings
    dpi = plot_settings["dpi"]
    fig_ext = plot_settings["fig_ext"]

    stat_data = None 
    img_filename = output_dir + stat_x + "_" + stat_y + "_" + "hexbin_plot"

    ax = data.plot.hexbin(x=stat_x, y=stat_y, **hexbin_config)
    fig = ax.get_figure()

    fig.savefig(img_filename + fig_ext, dpi=dpi)
    ax.clear()    

def regularity_scatter_plot(data, stat_x, stat_y, plot_settings, scatter_config, output_dir):
    """
    Small subroutine for plotting two statistics on a hexbin plot (histogram + probability density estimate).
    data: Statistics data frame
    stat_x: str, statistic name on the x axis
    stat_y: str, statistic name on the y axis
    plot_settings: dict, general plot settings like file format, dpi, figsize etc.
    hexbin_config: dict, hexbin plot configuration
    """
    
    #general plot settings
    dpi = plot_settings["dpi"]
    fig_ext = plot_settings["fig_ext"]

    stat_data = None 
    img_filename = output_dir + stat_x + "_" + stat_y + "_" + "scatter_plot"

    ax = data.plot.scatter(x=stat_x, y=stat_y, **scatter_config)
    fig = ax.get_figure()

    fig.savefig(img_filename + fig_ext, dpi=dpi)
    ax.clear()    

def plot_user_histogram_with_density(user, means, variances, weights, mixture_type="gaussian", ax=None, show=False):
    """
    Small routine for plotting the user regularity histograms together with the probability density function of the used mixture model.
    Arguments:
    user (pd.Series): Output argument of simulate_user_xxx_timedeltas, where xxx = "gaussian" or "gamma".
    means: np.ndarray, shape = (N,): Means of the mixture components. 
    variances: np.ndarray, shape = (N,): Variances of the mixture components. 
    weights: np.ndarray, shape = (N,): Weights in the mixture model. 
    mixture: str, "gaussian" for a Gaussian Mixture Model or "gamma" for a Gamma Mixture Model.
    """
    
    if mixture_type.lower() not in ["gaussian", "gamma"]:
        raise ValueError("Error: Unrecognized mixture type. Available mixture models are: \"gaussian\", \"gamma\").")
        
    if len(means) != len(weights):
        raise ValueError("Error: Dimension mismatch. Number of mixture components: {}, number of weights: {}".format(len(means), len(weights)))
    
    #if the weights are not normalized, normalize them 
    if not isclose(np.sum(weights), 1.0):
        weights = weights / np.sum(weights)

    # Theoretical PDF plotting -- generate the x and y plotting positions
    #alternatively, 0, 7 user.min(), user.max()
    xs = np.linspace(user.min(), user.max(), 200, endpoint=True)
    ys = np.zeros_like(xs)

    for l, s, w in zip(means, variances, weights):
        if mixture_type.lower() == "gaussian":
            ys += norm.pdf(xs, loc=l, scale=s) * w
        else:
            scale = s/l
            shape = l/scale
            ys += gamma.pdf(xs, a=shape, loc=0, scale=scale) * w

    if ax is not None:
        ax.plot(xs, ys, lw=2)
        user.plot.hist(density=True, bins="fd", ax=ax)
        ax.set_xlabel("x")
        ax.set_ylabel("p(x)")
    else:
        plt.plot(xs, ys, lw=2)
        user.plot.hist(density=True, bins="fd")
        plt.xlabel("x")
        plt.ylabel("p(x)")
    
    if show:
        plt.show()

def tsplot(y, lags=None, dpi=220, style='bmh'):
    """
        Plot time series, its ACF and PACF, calculate Dickey–Fuller test
        
        y - timeseries
        lags - how many lags to include in ACF, PACF calculation
    """
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
        
    with plt.style.context(style):    
        plt.figure(figsize=(3840/dpi, 2160/dpi), dpi=dpi)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))

        acf_ax.set_xlabel("Lag h")
        pacf_ax.set_xlabel("Lag h")
        
        y.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(y)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax, method="ld")
        plt.tight_layout()

def tsplot_periodogram(cust_code, result_data, login_data, stat_name, dpi=220, style='bmh', p_type="classical", freq="1D", **p_config):

    customer_logins = login_data[login_data["CUST_CODE"] == cust_code]
    customer = result_data.loc[cust_code]
    bin_series = construct_binary_visit_series(customer, customer_logins, login_data["DATE_SAVED"].max(), freq=freq)

    periodogram = None 

    if p_type not in ["classical", "welch"]:
        raise ValueError("Error: Unknown periodogram type selected")

    if p_type == "classical":
        periodogram = get_periodogram(bin_series, **p_config)
    elif p_type == "welch":
        periodogram = get_welch_periodogram(bin_series, **p_config)

    with plt.style.context(style):    
        plt.figure(figsize=(3840/dpi, 2160/dpi), dpi=dpi)
        layout = (2, 2)
        ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
        p_ax = plt.subplot2grid(layout, (1, 0), colspan=2)

        bin_series.plot(ax=ts_ax)
        periodogram.plot(ax=p_ax, label="{} Periodogram".format(p_type.capitalize()))
        
        ts_ax.set_title("Fourier Spectrum Entropy: {0:.4f}".format(customer[stat_name]))
        plt.tight_layout()

def tsplot_cpd(cust_code, result_data, login_data, dpi=220, style='bmh', freq="1D", model="l2", min_size=2, jump=5, ic="bic"):

    customer_logins = login_data[login_data["CUST_CODE"] == cust_code]
    customer = result_data.loc[cust_code]
    bin_series = construct_binary_visit_series(customer, customer_logins, login_data["DATE_SAVED"].max(), freq=freq)

    penalty_funcs = {"aic": aic_penalty, "bic": bic_penalty, "hqc": hqc_penalty}

    pelt_instance = rpt.Pelt(model=model, min_size=min_size, jump=jump).fit(bin_series.values)
    binseg_instance = rpt.Binseg(model=model, min_size=min_size, jump=jump).fit(bin_series.values)

    pen = penalty_funcs.get(ic)(bin_series.var(), len(bin_series.index))
    pelt_cps = pelt_instance.predict(pen=pen)
    binseg_cps = binseg_instance.predict(pen=pen)

    hamming_distance = rptm.hamming(pelt_cps, binseg_cps)

    with plt.style.context(style):   
        fig, ax = plt.subplots(sharex=True, figsize=(3840/dpi, 2160/dpi), dpi=dpi) 

        ax.set_title("Hamming Distance to BinSeg segmentation: {0:.3f}".format(hamming_distance))
        ax.set_xlabel("Contract Duration")
        ax.set_ylabel("Binary Visit Indicator")
        bin_series.plot(ax=ax)
        color_cycle = cycle(COLOR_CYCLE)

        # color each (true) regime
        bkps = [0] + sorted(pelt_cps)
        bkps[-1] -= 1
        dates = bin_series.index[bkps]
        alpha = 0.2  # transparency of the colored background

        for (start, end), col in zip(pairwise(dates), color_cycle):
            ax.axvspan(start,
                        end,
                        facecolor=col, alpha=alpha)

        color = "k"  # color of the lines indicating the computed_chg_pts
        linewidth = 3  # linewidth of the lines indicating the computed_chg_pts
        linestyle = "--"  # linestyle of the lines indicating the computed_chg_pts

        for bkp in binseg_cps:
            if bkp != 0 and bkp < len(bin_series.index):
                ax.axvline(x=bin_series.index[bkp],
                            color=color,
                            linewidth=linewidth,
                            linestyle=linestyle)

        plt.tight_layout()