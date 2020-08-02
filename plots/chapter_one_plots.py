import numpy as np
import pandas as pd
import statsmodels as sm
from statsmodels import tsa
from scipy import signal
import matplotlib.pyplot as plt
import os


from utils.plot_utils import tsplot, tsplot_cpd, tsplot_periodogram
from utils.time_utils import get_periodogram, get_welch_periodogram

def autoregressive_example(ar_params):
    """
    Generates a sample of an Gaussian AR(p) process and a plot.
    """
    np.random.seed(123456)
    ar =  np.r_[1, -ar_params] # add zero-lag
    ma = [1]
    y = sm.tsa.arima_process.arma_generate_sample(ar, ma, 500)
    
    tsplot(y, lags=50, dpi=300, style="fivethirtyeight")

    out_dir = "./plots/ch1_plots/"
    out_name = "autoregressive_sample"
    ext = ".pdf"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    plt.savefig(out_dir + out_name + ext, format="pdf", bbox_inches="tight")

def moving_average_example(ma_params):
    """
    Generates a sample of a Gaussian MA(q) process and a plot.
    """
    np.random.seed(12345)
    ar = [1]
    ma = np.r_[1, ma_params] # add zero-lag
    y = sm.tsa.arima_process.arma_generate_sample(ar, ma, 500)

    dpi = 220
    
    tsplot(y, lags=50, dpi=dpi, style="fivethirtyeight")

    out_dir = "./plots/ch1_plots/"
    out_name = "moving_average_sample"
    ext = ".pdf"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    plt.savefig(out_dir + out_name + ext, format="pdf", bbox_inches="tight")

def moving_average_spectral_density(ma_params):
    ar = 1
    ma = np.r_[1, ma_params] # add zero-lag

    dpi = 220
    style = "fivethirtyeight"

    #frequency response
    w, h = signal.freqz(ma, a=ar)

    w = w / (2 * np.pi)

    out_dir = "./plots/ch1_plots/"
    out_name = "moving_average_spectraldensity"
    ext = ".pdf"

    with plt.style.context(style):   

        fig, ax1 = plt.subplots(figsize=(8,5), dpi=dpi)
        ax1.set_title('MA(1) spectral density')   
        ax1.plot(w, np.abs(h) ** 2)
        ax1.set_ylabel(r'Spectral density $f (2\pi \nu)$', color='b')
        ax1.set_xlabel(r'Frequency $\nu$ (radians)')

        fig.savefig(out_dir + out_name + ext, format="pdf", bbox_inches="tight")

def autoregressive_spectral_density(ar_params):
    ar = np.r_[1, -ar_params]
    ma = 1 # add zero-lag

    dpi = 220
    style = "fivethirtyeight"

    #frequency response
    w, h = signal.freqz(ma, a=ar)

    w = w / (2 * np.pi)

    out_dir = "./plots/ch1_plots/"
    out_name = "autoregressive_spectraldensity"
    ext = ".pdf"

    with plt.style.context(style):   

        fig, ax1 = plt.subplots(figsize=(8,5), dpi=dpi)
        ax1.set_title('AR(1) spectral density')   
        ax1.plot(w, np.abs(h) ** 2)
        ax1.set_ylabel(r'Spectral density $f (2\pi \nu)$', color='b')
        ax1.set_xlabel(r'Frequency $\nu$ (radians)')

        fig.savefig(out_dir + out_name + ext, format="pdf", bbox_inches="tight")

def deterministic_bernoulli_process(periodicity=7, num_periods=20):

    tile_gen = np.zeros(periodicity)
    tile_gen[0] = 1.0
    y = np.tile(tile_gen, num_periods)

    tsplot(y, lags=50, dpi=200, style="fivethirtyeight")

    out_dir = "./plots/ch1_plots/"
    out_name = "deterministic_sample"
    ext = ".pdf"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    plt.savefig(out_dir + out_name + ext, format="pdf", bbox_inches="tight")

def deterministic_bernoulli_spectral(periodicity=7, num_periods=250):

    tile_gen = np.zeros(periodicity)
    tile_gen[0] = 1.0
    y = pd.Series(np.tile(tile_gen, num_periods))

    p = get_periodogram(y, window="none", detrend=False, return_onesided=False, scaling="density", mode="none")

    p = p / len(p.index)

    out_dir = "./plots/ch1_plots/"
    out_name = "deterministic_spectraldensity"
    ext = ".pdf"

    dpi = 220
    style = "fivethirtyeight"

    with plt.style.context(style):   

        fig, ax1 = plt.subplots(figsize=(8,5), dpi=dpi)
        ax1.set_title('Deterministic Bernoulli spectral density')   
        p.plot(ax=ax1)
        ax1.set_ylabel(r'Spectral density $f (2\pi \nu)$', color='b')
        ax1.set_xlabel(r'Frequency $\nu$ (radians)')

        fig.savefig(out_dir + out_name + ext, format="pdf", bbox_inches="tight")

def deterministic_spectral_dist(periodicity=7, num_periods=250):
    tile_gen = np.zeros(periodicity)
    tile_gen[0] = 1.0
    y = pd.Series(np.tile(tile_gen, num_periods))

    p = get_periodogram(y, window="none", detrend=False, return_onesided=False, scaling="density", mode="none")

    p = p / len(p.index)

    out_dir = "./plots/ch1_plots/"
    out_name = "deterministic_spectraldist"
    ext = ".pdf"

    dpi = 220
    style = "fivethirtyeight"

    spectral_dist = pd.Series(np.cumsum(p.values), index=np.fft.fftshift(p.index))

    with plt.style.context(style):   

        fig, ax1 = plt.subplots(figsize=(8,5), dpi=dpi)
        ax1.set_title(r'Deterministic Bernoulli spectral distribution $F(\nu)$')   
        spectral_dist.plot(ax=ax1)
        ax1.set_ylabel(r'Spectral distribution $F(2\pi \nu)$', color='b')
        ax1.set_xlabel(r'Frequency $\nu$ (radians)')

        fig.savefig(out_dir + out_name + ext, format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    ar_params = np.array([-.2])
    ma_params = np.array([.9])

    autoregressive_example(ar_params)
    moving_average_example(ma_params)

    autoregressive_spectral_density(ar_params)
    moving_average_spectral_density(ma_params)

    deterministic_bernoulli_process()
    deterministic_bernoulli_spectral()

    deterministic_spectral_dist()