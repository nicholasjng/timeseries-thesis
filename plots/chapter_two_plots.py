import numpy as np
import pandas as pd
import statsmodels as sm
from statsmodels import tsa
from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt
import os


from utils.plot_utils import tsplot, tsplot_cpd, tsplot_periodogram
from utils.time_utils import get_periodogram, get_welch_periodogram


def create_convolution_plot():
    sig = np.zeros(250)
    sig[20] = 1.0
    sig[100] = 1.0
    sig[110] = 1.0
    sig[180] = 1.0
    win = signal.hann(25)
    filtered = signal.convolve(sig, win, mode='same') / sum(win)

    dpi = 300

    fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True, figsize=(16, 10), dpi=dpi)
    ax_orig.plot(sig)
    ax_orig.set_title('Original delta peak spectrum')
    ax_orig.margins(0, 0.1)
    ax_win.plot(win)
    ax_win.set_title('Window function')
    ax_win.margins(0, 0.1)
    ax_filt.plot(filtered)
    ax_filt.set_title('Windowed signal')
    ax_filt.margins(0, 0.1)
    fig.tight_layout()

    ax_orig.set_ylim(0,1)
    ax_win.set_ylim(0,1)
    ax_filt.set_ylim(0,.2)

    out_dir = "./plots/ch2_plots/"
    out_name = "window_smearing"
    ext = ".png"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    fig.savefig(out_dir + out_name + ext, dpi=dpi)

def create_fft_windows():

    dpi = 200

    fig, ((win1_ax, fft1_ax), (win2_ax, fft2_ax), (win3_ax, fft3_ax)) = plt.subplots(3, 2, figsize=(8, 5), dpi=dpi)
    window = signal.boxcar(51)
    win1_ax.plot(window)
    win1_ax.set_title("Rectangular window")
    #win1_ax.set_ylabel("Amplitude")
    #win1_ax.set_xlabel("Rectangular window")
    win1_ax.set_xlim(0,52)
    win1_ax.set_ylim(0, 2)
    win1_ax.axes.xaxis.set_ticks([])

    A = fft(window, 2048) / (len(window)/2.0)
    freq = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    fft1_ax.plot(freq, response)
    fft1_ax.set_xlim(-0.5, 0.5)
    fft1_ax.set_ylim(-120, 0)
    fft1_ax.set_title("Rectangular Spectral window")
    #fft1_ax.set_ylabel("Magnitude [dB]")
    #fft1_ax.set_xlabel("Frequency [rad]")
    fft1_ax.yaxis.tick_right()
    fft1_ax.axes.xaxis.set_ticks([])
    
    window = signal.hann(51)
    win2_ax.plot(window)
    win2_ax.set_title("Hann window")
    win2_ax.set_ylabel("Amplitude")
    #win2_ax.set_xlabel("Hann window")
    win2_ax.axes.xaxis.set_ticks([])

    A = fft(window, 2048) / (len(window)/2.0)
    freq = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    fft2_ax.plot(freq, response)
    fft2_ax.set_xlim(-0.5, 0.5)
    fft2_ax.set_ylim(-120, 0)
    fft2_ax.set_title("Hann Spectral window")
    fft2_ax.set_ylabel("Magnitude [dB]")
    #fft2_ax.set_xlabel("Frequency [rad]")
    fft2_ax.yaxis.tick_right()
    fft2_ax.axes.xaxis.set_ticks([])

    window = signal.bartlett(51)
    win3_ax.plot(window)
    win3_ax.set_title("Bartlett window")
    #win3_ax.set_ylabel("Amplitude")
    win3_ax.set_xlabel("Window width")

    A = fft(window, 2048) / (len(window)/2.0)
    freq = np.linspace(-0.5, 0.5, len(A))
    response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
    fft3_ax.plot(freq, response)
    fft3_ax.set_xlim(-0.5, 0.5)
    fft3_ax.set_ylim(-120, 0)
    fft3_ax.set_title("Bartlett Spectral window")
    #fft3_ax.set_ylabel("Magnitude [dB]")
    fft3_ax.set_xlabel("Frequency [rad]")
    fft3_ax.yaxis.tick_right()

    out_dir = "./plots/ch2_plots/"
    out_name = "fft_wins"
    ext = ".png"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    fig.savefig(out_dir + out_name + ext, dpi=dpi)


if __name__ == "__main__":
    create_convolution_plot()

    create_fft_windows()