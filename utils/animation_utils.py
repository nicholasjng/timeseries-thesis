import matplotlib.pyplot as plt 
from matplotlib import animation
import matplotlib.gridspec as gridspec

from regularity_analysis.feature_extraction import approximate_entropy, sample_entropy, permutation_entropy, fourier_spectrum_entropy
from regularity_analysis.user_simulations import simulate_user_with_mixture, simulate_user_with_hmm
from utils.plot_utils import plot_user_histogram_with_density
from utils.time_utils import get_periodogram, get_welch_periodogram, binary_user_from_td

import numpy as np 
import pandas as pd 

func_dispatch = {"apen": approximate_entropy, "sampen": sample_entropy, "entropy": fourier_spectrum_entropy}

def single_entropy_animation(timeseries, animation_config, ms=[2,3], rs=[0.01,3.0], entropy_type="ApEn"):
    """
    Small function for animating ApEn or SampEn as a function of the noise level r.
    Arguments:
    timeseries: pd.Series, holds values for simulation
    animation_config: dict, contains config for the animation
    ms: list, list of m parameters to use for ApEn and SampEn calculation
    rs: list, interval of rs to compute the ApEns and SampEns over 
    entropy_type: str, type of entropy to use
    """
    global func_dispatch 

    # set up figure and animation
    fig = plt.figure(figsize=(16,10))

    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,:], autoscale_on=animation_config["autoscale_on"])

    lines = []

    plotcols = ["red", "orange", "green", "blue"]
    frames = animation_config["frames"]
    interval = animation_config["interval"]
    blit = animation_config["blit"]

    key = entropy_type.lower()

    #clutter code to process line objects
    user_lobj = ax1.plot([],[],"o-", lw=0.5)[0]
    lines.append(user_lobj)
    user_fft_lobj = ax2.plot([],[],"o-", lw=0.5)[0]
    lines.append(user_fft_lobj)

    periodogram = get_periodogram(timeseries)

    for m in ms:
        #initialize the plot objects sequentially
        lobj = ax3.plot([],[],"o-", lw=2, ms=0.5, label=entropy_type + " m = {}".format(m))[0]
        lines.append(lobj) 
    
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.title.set_text("Timeseries")
    ax2.title.set_text("FFT Power Spectrum")
    ax3.legend()

    texts = []
    for i,m in enumerate(ms):
        text = ax3.text(0.05, 0.95 - 0.07*i, '', transform=ax3.transAxes)
        texts.append(text)

    running = np.linspace(rs[0], rs[1], num=frames, endpoint=True)

    reg_data = np.zeros((frames, len(ms)))

    for i, m in enumerate(ms):
        reg_data[:,i] = np.array([func_dispatch[key](timeseries, m=m, r=r) for r in running])

    def init():
        """initialize animation"""
        for line in lines:
            line.set_data([],[])
        
        timeseries.plot(ax=ax1, color="blue")
        periodogram.plot(ax=ax2, color="blue")

        for i, text in enumerate(texts):
            text.set_text(entropy_type + ": m = {}, r = {:0.2f}".format(ms[i], running[0]))

        ax3.set_xlabel("r")
        ax3.set_ylabel(entropy_type + "(m,r)")
        ax3.set_xlim(running[0], running[-1])
        ax3.set_ylim(0.0, np.amax(reg_data)+0.1)
        return lines, texts

    def animate(i):
        """perform animation step"""
        
        for k, text in enumerate(texts):
            text.set_text(entropy_type + ": m = {}, r = {:0.2f}".format(ms[k], running[i]))
        
        for k, line in enumerate(lines[2:]):
            line.set_data(running[:i], reg_data[:i, k])

        return lines, texts
    
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=blit, init_func=init)

    if animation_config["show"]:
        plt.show()

    return ani

def mixed_entropy_animation(timeseries, animation_config, ms=[2,3], rs=[0.01,3.0]):
    """
    Small function for animating ApEn and SampEn as a function of the noise level r.
    Arguments:
    timeseries: pd.Series, holds values for simulation
    animation_config: dict, contains config for the animation
    ms: list, list of m parameters to use for ApEn and SampEn calculation
    rs: list, interval of rs to compute the ApEns and SampEns over 
    """

    # set up figure and animation
    fig = plt.figure(figsize=(16,10))

    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,:], autoscale_on=animation_config["autoscale_on"]) #xlim=animation_config["xlim"], ylim=animation_config["ylim"]

    lines = []

    plotcols = ["red", "orange", "green", "blue"]
    frames = animation_config["frames"]
    interval = animation_config["interval"]
    blit = animation_config["blit"]
    entropy_names = ["ApEn", "SampEn"]

    #clutter code to process line objects
    user_lobj = ax1.plot([],[],"o-", lw=0.5)[0]
    lines.append(user_lobj)
    user_fft_lobj = ax2.plot([],[],"o-", lw=0.5)[0]
    lines.append(user_fft_lobj)

    #stack same list to make iterating easier
    stacked_ms = ms + ms 
    periodogram = get_periodogram(timeseries)

    for i,m in enumerate(stacked_ms):
        #initialize the plot objects sequentially
        lobj = ax3.plot([],[],"o-", lw=2, ms=0.5, label="Timeseries " + entropy_names[i // 2] + " m = {}".format(m))[0]
        lines.append(lobj) 
    
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.title.set_text("Timeseries")
    ax2.title.set_text("FFT Power Spectrum")
    ax3.legend()

    texts = []
    for i,m in enumerate(stacked_ms):
        text = ax3.text(0.06, 0.95 - 0.07*i, '', transform=ax3.transAxes)
        texts.append(text)

    running = np.linspace(rs[0], rs[1], num=frames, endpoint=True)

    reg_data = np.zeros((frames, 2*len(ms)))

    for i, m in enumerate(ms):
            reg_data[:,i] = np.array([approximate_entropy(timeseries, m=m, r=r) for r in running])
            reg_data[:,i + len(ms)] = np.array([sample_entropy(timeseries, m=m, r=r) for r in running])

    ax3.set_xlim(0.0, running[-1])
    ax3.set_ylim(0.0, 2.5)

    def init():
        """initialize animation"""
        for line in lines:
            line.set_data([],[])
        
        timeseries.plot(ax=ax1, color="blue")
        periodogram.plot(ax=ax2, color="blue")

        for i, text in enumerate(texts):
            text.set_text(entropy_names[i // 2] + ": m = {}, r = {:0.2f}".format(stacked_ms[i], running[0]))

        ax3.set_xlabel("r")
        ax3.set_ylabel("ApEn / SampEn (m,r)")
        return lines, texts

    def animate(i):
        """perform animation step"""
        for k, text in enumerate(texts):
            text.set_text(entropy_names[k // 2] + ": m = {}, r = {:0.2f}".format(stacked_ms[k], running[i]))
        
        for k, line in enumerate(lines[2:]):
            line.set_data((running[:i], reg_data[:i, k]))

        return line, texts
    
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=blit, init_func=init)

    if animation_config["show"]:
        plt.show()

    return ani

def single_user_sim_animation(animation_config, means=[5.0], sigmas=(0.001,3.0), weights=[1.0], entropy_type="ApEn", m=2, r=0.2):
    """
    Small function for animating a single Fourier spectrum entropy as a function of sigma in a user simulation.
    Arguments:
    animation_config: dict, contains config for the plot   
    mean: Mean of the Gaussian for timedelta simulation
    sigmas: tuple, range of sigmas to use in computation
    entropy_type: str, type of entropy to compute
    m: m used for ApEn and SampEn
    r: r used for ApEn and SampEn 
    """
    means = np.array(means)
    weights = np.array(weights)

    # set up figure and animation
    fig = plt.figure(figsize=(16,10))

    gs = fig.add_gridspec(2,3)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    ax4 = fig.add_subplot(gs[1,:], autoscale_on=animation_config["autoscale_on"])

    lines = []

    key = entropy_type.lower()

    cmap = plt.get_cmap("tab20", 2)
    frames = animation_config["frames"]
    interval = animation_config["interval"]
    blit = animation_config["blit"]
    ts_entropy_names = {"apen": "Time Series ApEn(m = {}, r = {})".format(m,r), "sampen": "Time Series SampEn (m = {}, r = {})".format(m,r)}
    entropy_names = {"apen": "Power Spectrum ApEn(m = {}, r = {})".format(m,r), "sampen": "Power Spectrum SampEn (m = {}, r = {})".format(m,r), "entropy": "Power Spectrum Entropy"}
    kwds = {"apen": {"m": m, "r": r}, "sampen": {"m": m, "r": r}, "entropy": {"normalize": True}} 
    #clutter code to process line objects
    user_lobj = ax1.plot([],[],"o-", lw=0.5)[0]
    lines.append(user_lobj)
    density_lobj = ax2.plot([],[],"o-", lw=0.5)[0]
    lines.append(density_lobj)
    user_fft_lobj = ax3.plot([],[],"o-", lw=0.5)[0]
    lines.append(user_fft_lobj)

    for i in range(2):
        lobj = ax4.plot([],[],"o-", lw=2, ms=0.5, color=cmap(i), label=entropy_names[key])[0]
        lines.append(lobj) 
    
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax1.title.set_text("Timeseries")
    ax2.title.set_text("Density Histogram")
    ax3.title.set_text("FFT Power Spectrum")
    ax4.legend()

    text = ax4.text(0.12, 0.95 - 0.07, '', transform=ax3.transAxes)

    running = np.linspace(sigmas[0], sigmas[1], num=frames, endpoint=True)

    reg_data = np.zeros((frames, 2))

    def init():
        """initialize animation"""
        for line in lines:
            line.set_data([],[])

        text.set_text(entropy_names[key])

        ax4.set_xlabel("sigma^2")
        ax4.set_ylabel("Entropies")
        ax4.set_xlim(running[0], running[-1])
        ax4.set_ylim(0.0, 1.5)
        return line, text

    def animate(i):
        """perform animation step"""
        global func_dispatch

        ax1.clear()
        ax2.clear()
        ax3.clear()

        variances = np.ones(len(means)) * running[i]

        timeseries = simulate_user_with_mixture("01-01-2016", means=means, variances=variances, weights=weights, num_samples=100)
        bin_series = binary_user_from_td(timeseries)
        periodogram = get_periodogram(bin_series)
        bin_series.plot(ax=ax1, color="blue")
        plot_user_histogram_with_density(timeseries, means=means, variances=variances, weights=weights, ax=ax2)
        periodogram.plot(ax=ax3, color="blue")

        reg_data[i,0] = func_dispatch[key](periodogram, **kwds[key])
        if key != "entropy":
            reg_data[i,1] = func_dispatch[key](timeseries, **kwds[key])
         
        for k, line in enumerate(lines[3:]):
            line.set_data((running[:i], reg_data[:i,k]))

        return line, text
    
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=blit, init_func=init)

    if animation_config["show"]:
        plt.show()

    return ani

def mixed_user_sim_animation(animation_config, means=[5.0], sigmas=(0.001,3.0), weights=[1.0], mixture_type="gaussian", m=2, r=0.2):
    """
    Small function for animating all Fourier spectrum entropies as a function of sigma in a user simulation.
    Arguments:
    animation_config: dict, contains config for the plot   
    mean: Mean of the Gaussian for user simulation
    sigmas: tuple, range of sigmas to use in computation
    m: m used for ApEn and SampEn
    r: r used for ApEn and SampEn 
    """
    means = np.array(means)
    weights = np.array(weights)

    # set up figure and animation
    fig = plt.figure(figsize=(16,10))

    gs = fig.add_gridspec(2,3)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    ax4 = fig.add_subplot(gs[1,:], autoscale_on=animation_config["autoscale_on"]) #aspect="equal"

    lines = []

    plotcols = ["red", "green", "blue"]
    frames = animation_config["frames"]
    interval = animation_config["interval"]
    blit = animation_config["blit"]
    entropy_names = ["Power Spectrum ApEn(m = {}, r = {})".format(m,r), "Power Spectrum SampEn (m = {}, r = {})".format(m,r), "Power Spectrum Entropy"]

    #clutter code to process line objects
    user_lobj = ax1.plot([],[],"o-", lw=0.5)[0]
    lines.append(user_lobj)
    density_lobj = ax2.plot([],[],"o-", lw=0.5)[0]
    lines.append(density_lobj)
    user_fft_lobj = ax3.plot([],[],"o-", lw=0.5)[0]
    lines.append(user_fft_lobj)

    for i in range(3):
        #initialize the plot objects sequentially
        lobj = ax4.plot([],[],"o-", lw=2, ms=0.5, color=plotcols[i], label=entropy_names[i])[0]
        lines.append(lobj) 
    
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax1.title.set_text("Timeseries")
    ax2.title.set_text("Density Histogram")
    ax3.title.set_text("FFT Power Spectrum")
    ax4.legend()

    texts = []
    for i in range(3):
        text = ax3.text(0.12, 0.95 - 0.07*i, '', transform=ax3.transAxes)
        texts.append(text)

    running = np.linspace(sigmas[0], sigmas[1], num=frames, endpoint=True)

    reg_data = np.zeros((frames, 3))

    def init():
        """initialize animation"""
        for line in lines:
            line.set_data([],[])

        for i, text in enumerate(texts):
            text.set_text(entropy_names[i])

        ax4.set_xlabel("sigma^2")
        ax4.set_ylabel("Entropies")
        ax4.set_xlim(running[0], running[-1])
        ax4.set_ylim(0.0, 1.5)
        return lines, texts

    def animate(i):
        """perform animation step"""
        #clear axes to avoid replotting
        ax1.clear()
        ax2.clear()
        ax3.clear()

        variances = np.ones(len(means)) * running[i]

        timeseries = simulate_user_with_mixture("01-01-2016", means=means, variances=variances, weights=weights, mixture_type=mixture_type, num_samples=100)
        bin_series = binary_user_from_td(timeseries)
        periodogram = get_periodogram(bin_series)
        bin_series.plot(ax=ax1, color="blue")
        plot_user_histogram_with_density(timeseries, means=means, variances=variances, weights=weights, mixture_type=mixture_type, ax=ax2)
        periodogram.plot(ax=ax3, color="blue")

        reg_data[i,0] = approximate_entropy(periodogram, m=m, r=r)
        reg_data[i,1] = sample_entropy(periodogram, m=m, r=r)
        reg_data[i,2] = fourier_spectrum_entropy(periodogram, normalize=True)
         
        for k, line in enumerate(lines[3:]):
            line.set_data((running[:i], reg_data[:i, k]))

        return line, texts
    
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=blit, init_func=init)

    if animation_config["show"]:
        plt.show()

    return ani

def fourier_spectrum_window_test(animation_config, means=[3.0,7.0], sigmas=[0.001,3.0], weights=[0.5,0.5], windows=["boxcar", "hann", "parzen"], entropy_type="ps"):
    """
    Small function for animating the Fourier spectrum entropy as a function of sigma in a user simulation, with different window functions.
    Arguments:
    animation_config: dict, contains config for the plot   
    mean: Means of the Gaussians/Gammas for user simulation
    sigmas: tuple, range of sigmas to use in animation
    m: m used for ApEn and SampEn
    r: r used for ApEn and SampEn 
    """
    means = np.array(means)
    weights = np.array(weights)

    # set up figure and animation
    fig = plt.figure(figsize=(16,10))

    gs = fig.add_gridspec(2,3)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[0,2])
    ax4 = fig.add_subplot(gs[1,:], autoscale_on=animation_config["autoscale_on"])

    lines = []

    plotcols = ["red", "green", "blue"]
    frames = animation_config["frames"]
    interval = animation_config["interval"]
    blit = animation_config["blit"]

    entropy_names = ["Power Spectrum Entropy, {} window ".format(window.capitalize()) for window in windows]

    #clutter code to process line objects
    user_lobj = ax1.plot([],[],"o-", lw=0.5)[0]
    lines.append(user_lobj)
    density_lobj = ax2.plot([],[],"o-", lw=0.5)[0]
    lines.append(density_lobj)
    user_fft_lobj = ax3.plot([],[],"o-", lw=0.5)[0]
    lines.append(user_fft_lobj)

    num_windows = len(windows)

    for i in range(num_windows):
        #initialize the plot objects sequentially
        lobj = ax4.plot([],[],"o-", lw=2, ms=0.5, color=plotcols[i], label=entropy_names[i])[0]
        lines.append(lobj) 
    
    running = np.linspace(sigmas[0], sigmas[1], num=frames, endpoint=True)
    reg_data = np.zeros((frames, num_windows))
    
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()
    ax1.title.set_text("Timeseries")
    ax2.title.set_text("Density Histogram")
    ax3.title.set_text("FFT Power Spectrum")
    ax4.legend()
    ax4.set_xlim(0.0, running[-1])
    ax4.set_ylim(0.0, 10.0)

    def init():
        """initialize animation"""
        for line in lines:
            line.set_data([],[])

        ax4.set_xlabel("sigma^2")
        ax4.set_ylabel("Entropies")
        return lines

    def animate(i):
        """perform animation step"""
        ax1.clear()
        ax2.clear()
        ax3.clear()

        variances = np.ones(len(means)) * running[i]
        timeseries = simulate_user_with_mixture("01-01-2016", means=means, variances=variances, weights=weights, num_samples=100)
        bin_series = binary_user_from_td(timeseries)
        periodogram = get_periodogram(bin_series)
        bin_series.plot(ax=ax1, color="blue")
        plot_user_histogram_with_density(timeseries, means=means, variances=variances, weights=weights, ax=ax2)
        periodogram.plot(ax=ax3, color="blue")

        for k, window in enumerate(windows):
            reg_data[i,k] = fourier_spectrum_entropy(get_periodogram(bin_series, window=window), normalize=False)
         
        for k, line in enumerate(lines[3:]):
            line.set_data((running[:i], reg_data[:i, k]))

        return lines
    
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=blit, init_func=init)

    if animation_config["show"]:
        plt.show()

    return ani

def fourier_spectrum_duration_test(animation_config, means=[3.0,7.0], sigmas=[0.01, 0.01], weights=[0.5, 0.5], windows=["boxcar", "hann", "parzen"], num_samples=[1, 500]):
    """
    Small function for animating the Fourier spectrum entropy as a function of contract duration N.
    Arguments:
    animation_config: dict, contains config for the plot   
    mean: Means of the Gaussians/Gammas for user simulation
    sigmas: tuple, range of sigmas to use in animation
    m: m used for ApEn and SampEn
    r: r used for ApEn and SampEn 
    """
    means = np.array(means)
    sigmas = np.array(sigmas)
    weights = np.array(weights)

    # set up figure and animation
    fig = plt.figure(figsize=(16,10))

    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,:], autoscale_on=animation_config["autoscale_on"])

    lines = []
    
    plotcols = ["red", "green", "blue"]
    frames = num_samples[-1]
    interval = animation_config["interval"]
    blit = animation_config["blit"]
    entropy_names = ["Power Spectrum Entropy, {} window".format(window.capitalize()) for window in windows]
    welch_entropy_names = ["Welch Power Spectrum Entropy, {} window".format(window.capitalize()) for window in windows]

    names = entropy_names + welch_entropy_names

    cmap = plt.get_cmap("tab20", len(names))
    #clutter code to process line objects
    user_lobj = ax1.plot([],[],"o-", lw=0.5)[0]
    lines.append(user_lobj)
    user_fft_lobj = ax2.plot([],[],"o-", lw=0.5)[0]
    lines.append(user_fft_lobj)

    num_windows = len(windows)

    for i in range(len(names)):
        #initialize the plot objects sequentially
        lobj = ax3.plot([],[],"o-", lw=2, ms=0.5, color=cmap(i), label=names[i])[0]
        lines.append(lobj) 

    #reset the variances every few frames
    running = np.arange(num_samples[0], num_samples[-1])

    reg_data = np.zeros((frames, 2 * num_windows))
    
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.title.set_text("Timeseries")
    ax2.title.set_text("Periodogram")
    ax3.legend()
    ax3.set_xlim(running[0], running[-1])
    ax3.set_ylim(0.0, 1.0) #Fourier Spectrum Entropy is normalized

    def init():
        """initialize animation"""
        for line in lines:
            line.set_data([],[])

        ax3.set_xlabel("Sample Size N")
        ax3.set_ylabel("Entropies")
        return lines

    def animate(i):
        """perform animation step"""
        ax2.cla()
        ax2.grid()
        ax2.title.set_text("Periodogram")
        
        visits = np.array([0.0, 0.0, 1.0])#, 0.0, 0.0, 1.0, 0.0])

        vals = np.tile(visits, i+1)
        timeseries = pd.Series(vals)

        timeseries.plot(ax=ax1, color="blue")

        periodogram = get_periodogram(timeseries)
        welch_periodogram = get_welch_periodogram(timeseries, window="flattop")
        periodogram.plot(ax=ax2, color="green")
        welch_periodogram.plot(ax=ax2, color="orange")
        for k, window in enumerate(windows):
            reg_data[i,k] = fourier_spectrum_entropy(get_periodogram(timeseries, window=window), normalize=True)
            reg_data[i,k+num_windows] = fourier_spectrum_entropy(get_welch_periodogram(timeseries, window=window, nfft=512, nperseg=365), normalize=True)           

        for k, line in enumerate(lines[2:]):
            line.set_data((running[:i], reg_data[:i, k]))

        return lines
    
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=100, blit=blit, init_func=init)

    if animation_config["show"]:
        plt.show()

    return ani

def fourier_spectrum_winsize_test(timeseries, animation_config, windows=["boxcar", "hann", "parzen"], window_sizes=[1, 250]):
    """
    Small function for animating the Fourier spectrum entropy as a function of sigma in a user simulation, with different window sizes.
    Arguments:
    animation_config: dict, contains config for the plot   
    mean: Means of the Gaussians/Gammas for user simulation
    sigmas: tuple, range of sigmas to use in animation
    m: m used for ApEn and SampEn
    r: r used for ApEn and SampEn 
    """

    # set up figure and animation
    fig = plt.figure(figsize=(16,10))

    gs = fig.add_gridspec(2,2)
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,:], autoscale_on=animation_config["autoscale_on"])

    lines = []

    plotcols = ["red", "green", "blue"]
    frames = window_sizes[-1]
    interval = animation_config["interval"]
    blit = animation_config["blit"]
    entropy_names = ["Power Spectrum Entropy, {} window".format(window.capitalize()) for window in windows]

    #clutter code to process line objects
    user_lobj = ax1.plot([],[],"o-", lw=0.5)[0]
    lines.append(user_lobj)
    user_fft_lobj = ax2.plot([],[],"o-", lw=0.5)[0]
    lines.append(user_fft_lobj)

    num_windows = len(windows)

    for i in range(num_windows):
        #initialize the plot objects sequentially
        lobj = ax3.plot([],[],"o-", lw=2, ms=0.5, color=plotcols[i], label=entropy_names[i])[0]
        lines.append(lobj) 
    
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.title.set_text("Timeseries")
    ax2.title.set_text("Periodogram")
    ax3.legend()

    #reset the variances every few frames
    running = np.arange(window_sizes[0], window_sizes[-1] + 1)

    reg_data = np.zeros((frames, num_windows))

    for k, window in enumerate(windows):
        reg_data[:,k] = np.array([fourier_spectrum_entropy(get_periodogram(timeseries, window=window, window_size=n), normalize=True) for n in running])
    
    def init():
        """initialize animation"""
        for line in lines:
            line.set_data([],[])

        #calculate periodogram for some default values
        timeseries.plot(ax=ax1, color="blue")

        ax3.set_xlabel("Window Size N")
        ax3.set_ylabel("Entropies")
        ax3.set_xlim(running[0], running[-1])
        ax3.set_ylim(0.0, 1.0) #Fourier Spectrum Entropy is normalized
        return lines

    def animate(i):
        """perform animation step"""
        ax2.cla()
        ax2.grid()
        ax2.title.set_text("Periodogram")

        for k, line in enumerate(lines[2:]):
            line.set_data((running[:i], reg_data[:i, k]))

        for k,window in enumerate(windows):
            get_periodogram(timeseries, window=window, window_size=i+1).plot(ax=ax2, label = entropy_names[k])

        ax3.set_ylim(0.0, 1.1 * np.amax(reg_data)) #Fourier entropy is normalized to 1

        ax2.legend()

        return lines
    
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=100, blit=blit, init_func=init)

    if animation_config["show"]:
        plt.show()

    return ani

def entropy_longterm_behavior(N, animation_config, ms=[2,3], rs=[0.01,3.0], entropy_type="ApEn"):
    """
    Small function for animating ApEn or SampEn as a function of the noise level r.
    Arguments:
    timeseries: pd.Series, holds values for simulation
    animation_config: dict, contains config for the animation
    ms: list, list of m parameters to use for ApEn and SampEn calculation
    rs: list, interval of rs to compute the ApEns and SampEns over 
    entropy_type: str, type of entropy to use
    """
    global func_dispatch 

    # set up figure and animation
    fig = plt.figure(figsize=(16,10))
    ax = fig.add_subplot(1, 1, 1)

    lines = []

    frames = N
    interval = animation_config["interval"]
    blit = animation_config["blit"]

    key = entropy_type.lower()

    #clutter code to process line objects
    user_lobj = ax.plot([],[],"o-", lw=0.5)[0]
    lines.append(user_lobj)

    for m in ms:
        #initialize the plot objects sequentially
        lobj = ax.plot([],[],"o-", lw=2, ms=0.5, label=entropy_type + " m = {}".format(m))[0]
        lines.append(lobj) 
    
    ax.grid()
    ax.title.set_text("Timeseries")
    ax.legend()

    texts = []
    for i,m in enumerate(ms):
        text = ax.text(0.05, 0.95 - 0.07*i, '', transform=ax.transAxes)
        texts.append(text)

    running = np.linspace(rs[0], rs[1], num=frames, endpoint=True)

    reg_data = np.zeros((frames, len(ms)))

    for i, m in enumerate(ms):
        timeseries = pd.Series(0, index=np.arange(i))
        timeseries.iloc[0] = 1.
        reg_data[:,i] = np.array([func_dispatch[key](timeseries, m=m, r=r) for r in running])

    def init():
        """initialize animation"""
        for line in lines:
            line.set_data([],[])

        for i, text in enumerate(texts):
            text.set_text(entropy_type + ": m = {}, r = {:0.2f}".format(ms[i], running[0]))

        ax.set_xlabel("r")
        ax.set_ylabel(entropy_type + "(m,r)")
        ax.set_xlim(running[0], running[-1])
        ax.set_ylim(0.0, np.amax(reg_data)+0.1)
        return lines, texts

    def animate(i):
        """perform animation step"""
        
        for k, text in enumerate(texts):
            text.set_text(entropy_type + ": m = {}, r = {:0.2f}".format(ms[k], running[i]))
        
        for k, line in enumerate(lines[2:]):
            line.set_data(running[:i], reg_data[:i, k])

        return lines, texts
    
    ani = animation.FuncAnimation(fig, animate, frames=frames, interval=interval, blit=blit, init_func=init)

    if animation_config["show"]:
        plt.show()

    return ani