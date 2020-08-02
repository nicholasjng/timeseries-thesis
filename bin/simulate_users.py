import pandas as pd
import os

import yaml
import argparse

import matplotlib.pyplot as plt 

from regularity_analysis.user_simulations import * 
from regularity_analysis.feature_extraction import *


stat_funcs = {"apen": approximate_entropy, "sampen": sample_entropy, "intrepen": integer_representation_entropy, "permen": permutation_entropy, "pse": classical_periodogram_entropy, "wse": welch_entropy, "lse": lombscargle_entropy, "shannon": shannon_entropy, "shannon_markov": shannon_markov_entropy}

def simulate_users_by_noisefrac(num_users=100, num_fracs=50, ts_length=1000, periodicity=7, success_prob=0.14):

    stats = np.zeros((num_fracs,11))
    sds = np.zeros((num_fracs,11))

    noise_fracs = np.linspace(0,1,num_fracs, endpoint=True)

    for k, frac in enumerate(noise_fracs):
        res = np.zeros((num_users, 11))
        for i in range(num_users):
            user = simulate_gym_user(num_days=ts_length, noise_frac=frac, periodicity=periodicity, success_prob=success_prob)
            user_p = get_periodogram(user)
            user_w = get_welch_periodogram(user)

            res[i, 0] = approximate_entropy(user)
            res[i, 1] = sample_entropy(user)
            res[i, 2] = permutation_entropy(user) 
            res[i, 3] = approximate_entropy(user_p)
            res[i, 4] = sample_entropy(user_p)
            res[i, 5] = permutation_entropy(user_p)
            res[i, 6] = fourier_spectrum_entropy(user_p)
            res[i, 7] = approximate_entropy(user_w)
            res[i, 8] = sample_entropy(user_w)
            res[i, 9] = permutation_entropy(user_w)
            res[i, 10]= fourier_spectrum_entropy(user_w)

        stats[k,:] = np.mean(res, axis=0)
        sds[k,:] = np.std(res, axis=0)
    
    
    dpi = 300
    style = "fivethirtyeight"
    out_dir = "./plots/ch5_plots/"
    out_name = "binary_entropy_metrics"
    f_out_name = "periodogram_entropy_metrics"
    w_out_name = "welch_entropy_metrics"
    ext = ".pdf"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with plt.style.context(style):   

        fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax1.set_title('Simulated Binary Timeseries Statistics')   
        apen_series = pd.Series(stats[:,0], index=noise_fracs)
        apen_sds = pd.Series(sds[:,0], index=noise_fracs)
        sampen_series = pd.Series(stats[:,1], index=noise_fracs)
        sampen_sds = pd.Series(sds[:,1], index=noise_fracs)
        permen_series = pd.Series(stats[:,2], index=noise_fracs)
        permen_sds = pd.Series(sds[:,2], index=noise_fracs)

        apen_series.plot(ax=ax1, label="ApEn(m=2,r=0.2)", yerr=apen_sds, capsize=4)
        sampen_series.plot(ax=ax1, label="SampEn(m=2,r=0.2)", yerr=sampen_sds, capsize=4)
        permen_series.plot(ax=ax1, label="PermEn(n=5,lag=1)", yerr=permen_sds, capsize=4)

        ax1.set_ylabel('Statistical values')
        ax1.set_xlabel(r'Noise level $\varepsilon$')
        ax1.set_ylim(0,1)
        ax1.legend()

        fig1.savefig(out_dir + out_name + ext, format="pdf", bbox_inches="tight")

    with plt.style.context(style):   

        fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax2.set_title('Simulated Binary Timeseries Periodogram Statistics')   
        apen_series = pd.Series(stats[:,3], index=noise_fracs)
        apen_sds = pd.Series(sds[:,3], index=noise_fracs)
        sampen_series = pd.Series(stats[:,4], index=noise_fracs)
        sampen_sds = pd.Series(sds[:,4], index=noise_fracs)
        permen_series = pd.Series(stats[:,5], index=noise_fracs)
        permen_sds = pd.Series(sds[:,5], index=noise_fracs)
        pse_series = pd.Series(stats[:,6], index=noise_fracs)
        pse_sds = pd.Series(sds[:,6], index=noise_fracs)

        apen_series.plot(ax=ax2, label="Periodogram ApEn(m=2,r=0.2)", yerr=apen_sds, capsize=4)
        sampen_series.plot(ax=ax2, label="Periodogram SampEn(m=2,r=0.2)", yerr=sampen_sds, capsize=4)
        permen_series.plot(ax=ax2, label="Periodogram PermEn(n=5,lag=1)", yerr=permen_sds, capsize=4)
        pse_series.plot(ax=ax2, label="Spectral Entropy", yerr=pse_sds, capsize=4)

        ax2.set_ylabel('Statistical values')
        ax2.set_xlabel(r'Noise level $\varepsilon$')
        ax2.set_ylim(0,1)
        ax2.legend()

        fig2.savefig(out_dir + f_out_name + ext, format="pdf", bbox_inches="tight")

    with plt.style.context(style):   

        fig3, ax3 = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax3.set_title('Simulated Binary Timeseries Welch Periodogram Statistics')   
        apen_series = pd.Series(stats[:,7], index=noise_fracs)
        apen_sds = pd.Series(sds[:,7], index=noise_fracs)
        sampen_series = pd.Series(stats[:,8], index=noise_fracs)
        sampen_sds = pd.Series(sds[:,8], index=noise_fracs)
        permen_series = pd.Series(stats[:,9], index=noise_fracs)
        permen_sds = pd.Series(sds[:,9], index=noise_fracs)
        pse_series = pd.Series(stats[:,10], index=noise_fracs)
        pse_sds = pd.Series(sds[:,10], index=noise_fracs)

        apen_series.plot(ax=ax3, label="Welch Periodogram ApEn(m=2,r=0.2)", yerr=apen_sds, capsize=4)
        sampen_series.plot(ax=ax3, label="Welch Periodogram SampEn(m=2,r=0.2)", yerr=sampen_sds, capsize=4)
        permen_series.plot(ax=ax3, label="Welch Periodogram PermEn(n=5,lag=1)", yerr=permen_sds, capsize=4)
        pse_series.plot(ax=ax3, label="Welch Spectral Entropy", yerr=pse_sds, capsize=4)

        ax3.set_ylabel('Statistical values')
        ax3.set_xlabel(r'Noise level $\varepsilon$')
        ax3.set_ylim(0,1)
        ax3.legend()

        fig3.savefig(out_dir + w_out_name + ext, format="pdf", bbox_inches="tight")

def simulate_users_by_length(num_users=1, ts_length=1000, periodicity=7, success_prob=0.14):

    stats = np.zeros((100,11))

    #entirely regular user
    noise_frac = 0.0

    contract_lens = np.linspace(30, 1000, num=100, dtype=int)

    for k, c_len in enumerate(contract_lens):
        res = np.zeros((num_users, 11))
        for i in range(num_users):
            user = simulate_gym_user(num_days=c_len, noise_frac=noise_frac, periodicity=periodicity, success_prob=success_prob)
            user_p = get_periodogram(user)
            user_w = get_welch_periodogram(user)

            res[i, 0] = approximate_entropy(user)
            res[i, 1] = sample_entropy(user)
            res[i, 2] = permutation_entropy(user) 
            res[i, 3] = approximate_entropy(user_p)
            res[i, 4] = sample_entropy(user_p)
            res[i, 5] = permutation_entropy(user_p)
            res[i, 6] = fourier_spectrum_entropy(user_p)
            res[i, 7] = approximate_entropy(user_w)
            res[i, 8] = sample_entropy(user_w)
            res[i, 9] = permutation_entropy(user_w)
            res[i, 10]= fourier_spectrum_entropy(user_w)

        stats[k,:] = np.mean(res, axis=0)
    
    
    dpi = 300
    style = "fivethirtyeight"
    out_dir = "./plots/ch5_plots/"
    out_name = "binary_reg_entropy_metrics"
    f_out_name = "periodogram_reg_entropy_metrics"
    w_out_name = "welch_reg_entropy_metrics"
    ext = ".pdf"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    with plt.style.context(style):   

        fig1, ax1 = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax1.set_title('Simulated Binary Timeseries Statistics')   
        apen_series = pd.Series(stats[:,0], index=contract_lens)
        sampen_series = pd.Series(stats[:,1], index=contract_lens)
        permen_series = pd.Series(stats[:,2], index=contract_lens)

        apen_series.plot(ax=ax1, label="ApEn(m=2,r=0.2)")
        sampen_series.plot(ax=ax1, label="SampEn(m=2,r=0.2)")
        permen_series.plot(ax=ax1, label="PermEn(n=5,lag=1)")

        ax1.set_ylabel('Statistical values')
        ax1.set_xlabel('Contract duration (days)')
        ax1.set_ylim(0,1)
        ax1.legend()

        fig1.savefig(out_dir + out_name + ext, format="pdf", bbox_inches="tight")

    with plt.style.context(style):   

        fig2, ax2 = plt.subplots(figsize=(8, 5), dpi=dpi)
        ax2.set_title('Simulated Binary Timeseries Periodogram Statistics')   
        apen_series = pd.Series(stats[:,3], index=contract_lens)
        sampen_series = pd.Series(stats[:,4], index=contract_lens)
        permen_series = pd.Series(stats[:,5], index=contract_lens)
        pse_series = pd.Series(stats[:,6], index=contract_lens)

        apen_series.plot(ax=ax2, label="Periodogram ApEn(m=2,r=0.2)")
        sampen_series.plot(ax=ax2, label="Periodogram SampEn(m=2,r=0.2)")
        permen_series.plot(ax=ax2, label="Periodogram PermEn(n=5,lag=1)")
        pse_series.plot(ax=ax2, label="Spectral Entropy")

        ax2.set_ylabel('Statistical values')
        ax2.set_xlabel('Contract duration (days)')
        ax2.set_ylim(0,1)
        ax2.legend()

        fig2.savefig(out_dir + f_out_name + ext, format="pdf", bbox_inches="tight")

    with plt.style.context(style):   

        fig3, ax3 = plt.subplots(figsize=(8,5), dpi=dpi)
        ax3.set_title('Simulated Binary Timeseries Welch Periodogram Statistics')   
        apen_series = pd.Series(stats[:,7], index=contract_lens)
        sampen_series = pd.Series(stats[:,8], index=contract_lens)
        permen_series = pd.Series(stats[:,9], index=contract_lens)
        pse_series = pd.Series(stats[:,10], index=contract_lens)

        apen_series.plot(ax=ax3, label="Welch Periodogram ApEn(m=2,r=0.2)")
        sampen_series.plot(ax=ax3, label="Welch Periodogram SampEn(m=2,r=0.2)")
        permen_series.plot(ax=ax3, label="Welch Periodogram PermEn(n=5,lag=1)")
        pse_series.plot(ax=ax3, label="Welch Periodogram Entropy")

        ax3.set_ylabel('Statistical values')
        ax3.set_xlabel('Contract duration (days)')
        ax3.set_ylim(0,1)
        ax3.legend()

        fig3.savefig(out_dir + w_out_name + ext, format="pdf", bbox_inches="tight")

if __name__ == "__main__":
    simulate_users_by_noisefrac()
    simulate_users_by_length()
