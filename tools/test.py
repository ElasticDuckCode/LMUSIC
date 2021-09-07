#!/usr/bin/env python3

import numpy as np
#import numpy.fft as fft
import matplotlib.pyplot as plt
#from tqdm import tqdm
#from scipy.signal import find_peaks

#import torch
#from torch.utils.data import DataLoader

from algorithms import singleSnapshotMUSIC as music
from algorithms import singleSnapshotMUSIC_OffGrid as umusic
from tools import SparseDataset


def test():
    # Perform Monte Carlo Experiment
    NMonteCarlo = 300
    M, N = 64, 512
    K = 5
    npwrs = np.logspace(-2.5, 2.0, 15)
    #npwrs = [0]

    uniform = np.arange(M).reshape(-1, 1)
    #fgrid = fft.fftfreq(N).reshape(-1, 1)
    fgrid = 1/N * np.arange(N).reshape(-1, 1)

    A = np.exp(1j*2*np.pi * uniform @ fgrid.T)
    #A = np.random.randn(M, N)

    dataset_MC = SparseDataset(N, K, NMonteCarlo, 0, A)

    # music
    nmse_losses = []
    hit_rates = []

    for j, npwr in enumerate(npwrs):
        dataset_MC = SparseDataset(N, K, NMonteCarlo, npwr, A)
        #dataset_MC = dataset_MC.regenerateData(npwr, A)
        for k in range(NMonteCarlo):

            # Get data points
            x_t, y = dataset_MC[k]
            #x_t, y = x_t.detach().cpu().numpy(), y.detach().cpu().numpy()

            # Get MUSIC prediction
            x_p, pred_peaks = music(y, A, K)

            # NMSE
            nmse = np.linalg.norm(x_t - x_p)**2 / np.linalg.norm(x_t)**2
            nmse_losses.append(nmse)

            # hit-rate
            true_peaks = np.nonzero((x_t))[0]
            hit_rates.append(np.mean(np.in1d(true_peaks, pred_peaks)))

    # Plot Results
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({"font.sans-serif": ["Helvetica Neue", "sans-serif"]})

    # music
    nmse = np.asarray(nmse_losses).reshape(-1, NMonteCarlo)
    nmse = np.mean(nmse, axis=1)
    nmse = 10*np.log10(nmse)
    hit_rate = np.asarray(hit_rates).reshape(-1, NMonteCarlo)
    hit_rate = np.mean(hit_rate, axis=1)

    # noise powe axis
    noise_power_db = 10*np.log10(npwrs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(noise_power_db, nmse, marker='o', linewidth=2, markersize=8, markerfacecolor="None", label="SS-MUSIC")
    
    ax.set_xlabel('Noise Power/dB', fontsize=18)
    ax.set_ylabel('NMSE/dB', fontsize=18)
    ax.set_title('')
    ax.grid(color='#99AABB', linestyle=':')
    ax.set_facecolor('#E0F0FF')
    #ax.set_yticks(np.geomspace(-10, 10, 5))
    ax.set_xticks(np.floor(np.arange(-25, 20+1, 5)))
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.patch.set_alpha(1)
    plt.savefig("results/nmse.png", transparent=False)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(noise_power_db, hit_rate, marker='o', linewidth=2, markersize=8, markerfacecolor="None", label="SS-MUSIC")
    ax.set_xlabel('Noise Power/dB', fontsize=18)
    ax.set_ylabel('Hit Rate', fontsize=18)
    ax.set_title('')
    ax.grid(color='#99AABB', linestyle=':')
    ax.set_facecolor('#E0F0FF')

    #ax.set_yticks(np.arange(0, 1+0.1, 0.1))
    ax.set_xticks(np.floor(np.arange(-25, 20+1, 5)))

    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.patch.set_alpha(1)
    plt.savefig("results/hit_rate.png", transparent=False)
    #plt.show()
