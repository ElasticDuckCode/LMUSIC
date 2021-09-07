#!/usr/bin/env python3

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import trange, tqdm

from network import LMUSIC, device
from tools import SparseDataset, SimpleDataset
from torch.utils.data import DataLoader

def train(verbose=True):

    # signal parameters
    M, N = 64, 512
    K = 5
    npwr = 2

    # training parameters
    training_points = 1_000
    testing_points = 100
    batch_size = 100
    epochs = 50

    uniform = np.arange(M).reshape(-1, 1)
    fgrid = 1/N * np.arange(N).reshape(-1, 1)
    A = np.exp(1j*2*np.pi * uniform @ fgrid.T)

    model = LMUSIC(signal_dim=N, inner_dim=2*N, n_filters=16, n_layers=5)
    model = model.to(device)

    if verbose: print(model, "\n")

    # preprocess data to get MUSIC solution
    dataset = SparseDataset(N, K, training_points, npwr=npwr, A=A)
    dataset_t = SparseDataset(N, K, testing_points, npwr=npwr, A=A)
    truth, data = dataset[:]
    truth_t, data_t = dataset_t[:]
    del dataset # don't need extra copy
    del dataset_t

    if verbose: print("Preparing Dataset:")
    if verbose: print("\tHankelizing data...", end="")
    hankel_data = model.batch_hankel(data)
    hankel_data_t = model.batch_hankel(data_t)
    if verbose: print("done.")

    if verbose: print("\tGetting MUSIC pseudospectrum...", end="")
    spectrum_data = model.batch_music_spectrum(hankel_data, A, K)
    spectrum_data_t = model.batch_music_spectrum(hankel_data_t, A, K)
    if verbose: print("done.")

    if verbose: print("\tFinding peaks...", end="")
    peak_data = model.batch_find_peaks(spectrum_data, K)
    peak_data_t = model.batch_find_peaks(spectrum_data_t, K)
    if verbose: print("done.")

    if verbose: print("\tUsing peaks to get Least Squares solution...", end="")
    music_data = model.batch_support_least_squares(data, peak_data, A)
    music_data_t = model.batch_support_least_squares(data_t, peak_data_t, A)
    if verbose: print("done.")

    # train network to correct MUSIC solutions (for now assuming amplitudes real)
    if verbose: print("Beginning Training:")
    music_data = music_data.real.to(torch.float32).to(device)
    music_data_t = music_data_t.real.to(torch.float32).to(device)
    truth = truth.to(torch.float32).to(device)
    truth_t = truth_t.to(torch.float32).to(device)
    dataset = SimpleDataset(truth, music_data)
    dataset_t = SimpleDataset(truth_t, music_data_t)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    dataloader_t = DataLoader(dataset_t, batch_size=testing_points, shuffle=True)
    n_batches = int(training_points/batch_size)

    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    l = torch.nn.MSELoss()

    t_loop = tqdm(range(epochs), file=sys.stdout)
    training_losslist = np.zeros(epochs * n_batches)
    testing_losslist = np.zeros(epochs * n_batches)

    #results = model(music_data)

    min_loss = np.inf
    for e in t_loop:
        for i, data in enumerate(dataloader):
            idx = e * n_batches + i

            x_t, x_p = data
            correction = model(x_p)

            loss = l(x_t, x_p + correction)

            with torch.no_grad():
                training_losslist[idx] = loss

            loss.backward()
            optim.step()
            optim.zero_grad()

            for j, test_data in enumerate(dataloader_t):
                x_t, x_p = test_data
                correction = model(x_p)
                testing_losslist[idx] += torch.mean((x_t.cpu() - x_p.cpu() -  correction.cpu())**2)

                if testing_losslist[idx] < min_loss:
                    torch.save(model.state_dict(), "weights.pt")
                    min_loss = testing_losslist[idx]

            t_loop.set_description("Batch: {}/{}, Training Loss: {}, Validation Loss: {}".format(i, len(dataloader), training_losslist[idx], testing_losslist[idx]), refresh=True)

    np.save("results/train_loss.npy", np.asarray(training_losslist))
    np.save("results/test_loss.npy", np.asarray(testing_losslist))

    fig, ax = plt.subplots()
    ax.plot(training_losslist, label="Training Loss")
    ax.plot(testing_losslist, label="Testing Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig("results/losses.png")

    #test_truth = dataset[0][0]
    #music_pred = dataset[0][1][None]

    #model.eval()
    #network_correction = model(music_pred)
    #network_pred = music_pred - network_correction

    #test_truth = test_truth.detach().cpu().numpy()
    #music_pred = music_pred.detach().cpu().numpy().ravel()
    #network_pred = network_pred.detach().cpu().numpy().ravel()
    pass
