#!/usr/bin/env python3

import numpy as np
from scipy.linalg import hankel
from scipy.signal import find_peaks
#from matplotlib import pyplot as plt

import torch
from torch.nn import Module, Parameter, Linear, Conv1d, BatchNorm1d, ReLU, Sequential
from torch.nn.functional import normalize

#from algorithms import singleSnapshotMUSIC as MUSIC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LMUSIC(Module):
    def __init__(self, signal_dim, inner_dim, n_filters, n_layers, kernel_size=3):
        super(LMUSIC, self).__init__()

        # save arguements in object
        self.n_filters = n_filters
        self.n_layers = n_layers
        self.signal_dim = signal_dim
        self.inner_dim = inner_dim

        # build network layers
        self.first_layer = Linear(signal_dim, inner_dim * n_filters, bias=False)
        self.hidden_layers = []
        for i in range(n_layers - 2):
            self.hidden_layers.append(
                Sequential(
                    Conv1d(n_filters, n_filters, kernel_size=kernel_size, padding='same', bias=False),
                    BatchNorm1d(n_filters), 
                    ReLU()
                )
            )
        self.hidden_layers = Sequential(*self.hidden_layers)
        self.outer_layer = Linear(inner_dim * n_filters, signal_dim, bias=False)

    def forward(self, data):
        '''
        Assuming that input data is result of Single-Snapshot MUSIC.
        To get SS-MUSIC results from data, run 
        -   self.batch_hankel
        -   self.batch_music, 
        -   self.batch_music_pseudospectrum
        -   self.batch_find_peaks
        -   self.batch_support_least_squares
        '''

        # send through layers of network
        batch_size, signal_dim = data.shape
        #print(data.shape)
        output = self.first_layer(data).view(batch_size, self.n_filters, -1)
        for layer in self.hidden_layers:
            output = layer(output)
        output = output.view(batch_size, -1)
        output = self.outer_layer(output)
        #print(output.shape)


        return output # should be same size as input

    def batch_hankel(self, data):
        '''
        Convert batch of data into hankel matricies.

        Assuming data.shape = [batch_size, num_measurements]

        Returns new_data.shape = [batch_size, L+1, M-L-1]
        where M = num_measurements, and L is the floored half of M.
        '''

        # need to convert to numpy for hankelization
        Y = data.detach().cpu().numpy()

        # convert to hankel matricies
        batch_num, M = Y.shape
        L = M // 2
        new_data = np.zeros([batch_num, L+1, M-L], dtype=Y.dtype)
        for matrix, vector in zip(new_data, Y):
            a = hankel(vector[:L+1], vector[L:])
            matrix[:] = a

        # return back torch tensor
        return torch.from_numpy(new_data).to(device)

    def batch_music_spectrum(self, data, A, K):

        U, _, _ = torch.svd(data)
        U2 = U[..., K:]
        U2H = torch.conj(torch.transpose(U2, 1, 2))
        term = U2H @ A[:U2H.shape[-1], :] # do in case given hankel data
        projection = torch.linalg.norm(term, dim=1)
        pseudospectrum = 1 / projection
        pseudospectrum = normalize(pseudospectrum, dim=1)

        return pseudospectrum.to(device)

    def batch_find_peaks(self, pseudospectrum, K):

        # need to convert to numpy to get peaks
        data = pseudospectrum.detach().cpu().numpy()
        batch_size, _ = data.shape

        peaks = np.zeros([batch_size, K])
        for pred_peaks, spectrum in zip(peaks, data):
            peak_ind, peak_info = find_peaks(spectrum, height=0)    
            peak_height = peak_info['peak_heights']
            peak_sortind = peak_ind[peak_height.argsort()]
            pred_peaks[:] = np.sort(peak_sortind[-K:])

        return torch.from_numpy(peaks).to(device)

    def batch_support_least_squares(self, data, peaks, A):

        batch_size, K = peaks.shape
        M, N = A.shape

        A_supp = torch.zeros([batch_size, M, K]).to(torch.complex128)
        peaks = peaks.to(torch.int)
        for i in range(batch_size):
            A_supp[i] = torch.from_numpy(A[:, peaks[i]])
        data = data.view(batch_size, -1, 1)
        x_supp = torch.bmm(torch.linalg.pinv(A_supp), data).view(batch_size, K)

        x_ls = np.zeros([batch_size, N], dtype=complex)
        for i in range(batch_size):
            x_ls[i, peaks[i]] = x_supp[i]

        return torch.from_numpy(x_ls).to(device)



