#!/usr/bin/env python3

import numpy as np
from scipy.linalg import hankel, svd, norm, pinv
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

def singleSnapshotMUSIC(y, A, k):
    M, N =  A.shape
    L = M // 2
    Hkl = hankel(y[:L+1], y[L:])
    U, s, _ = svd(Hkl, full_matrices=True)
    U2 = U[:, k:] # must know source number
    projection = norm(U2.conj().T @ A[:L+1, :], axis=0)
    pseudospectrum = 1 / projection
    pseudospectrum /= pseudospectrum.max()

    peak_ind, peak_info = find_peaks(pseudospectrum, height=0)    
    peak_height = peak_info['peak_heights']
    peak_sortind = peak_ind[peak_height.argsort()]
    pred_peaks = peak_sortind[-k:]


    A_supp = A[:, pred_peaks]
    x_supp = pinv(A_supp) @ y

    x_music = np.zeros(N, dtype=A_supp.dtype)
    x_music[pred_peaks] = x_supp

    return x_music, np.sort(pred_peaks)

def singleSnapshotMUSIC_OffGrid(y, A, manifold, k):
    M, N =  A.shape
    L = M // 2
    Hkl = hankel(y[:L+1], y[L:])
    U, s, _ = svd(Hkl, full_matrices=True)
    U2 = U[:, k:] # must know source number

    upsample = 4
    uniform = np.arange(M).reshape(-1, 1)
    hfgrid = (1/(upsample*N)) * np.arange(upsample*N).reshape(-1, 1)
    A_new = manifold(uniform @ hfgrid.T)

    projection = norm(U2.conj().T @ A_new[:L+1, :], axis=0)
    pseudospectrum = 1 / projection
    pseudospectrum /= pseudospectrum.max()

    peak_ind, peak_info = find_peaks(pseudospectrum, height=0)    
    peak_height = peak_info['peak_heights']
    peak_sortind = peak_ind[peak_height.argsort()]
    pred_peaks = peak_sortind[-k:]


    A_supp = A_new[:, pred_peaks]
    x_supp = pinv(A_supp) @ y

    x_music = np.zeros(N*upsample, dtype=A_supp.dtype)
    x_music[pred_peaks] = x_supp

    return x_music, np.sort(pred_peaks)
