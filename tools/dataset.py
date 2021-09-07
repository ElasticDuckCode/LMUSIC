#!/usr/bin/env python3

import torch
from torch.utils.data import Dataset
import numpy as np

class SimpleDataset(Dataset):
    '''
    Simple Dataset
    --------------
    Just give me data, and I work
    '''
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return (self.X[index], self.Y[index])

    def __len__(self):
        return self.X.shape[0]

class SparseDataset(Dataset):
    def __init__(self, N, k, T, npwr=0.0, A=None):
        '''
        Create linear measurements from sparse support
        y = Ax + w 
        with additive Gaussian noise.
        '''
        self.X = np.zeros([T, N])
        for i in range(T):
            self.X[i] = self.generateSparseSupport(N, k)
        self.Y = self.X @ A.T
        self.W = np.sqrt(npwr) * np.random.randn(*self.Y.shape)
        self.Y = self.Y + self.W

        self.X = torch.from_numpy(self.X)
        self.Y = torch.from_numpy(self.Y)
        self.W = torch.from_numpy(self.W)
    
    def __getitem__(self, i):
        return (self.X[i], self.Y[i])
    
    def __len__(self, ):
        return self.X.shape[0]

    def generateSparseSupport(self, N, k):
        s = np.zeros(N-2) # assuming N even
        s[:k] = 1.0
        reshuffle = True
        while reshuffle:
            np.random.shuffle(s)
            for i in range(N-3):
                if s[i] > 0.0 and s[i+1] > 0.0:
                    reshuffle = True
                    break
                else:
                    reshuffle = False

        s = np.append(s, 0)
        s= np.insert(s, 0, 0)
        return s

    #def regenerateData(self, npwr=0.0, A=None):
    #    self.Y = self.X @ A.T
    #    self.W = np.sqrt(npwr) * np.random.randn(*self.Y.shape)
    #    self.W = torch.from_numpy(self.W)
    #    self.Y = self.Y + self.W

        return self

