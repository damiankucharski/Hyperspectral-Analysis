import torch
import torch.nn.functional as F
import numpy as np
import os
import scipy.io as sio

def loadData(name):
    data_path = os.path.join(os.getcwd(),'data')
    if name == 'IP':
        data = sio.loadmat(os.path.join(data_path, 'Indian_pines_corrected.mat'))['indian_pines_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Indian_pines_gt.mat'))['indian_pines_gt']
    elif name == 'SA':
        data = sio.loadmat(os.path.join(data_path, 'Salinas_corrected.mat'))['salinas_corrected']
        labels = sio.loadmat(os.path.join(data_path, 'Salinas_gt.mat'))['salinas_gt']
    elif name == 'PU':
        data = sio.loadmat(os.path.join(data_path, 'PaviaU.mat'))['paviaU']
        labels = sio.loadmat(os.path.join(data_path, 'PaviaU_gt.mat'))['paviaU_gt']
    
    return data, labels

def padder(X, window_size = 5):
    pad_size = int((window_size - 1) / 2)
    return F.pad(X, (0,0,*[pad_size for i in range(4)]))
    
def create_sliding_windows(X, window_size = 5):
    X = X.permute((-1,0,1))
    X = X.unfold(1,window_size,1).unfold(2,window_size,1)
    X = X.reshape(X.shape[0],-1,*X.shape[3:])
    X = X.permute((1,2,3,0))
    return X

def create_dataset(X, y, window_size = 5):
    
    paddedX = padder(X, window_size)
    windows = create_sliding_windows(paddedX, window_size)
    y = torch.flatten(y)
    assert len(y) == len(windows)
    return windows, y
    