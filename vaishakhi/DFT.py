import numpy as np

def DFT(x):
    N = len(x)
    n = np.arange(N)
    k = n.reshape((N,1))
    e = np.exp(-2j*np.pi*k*n/N)
    X = np.dot(e,x)
    return X

#sample
freq_x = 1
