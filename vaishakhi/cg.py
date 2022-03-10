import numpy as np
from scipy.sparse.linalg import cg
from itertools import islice
from fractions import Fraction
from numpy import linalg as LA
import scipy.sparse.linalg as spl
from scipy.linalg import solve

def conjugate_grad(A, X=None):
    """
    Description
    -----------
    Solve a linear equation Ax = b with conjugate gradient method.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    b: 1d numpy.array
    x: 1d numpy.array of initial point
    Returns
    -------
    1d numpy.array x such that Ax = b
    """
    n = A.shape[0]
    X = np.identity(n)
    B = np.identity(n)
    R = B - A.dot(X) 
    D = np.copy(R)
    Rk_norm = LA.norm(R)
    for i in range(2*n):
        AD = np.dot(A,D)
        AD = 0.5*(AD.T + AD)
        RR = np.dot(R,R)
        alpha = RR / np.dot(D,AD)
        X += alpha * D
        R -= alpha * AD
        Rk_norm = LA.norm(R)
        if Rk_norm < 1e-4:
            print ('Itr:', i)
            break
        beta = np.dot(R,R)/R
        D = beta * D + R
    return X

m  = [4,1,6,1,4,1,6,1,3,3]
M = []
with open('matrix.txt') as file:
    for row_count in m:
        rows = islice(file, row_count) # read `row_count` lines from the file
        rows = [row.split() for row in rows] # split every line in the file on whitespace
        rows = [[float(Fraction(cell)) for cell in row] for row in rows] # convert to int
        M.append(rows) # add to `M
#x_real= spl.cg(np.array(M[4]),np.array(M[5]).ravel(), x0=None, tol=1e-04, maxiter=None, M=None, callback=None, atol=None)
print(conjugate_grad(np.array(M[2])))
print(LA.inv(np.array(M[2])))




