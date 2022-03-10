import numpy as np
from itertools import islice
from fractions import Fraction

def gauss_elimination(A,b):
    n=A.shape[0]
    for row in range(0,n):
        for i in range(row+1,n):
            factor=A[i,row]/A[row,row]
            for j in range(row,n):
                A[i,j]=A[i,j]-factor*A[row,j]
            b[i] = b[i] - factor * b[row]
    return back_substitution(A, b)

def lu_dolittle(A,b):
    n=A.shape[0]
    lower=np.identity(n)
    upper=np.identity(n) 
    for i in range(n):
        # Lower Triangular
        for j in range(i):
            s = sum([lower[i, k] * upper[k, j] for k in range(j)])
            lower[i,j] = (A[i,j] - s) / upper[j,j]

        # Upper Triangular
        for j in range(i, n):
            s = sum([lower[i, k] * upper[k, j] for k in range(i)])
            upper[i,j] = A[i,j] - s
    y = forward_substitution(lower, b)
    return back_substitution(upper, y)
    
# Finds X in UX=Y
def back_substitution(A,b):
    n=A.shape[0]
    x = np.zeros_like(b)
    x[n-1] = b[n-1]/A[n-1, n-1]
    C = np.zeros((n,n))
    for i in range(n-2, -1, -1):
        k= 0
        for j in range (i+1, n):
            k+= A[i, j]*x[j]

        C[i, i] = b[i] - k
        x[i] = C[i, i]/A[i, i] 
    return x

# Finds Y in LY=B where UX=Y
def forward_substitution(A,b):
    n=A.shape[0]
    x = np.zeros_like(b)
    x[0] = b[0]/A[0, 0]
    C = np.zeros((n,n))
    for i in range(1, n):
        k= 0
        for j in range (len(A)):
            k+= A[i, j]*x[j]

        C[i, i] = b[i] - k
        x[i] = C[i, i]/A[i, i]
    return x


