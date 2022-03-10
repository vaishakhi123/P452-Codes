import numpy as np
from itertools import islice
from fractions import Fraction

def Cholesky(A):
    n = A.shape[0]
    L = np.zeros_like(A)
    for i in range(n):
        for j in range(i+1):
            add = 0
            if(j == i):
                add = sum([L[j,k]**2 for k in range(j)])
                L[j,j] = (A[j,j] - add)**0.5
            elif(j < i):
                add = sum([L[i,k]*L[j,k] for k in range(j)])
                if(L[j,j] > 0):
                    L[i,j] = int((A[i,j] - add)/L[j,j])

    return L, L.T

m = [4, 1,6,1,4,1,6,1,3,3]
M = []
with open('matrix.txt') as file:
    for row_count in m:
        rows = islice(file, row_count) # read `row_count` lines from the file
        rows = [row.split() for row in rows] # split every line in the file on whitespace
        rows = [[float(Fraction(cell)) for cell in row] for row in rows] # convert to int
        M.append(rows) # add to `M

print(Cholesky(np.array(M[6])))


