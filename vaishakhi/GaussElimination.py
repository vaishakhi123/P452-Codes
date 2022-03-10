import numpy as np
from itertools import islice
from fractions import Fraction

def elimination(A,b):
    n=A.shape[0]
    for row in range(0,n):
        for i in range(row+1,n):
            factor=A[i,row]/A[row,row]
            for j in range(row,n):
                A[i,j]=A[i,j]-factor*A[row,j]
            b[i] = b[i] - factor * b[row]
    print('A = %s and b = %s' % (A,b))
    back_substitution(A, b)
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
    print("Solution: {0}".format(x))
    
m = [4,1,6,1,4,1,6,1]
M = []

with open('matrix.txt') as file:
    for row_count in m:
        rows = islice(file, row_count) # read `row_count` lines from the file
        rows = [row.split() for row in rows] # split every line in the file on whitespace
        rows = [[float(Fraction(cell)) for cell in row] for row in rows] # convert to int
        M.append(rows) # add to ``

A=np.array(M[2])
b=np.array(M[3]).ravel()
elimination(A, b)
