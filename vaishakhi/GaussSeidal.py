import numpy as np
from itertools import islice
from fractions import Fraction
from scipy.linalg import solve
import time 
from memory_profiler import memory_usage
import matplotlib.pyplot as plt

def gauss_seidal(A,b,tol):
    limit = 100
    x = np.zeros_like(b)
    l = 0
    n = A.shape[0]
    curve_r = []
    curve_itr = []
    for it_count in range(1, limit):
        x_new = np.zeros_like(x)
        #print("Iteration {0}: {1}".format(it_count, x))
        l = l + 1
        for i in range(n):
            k1, k2 = 0, 0
            for j in range (0, n):
                if(j < i):
                    k1 += A[i, j]*x_new[j]
                elif(j > i):
                    k2 += A[i, j]*x[j]
            x_new[i] = (b[i] - k1 - k2) / A[i, i]
        res = np.linalg.norm(x-x_new)
        if np.allclose(x, x_new, rtol=tol):
            break
        x = x_new
        curve_r.append(res)
        curve_itr.append(l)
    return x, curve_r, curve_itr

def inverse(A,tol):
    n= A.shape[0]
    X = []
    B = np.identity(n)
    for i in range(n):
        x,r,i = gauss_seidal(A,B[:,i],tol)
        X.append(x)
    return np.array(X).T

m  = [4,1,6,1,4,1,6,1]
M = []
with open('matrix.txt') as file:
    for row_count in m:
        rows = islice(file, row_count) # read `row_count` lines from the file
        rows = [row.split() for row in rows] # split every line in the file on whitespace
        rows = [[float(Fraction(cell)) for cell in row] for row in rows] # convert to int
        M.append(rows) # add to `

A=np.array(M[6])
b=np.array(M[7]).ravel()
tol = 1e-4

#solves the system of equation
tic1 = time.process_time()
x,r,i = gauss_seidal(A,b,tol)
error = np.dot(A, x) - b
print("The solution for Ax = b is {0} and error is {0}".format(x,error)) 
print("Gauss Seidal time",time.process_time()-tic1)
plt.plot(i,r)
plt.xlabel("Iteration")
plt.ylabel("Residue")
plt.show()
## Inverse of the matrix
tic2 = time.process_time()
A_inv = inverse(A,tol)
print("The inverse of the matrix by Jacobi method is \n {0}".format(A_inv))
print("Jacobi inverse time",time.process_time()-tic2)
