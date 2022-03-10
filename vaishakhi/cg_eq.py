import numpy as np
from scipy.sparse.linalg import cg
import time
from itertools import islice
from fractions import Fraction
import scipy.sparse.linalg as spl
from scipy.linalg import solve
import matplotlib.pyplot as plt
from memory_profiler import memory_usage

def conjugate_grad(A, b, tol):
    itr=0 
    n = len(b)
    x = np.ones(n)
    r = b - np.dot(A, x) 
    d = np.copy(r)
    rk_norm = np.sqrt(np.dot(r,r))
    curve_x = [x]
    curve_r = [rk_norm]
    curve_itr = [itr]
    for i in range (n):
        Ad = A.dot(d)
        rr = np.dot(r,r)
        alpha = rr / np.dot(d,Ad)
        x += alpha * d
        r -= alpha * Ad
        rk_norm = np.linalg.norm(r)
        if rk_norm < tol:
            break
        else:
            beta = np.dot(r,r)/ rr 
            d = r + (beta*d)
            itr += 1
            curve_x.append(x)
            curve_r.append(rk_norm)
            curve_itr.append(itr)   
    return x, curve_itr,curve_r

def inverse(A,tol):
    n= A.shape[0]
    X = []
    B = np.identity(n)
    for i in range(n):
        x,i,r = conjugate_grad(A,B[:,i],tol)
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

tic1 = time.process_time()
x,i,r = conjugate_grad(A,b,tol)
print("The solution for Ax = b is \n {0}".format(x)) 
print("Conjugate gradient time",time.process_time()-tic1)

mem_usage = memory_usage((conjugate_grad,(A,b,tol)))
print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
print('Maximum memory usage: %s' % max(mem_usage))

plt.plot(i,r)
plt.xlabel("Iteration")
plt.ylabel("Residue")
plt.show()

