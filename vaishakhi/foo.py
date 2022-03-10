import numpy as np
from scipy.sparse.linalg import cg
import tensorflow as tf
import time
from itertools import islice
from fractions import Fraction
import scipy.sparse.linalg as spl

def conjugate_grad(A, b, x=None):
     
    n = len(b)
    x = np.zeros(n)
    r = b - np.dot(A, x) 
    d = r
    rk_norm = np.linalg.norm(r)
    curve_x = [x]
    curve_r = [r]
    curve_itr = []
    for i in range(1000):
        Ad = A.dot(d)
        alpha = rk_norm / np.dot(d, Ad)
        x += alpha * d
        r -= alpha * Ad
        rkplus1_norm = np.linalg.norm(r)
        beta = rkplus1_norm / rk_norm
        rk_norm = rkplus1_norm
        curve_x.append(x)
        curve_r.append(r)
        curve_itr.append(i)
        if rkplus1_norm < 1e-4:
            
            print ('Itr:', i)
            break
        d = beta * d + r
    print(curve_itr)
    return x

m = [4, 1,4,1,6,1,3,3]
M = []
with open('matrix.txt') as file:
    for row_count in m:
        rows = islice(file, row_count) # read `row_count` lines from the file
        rows = [row.split() for row in rows] # split every line in the file on whitespace
        rows = [[float(Fraction(cell)) for cell in row] for row in rows] # convert to int
        M.append(rows) # add to `M
x_real= spl.cg(np.array(M[4]),np.array(M[5]).ravel(), x0=None, tol=1e-04, maxiter=None, M=None, callback=None, atol=None)
print(conjugate_grad(np.array(M[4]),np.array(M[5]).ravel()))
print(x_real)



##GIVENS

#... or generate a random symmetric matrix
ndim=5
mt=np.zeros((ndim,ndim))
for dim in range(ndim,0,-1):
    v=10*np.random.rand(dim)
    if dim==ndim:
        mt = np.diag(v)
    else:
        exdim=ndim-dim
        mt +=np.diag(v,-exdim)+np.diag(v,exdim)

a=np.copy(mt)


while rk_norm > tol:
        Ad = A.dot(d)
        rr = np.dot(r, r)
        alpha = rr / d.dot(Ad)
        x += alpha * d
        r -= alpha * Ad
        beta = r.dot(r)/ rr 
        d = r + (beta*d)
        rk_norm = np.linalg.norm(r)
        itr += 1
        
        curve_x.append(x)
        curve_r.append(rk_norm)
        curve_itr.append(itr)
        #print('Iteration: {} \t x = {} \t residual = {:.4f}'.format(itr, x, rk_norm))



itr=0 
    n = len(b)
    x = np.zeros(n)
    r = b - np.dot(A, x) 
    d = r
    rk_norm = np.linalg.norm(r)
    curve_x = [x]
    curve_r = [rk_norm]
    curve_itr = [itr]
    for i in range (n):
        Ad = A.dot(d)
        rr = np.dot(r, r)
        alpha = rr / d.dot(Ad)
        x += alpha * d
        r -= alpha * Ad
        rk_norm = np.linalg.norm(r)
        if rk_norm < tol:
            break
        else:
            beta = r.dot(r)/ rr 
            d = r + (beta*d)
            itr += 1
        curve_x.append(x)
        curve_r.append(rk_norm)
        curve_itr.append(itr)
        #print('Iteration: {} \t x = {} \t residual = {:.4f}'.format(itr, x, rk_norm))
        
    return x, curve_itr,curve_r


#cg
def conjugate_grad(A, b, tol,x=None):
    n=len(b)
    xk = np.zeros(n)
    rk = b -np.dot(A, xk)
    dk = rk
    rk_norm = np.linalg.norm(rk)
    
    itr = 0
    curve_x = [xk]
    curve_r = [rk_norm]
    curve_itr = [itr]
    while rk_norm > tol:
        Adk = np.dot(A, dk)
        rkrk = np.dot(rk, rk)
        
        alpha = rkrk / np.dot(dk, Adk)
        xk = xk + alpha * dk
        rk = rk + alpha * Adk
        beta = np.dot(rk, rk) / rkrk
        dk = rk + beta * dk
        
        itr += 1
        curve_x.append(xk)
        rk_norm = np.linalg.norm(rk)
        curve_x.append(x)
        curve_r.append(rk_norm)
        curve_itr.append(itr)
        #print('Iteration: {} \t x = {} \t residual = {:.4f}'.format(itr, x, rk_norm))
        
    return x, curve_itr,curve_r
        
def conjugate_grad(A, b, tol,x=None):
    itr=0 
    n = len(b)
    x = np.zeros(n)
    r = b - np.dot(A, x) 
    d = r
    rk_norm = np.linalg.norm(r)
    curve_x = [x]
    curve_r = [rk_norm]
    curve_itr = [itr]
    for i in range (n):
        Ad = A.dot(d)
        rr = r.dot(r)
        alpha = rr / np.dot(d,Ad)
        x += alpha * d
        r -= alpha * Ad
        rk_norm = np.linalg.norm(r)
        if rk_norm < tol:
            break
        else:
            beta = r.dot(r)/ rr 
            d = r + (beta*d)
            itr += 1
        curve_x.append(x)
        curve_r.append(rk_norm)
        curve_itr.append(itr)
        #print('Iteration: {} \t x = {} \t residual = {:.4f}'.format(itr, x, rk_norm))
        
    return x, curve_itr,curve_r

def conjugate_grad(A, b, tol,x=None):
    itr=0 
    n = len(b)
    x = np.zeros(n)
    r = b - np.dot(A, x) 
    d = r
    rk_norm = np.linalg.norm(r)
    curve_x = [x]
    curve_r = [rk_norm]
    curve_itr = [itr]
    
    while rk_norm > tol:
        Ad = A.dot(d)
        rr = np.dot(r, r)
        alpha = rr / d.dot(Ad)
        x += alpha * d
        r -= alpha * Ad
        beta = r.dot(r)/ rr 
        itr += 1
        rk_norm = np.linalg.norm(r)
        curve_x.append(x)
        curve_r.append(rk_norm)
        curve_itr.append(itr)
        #print('Iteration: {} \t x = {} \t residual = {:.4f}'.format(itr, x, rk_norm))
        d = beta *d + r
    return x, curve_itr,curve_r



import numpy as np
from itertools import islice
from fractions import Fraction

#change all lower to upper to get one matrix
def LUdolittle(A):
    n=A.shape[0]
    LU = np.copy(A)
    # take one matrix LU = A and do all computation same
    for j in range(n):
        # Upper Triangular
        for i in range(1,j):
            s = np.sum([LU[i, k] * LU[k, j] for k in range(i-1)])
            LU[i,j] = A[i,j] - s

        # Lower Triangular
        for i in range(j,n):
            s = np.sum([LU[i, k] * LU[k, j] for k in range(j-1)])
            LU[i,j] = (A[i,j] - s)/LU[j,j]
             
    #print("Lower Triangular \n", lower)
    print("Upper Triangular \n ",LU)
    #print(np.dot(lower,upper))

m = [4, 1,6,1,4,1,6,1]
M = []
with open('matrix.txt') as file:
    for row_count in m:
        rows = islice(file, row_count) # read `row_count` lines from the file
        rows = [row.split() for row in rows] # split every line in the file on whitespace
        rows = [[float(Fraction(cell)) for cell in row] for row in rows] # convert to int
        M.append(rows) # add to `M

LUdolittle(np.array(M[4]))

import numpy as np
from itertools import islice
from fractions import Fraction

#change all lower to upper to get one matrix
def LUdolittle(A):
    n=A.shape[0]
    lower=np.identity(n)
    upper=np.copy(A) # take one matrix LU = A and do all computation same
    for i in range(n):
        # Lower Triangular
        for j in range(i):
            s = sum([upper[i, k] * upper[k, j] for k in range(j)])
            upper[i,j] = (A[i,j] - s) / upper[j,j]

        # Upper Triangular
        for j in range(i, n):
            s = sum([upper[i, k] * upper[k, j] for k in range(i)])
            upper[i,j] = A[i,j] - s
             
    #print("Lower Triangular \n", lower)
    print("Upper Triangular \n ",upper)
    #print(np.dot(lower,upper))

m = [4, 1,6,1,4,1,6,1]
M = []
with open('matrix.txt') as file:
    for row_count in m:
        rows = islice(file, row_count) # read `row_count` lines from the file
        rows = [row.split() for row in rows] # split every line in the file on whitespace
        rows = [[float(Fraction(cell)) for cell in row] for row in rows] # convert to int
        M.append(rows) # add to `M

LUdolittle(np.array(M[4]))


