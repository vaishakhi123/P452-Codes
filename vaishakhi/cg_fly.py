import numpy as np
from scipy.sparse.linalg import cg
import time
from itertools import islice
from fractions import Fraction
import scipy.sparse.linalg as spl
from scipy.linalg import solve
import matplotlib.pyplot as plt

def conjugate_grad(func, b, tol):
    itr=0 
    n = len(b)
    x = np.zeros(n)
    r = b - multiply(func,x)
    d = np.copy(r)
    rk_norm = np.linalg.norm(r)
    curve_x = [x]
    curve_r = [rk_norm]
    curve_itr = [itr]
    for i in range (n):
        Ad = multiply(func,d)
        rr = np.dot(r,r)
        alpha = rr / d.dot(Ad)
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
    
def multiply(f,x):
    n = len(x)
    r = np.zeros(n)
    for i in range (n):
        r[i] = sum([f(i, j) * x[j] for j in range(n)])
    return r

def inverse(A,tol):
    n= A.shape[0]
    X = []
    B = np.identity(n)
    for i in range(n):
        x,i,r = conjugate_grad(A,B[:,i],tol)
        X.append(x)
    return np.array(X).T

def delta(i,j):
    return 1 if i==j else 0    

def func(i,j):
    return 0.5*(delta(i+1,j)+delta(i-1,j)+(2*delta(i,j))) + (0.2**2)*delta(i,j)

A = np.array([[func(i, j) for j in range(20)] for i in range(20)])

print(np.linalg.inv(A))
#print("Conjugate gradient time",time.process_time()-tic1)
#print("The solution for Ax = b is {0} and error is {0}".format(x,error))
b = np.zeros(20)
b[2] =1
x,i,r = conjugate_grad(func,b,1e-6)
error = multiply(func,x)-b
print(x,"\n",error)
print(A)

## Inverse of the matrix
#tic2 = time.process_time()
#A_inv = inverse(A,tol)
#print("The inverse of the matrix by conjugate gradient method is \n {0}".format(A_inv))
#print("Conjugate gradient inverse time",time.process_time()-tic2)

#plt.plot(i,r)
#plt.xlabel("Iteration")
#plt.ylabel("Residue")
#plt.show()
