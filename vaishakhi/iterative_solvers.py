import numpy as np
from itertools import islice
from fractions import Fraction
from scipy.linalg import solve
import time 

def multiply(A,x):
    n = len(x)
    r = np.zeros(n)
    for i in range (n):
        r[i] = sum([A[i, j] * x[j] for j in range(n)])
    return r

def norm(x,y):
    n = len(x)
    return (np.sum([x[i]*y[i] for i in range(n)]))**0.5

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
        res = norm(x-x_new, x-x_new)
        if np.allclose(x, x_new, rtol=tol):
            break
        x = x_new
        curve_r.append(res)
        curve_itr.append(l)
    return x, curve_r, curve_itr

def jacobi(A,b,tol):
    max_iter=55
    x = np.ones_like(b)
    curve_r = []
    curve_itr = []
    itr = 0
    D = np.diag(A)
    T = A - np.diagflat(D)
    x_new = np.ones_like(x)
    for i in range(max_iter):
        x_new = (b - multiply(T,x)) / D
        res = norm(x-x_new,x-x_new)
        if np.allclose (x, x_new, atol=tol, rtol=0.):
            break
        x = x_new
        itr += 1
        curve_r.append(res)
        curve_itr.append(itr)
    return x, curve_itr,curve_r

def conjugate_grad(A, b, tol):
    itr=0 
    n = len(b)
    x = np.ones(n)
    # r = b - np.dot(A, x) 
    r = b - A.dot(x)
    d = np.copy(r)
    rk_norm = np.sqrt(np.dot(r,r))
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
            curve_r.append(rk_norm)
            curve_itr.append(itr)   
    return x, curve_itr,curve_r

def cg_fly(func, b, tol):
    itr=0 
    n = len(b)
    x = np.zeros(n)
    r = b - multiply(func,x)
    d = np.copy(r)
    rk_norm = np.linalg.norm(r)
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
            curve_r.append(rk_norm)
            curve_itr.append(itr)   
    return x, curve_itr,curve_r
    
def mult(f,x):
    n = len(x)
    r = np.zeros(n)
    for i in range (n):
        r[i] = sum([f(i, j) * x[j] for j in range(n)])
    return r
   
def inverse(A, solver, tol):
    n= A.shape[0]
    X = []
    B = np.identity(n)
    for i in range(n):
        x,i,r = solver(A,B[:,i],tol)
        X.append(x)
    return np.array(X).T

def gs_inverse(A,tol):
    n= A.shape[0]
    X = []
    B = np.identity(n)
    for i in range(n):
        x,i = gauss_seidal(A,B[:,i],tol)
        X.append(x)
    return np.array(X).T

def jacobi_inverse(A,tol):
    n= A.shape[0]
    X = []
    B = np.identity(n)
    for i in range(n):
        x,i = jacobi(A,B[:,i],tol)
        X.append(x)
    return np.array(X).T

def cg_inverse(A,tol):
    n= A.shape[0]
    X = []
    B = np.identity(n)
    for i in range(n):
        x,i,r = conjugate_grad(A,B[:,i],tol)
        X.append(x)
    return np.array(X).T

def cg_fly_inverse(A,n,tol):
    X = []
    B = np.identity(n)
    for i in range(n):
        x,i,r = cg_fly(A,B[:,i],tol)
        X.append(x)
    return np.array(X).T
    
