import numpy as np
from itertools import islice
from numpy import linalg as LA
from scipy.linalg import solve
from fractions import Fraction
from math import sqrt
import time
from memory_profiler import memory_usage

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

#internally obtained eigen vals/vecs from numpy
print("matrix\n",a)
tic0=time.process_time()
w, v = LA.eigh(a)
print("\n---Internal numpy method:--\n")

print("Numpy time",time.process_time()-tic0)

print("eigenvalues:\n",w)
print("vectors:\n",v)
 
#@profile
def jacobi(ain,tol): # Jacobi method
 
    def maxElem(a): # Find largest off-diag. element a[k,l]
        n = len(a)
        aMax = 0.0
        for i in range(n-1):
            for j in range(i+1,n):
                if abs(a[i,j]) >= aMax:
                    aMax = abs(a[i,j])
                    k = i; l = j
        return aMax,k,l
 
    def rotate(a,p,k,l): # Rotate to make a[k,l] = 0
        n = len(a)
        aDiff = a[l,l] - a[k,k]
        if abs(a[k,l]) < abs(aDiff)*1.0e-36: t = a[k,l]/aDiff
        else:
            phi = aDiff/(2.0*a[k,l])
            t = 1.0/(abs(phi) + sqrt(phi**2 + 1.0))
            if phi < 0.0: t = -t
        c = 1.0/sqrt(t**2 + 1.0); s = t*c
        tau = s/(1.0 + c)
        temp = a[k,l]
        a[k,l] = 0.0
        a[k,k] = a[k,k] - t*temp
        a[l,l] = a[l,l] + t*temp
        for i in range(k):      # Case of i < k
            temp = a[i,k]
            a[i,k] = temp - s*(a[i,l] + tau*temp)
            a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
        for i in range(k+1,l):  # Case of k < i < l
            temp = a[k,i]
            a[k,i] = temp - s*(a[i,l] + tau*a[k,i])
            a[i,l] = a[i,l] + s*(temp - tau*a[i,l])
        for i in range(l+1,n):  # Case of i > l
            temp = a[k,i]
            a[k,i] = temp - s*(a[l,i] + tau*temp)
            a[l,i] = a[l,i] + s*(temp - tau*a[l,i])
        for i in range(n):      # Update transformation matrix
            temp = p[i,k]
            p[i,k] = temp - s*(p[i,l] + tau*p[i,k])
            p[i,l] = p[i,l] + s*(temp - tau*p[i,l])
 
    
    n = len(a)
    maxRot = 5*(n**2)       # Set limit on number of rotations
    p = np.identity(n)     # Initialize transformation matrix
    for i in range(maxRot): # Jacobi rotation loop 
        aMax,k,l = maxElem(a)
        if aMax < tol: return np.diagonal(a),p
        rotate(a,p,k,l)
    print('Jacobi method did not converge')

tol = 1e-9
print("\n---Jacobi method:---\n")

tic = time.process_time()

wj, vj = jacobi(a,tol)

print("Jacobi Givens time",time.process_time()-tic)
print("eigenvalues:\n",wj)
print ("vectors:\n",vj)
mem_usage = memory_usage((jacobi,(a,tol)))
print('Memory usage (in chunks of .1 seconds): %s' % mem_usage)
print('Maximum memory usage: %s' % max(mem_usage))

