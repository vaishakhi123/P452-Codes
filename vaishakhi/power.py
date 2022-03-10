import numpy as np
import numpy.linalg as la
from itertools import islice
from fractions import Fraction

def  eigenvalue(A,x):
    Ax = multiply(A,x)
    return norm(x,Ax)**2

def trans(v): # translates vector (v^T)
    v_1 = np.copy(v)
    return v_1.reshape((-1, 1))

def power(A):
    tol = 1e-8
    n,d = A.shape
    x = np.random.rand(n)
    x = x/norm(x,x)
    ev = eigenvalue(A,x)
    Ac = np.copy(A)
    eig = []
    eigvec = []
    for i in range(2):
        while True:
            Ax = multiply(Ac,x)
            x_new = Ax/norm(Ax, Ax)
            ev_new = eigenvalue(Ac,x_new)
            if np.abs((ev - ev_new)/ev_new) < tol: #if(abs(lamb - x_norm) <= eps): #if np.abs(ev - ev_new) < 0.01:
                break   
            else:   
                x = x_new
                ev = ev_new
        eig.append(ev)
        eigvec.append(x)
        Ac = Ac - (eig[i]*x_new*trans(x_new))
        
    return np.array(eig), np.array(eigvec)
 
def multiply(A,x):
    n = len(x)
    r = np.zeros(n)
    for i in range (n):
        r[i] = sum([A[i, j] * x[j] for j in range(n)])
    return r

def norm(x,y):
    n = len(x)
    return (np.sum([x[i]*y[i] for i in range(n)]))**0.5
  

m = [5]
M = []
with open('mstrimat.txt') as file:
    for row_count in m:
        rows = islice(file, row_count) # read `row_count` lines from the file
        rows = [row.split() for row in rows] # split every line in the file on whitespace
        rows = [[float(Fraction(cell)) for cell in row] for row in rows] # convert to int
        M.append(rows) # add to `M

A = np.array(M[0])
X, E = power(A)
print(X,E)
print("The dominant eigen value is {0: .5f} and its coreesponding eigen vector is {1}".format(X[0], E[0,]))
print("The second dominant eigen value is {0: .5f} and its coreesponding eigen vector is {1}".format(X[1], E[1,]))


