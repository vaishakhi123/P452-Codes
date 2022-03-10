import numpy as np
from scipy.sparse.linalg import cg
import time
from itertools import islice
from fractions import Fraction
import scipy.sparse.linalg as spl
from scipy.linalg import solve
import matplotlib.pyplot as plt


from vaishakhi.elimination_solvers import *
from vaishakhi.iterative_solvers import *

m =[5,6,1]
M = []
tol=1e-5
with open('mstrimat.txt') as file:
    for row_count in m:
        rows = islice(file, row_count) # read 'row_count' lines from the file
        rows = [row.split() for row in rows] # split every line in the file on whitespace
        rows = [[float(Fraction(cell)) for cell in row] for row in rows] # convert to int
        M.append(rows) # add to 'M'
A = np.array(M[1])
b = np.array(M[2]).ravel()
tol = 1e-04

#Checking whether the matrix is symmetric and positive definite or not
#np.all(np.linalg.eigvals(A) > 0) 
#print(np.linalg.cholesky(A))

## Itervative methods
print("\n \n----------Iterative methods----------\n")
## Gauss Seidal method
print("\n-----1. Gauss Seidal----\n")
tic1 = time.process_time()
x,l,r = gauss_seidal(A,b,tol)
print("Time Taken",time.process_time()-tic1)
print("The solution for Ax = b is {0}".format(x))
error = A.dot(x)-b
print("With error \n", error)


## Jacobi method
print("\n-----2. Jacobi iteration ----\n")
tic1 = time.process_time()
x,l,r = jacobi(A,b,tol)
print("Time Taken",time.process_time()-tic1)
print("The solution for Ax = b is {0}".format(x))
error = A.dot(x)-b
print("With error \n", error)



