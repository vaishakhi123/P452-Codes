import sys
from scipy.sparse.linalg import cg
import time
import sys
from itertools import islice
from fractions import Fraction
import scipy.sparse.linalg as spl
from scipy.linalg import solve
import matplotlib.pyplot as plt
from pprint import pprint

sys.path.insert(0,'../')
from vaishakhi.elimination_solvers import *
from vaishakhi.iterative_solvers import *

m =[4,1,6,1,4,1,6,1,3,3]
M = []
with open('matrix.txt') as file:
    for row_count in m:
        rows = islice(file, row_count) # read 'row_count' lines from the file
        rows = [row.split() for row in rows] # split every line in the file on whitespace
        rows = [[float(Fraction(cell)) for cell in row] for row in rows] # convert to int
        M.append(rows) # add to 'M'
A = np.array(M[6])
b = np.array(M[7]).ravel()
tol = 1e-04

figure, axis = plt.subplots(1, 3)

#The matrix given in the question was not symmteric. So while solving this question I changed the element A_12 = A_45 = -1. This made the matrix symmetric

##Solving the almost sparse system 
print("------------Ax=b----------\n")

print("\n----- LU decomposition----\n")
tic1 = time.process_time()
x = lu_dolittle(A,b)
print("Time Taken",time.process_time()-tic1)
print("The solution for Ax = b is \n{0}".format(x))
error = A.dot(x)-b
print("With error \n", error)

print("\n-----2. Jacobi iteration ----\n")
tic1 = time.process_time()
x,l,r = jacobi(A,b,tol)
print("Time Taken",time.process_time()-tic1)
print("The solution for Ax = b is {0} after {1} iterations".format(x,l))
error = A.dot(x)-b
print("With error \n", error)
axis[0].plot(l, r)
axis[0].set_title("Jacobi")

print("\n-----3. Gauss-seidal iteration ----\n")
tic1 = time.process_time()
x,l,r = gauss_seidal(A,b,tol)
print("Time Taken",time.process_time()-tic1)
print("The solution for Ax = b is {0} after {1} iterations".format(x,l))
error = A.dot(x)-b
print("With error \n", error)
axis[1].plot(l, r)
axis[1].set_title("Gauss Seidal")

print("\n-----4. Conjugate gradient iteration ----\n")
tic1 = time.process_time()
x,l,r = conjugate_grad(A,b,tol)
print("Time Taken",time.process_time()-tic1)
print("The solution for Ax = b is {0} after {1} iterations".format(x,l))
error = A.dot(x)-b
print("With error \n", error)
axis[2].plot(l, r)
axis[2].set_title("Conjugate gradient")
figure.supxlabel('Iteration')
figure.supylabel('Residue')

##Finding the inverse of the almost sparse matrix
print("\n------------Inverse of A----------\n")

print("\n-----1. Gauss Seidal----\n")
tic2 = time.process_time()
A_inv,l,r = inverse(A,gauss_seidal,tol)
print("The inverse of the matrix is \n {0}".format(A_inv))
print("Inverse time",time.process_time()-tic2)

print("\n-----2. Jacobi----\n")
tic2 = time.process_time()
A_inv,l,r = inverse(A,jacobi,tol)
print("The inverse of the matrix is \n {0}".format(A_inv))
print("Inverse time",time.process_time()-tic2)

print("\n-----3. Conjugate Gradient----\n")
tic2 = time.process_time()
A_inv,l,r = inverse(A,conjugate_grad,tol)
print("The inverse of the matrix is \n {0}".format(A_inv))
print("Inverse time",time.process_time()-tic2)


