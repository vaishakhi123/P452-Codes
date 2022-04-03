import numpy as np
from scipy.sparse.linalg import cg
import time
import sys
from itertools import islice
from fractions import Fraction
import scipy.sparse.linalg as spl
from scipy.linalg import solve
import matplotlib.pyplot as plt
from pprint import pprint


sys.path.insert(0, '../')
from vaishakhi.elimination_solvers import *
from vaishakhi.iterative_solvers import *
np.set_printoptions(linewidth=140)


class MatrixFly:
    r"""
    Generates a matrix without actually storing it. 
    """
    def __init__(self, rows, cols, func):
        
        r"""
        Constructs the ekements of the matrix from the given dimesnions and function
        """        
        self.rows = rows
        self.cols = cols
        self.shape = (rows,cols)
        self.func = func

    def __getitem__(self, indices):
        i, j = indices
        return self.func(i, j)

    def __str__(self):
        r"""
        For printing the array, we have to store it. 
        """ 
        B = []
        #return "Matrix with {0} rows and {1} columns".format(self.rows,self.cols)
        for i in range(self.rows): 
                B.append([self.func(i,j) for j in range(self.cols)])
        return np.array(B).__str__()
    
    def dot(self, b):
        r"""
        Dot product of matrices or matrix with a vector 
        """ 
        if len(b.shape) == 1:
            x = np.zeros(self.rows)
            for i in range(self.rows):
                x[i] = sum([self.func(i, j) * b[j] for j in range(self.cols)])

            return x
        elif len(b.shape) == 2:
            X = np.zeros((self.rows, b.shape[1]))
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    X[i, j] = sum([self.func(i, k) * b[k, j] for k in range(self.cols)])
            
            return X
        else:
            print("Error")


def delta(i,j):
    r"""
    Returns the 0 or 1 resembling the dirac delta function 
    """ 
    return 1 if i==j else 0 

def func(i,j):
    r"""
    Lagrangian of the system 
    """ 
    return 0.5*(delta(i+1,j)+delta(i-1,j)-(2*delta(i,j))) + (0.2**2)*delta(i,j)

tol = 1e-06
print("\n-----Inverse of 20x20 matrix without storing it using Conjugate Gradient----\n")
tic2 = time.process_time()
A = MatrixFly(20,20,func)
A_inv,l,r = inverse(A,conjugate_grad,tol)
print(A)
print("The inverse of the matrix is \n {0}".format(A_inv))
print("Conjugate gradient iverse inverse time",time.process_time()-tic2)


plt.plot(l,r)
plt.xlabel("Iteration")
plt.ylabel("Residue")
plt.show()
plt.show()



