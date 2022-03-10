import numpy as np
import numpy.linalg as la
from itertools import islice
from fractions import Fraction

#import the multiply and norm function written from scratch from "vaishakhi" library
from vaishakhi.iterative_solvers import multiply, norm

#returns the eigen value when called
def  eigenvalue(A,x):
    Ax = multiply(A,x)
    return norm(x,Ax)**2

#translates vector (v^T)
def trans(v): #
    v_1 = np.copy(v)
    return v_1.reshape((-1, 1))

#Power algorithm code
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

  
#reads the matrices from text input file
m = [5,6,1]
M = []
with open('mstrimat.txt') as file:
    for row_count in m:
        rows = islice(file, row_count) # read `row_count` lines from the file
        rows = [row.split() for row in rows] # split every line in the file on whitespace
        rows = [[float(Fraction(cell)) for cell in row] for row in rows] # convert to int
        M.append(rows) # add to `M

A = np.array(M[0])
E,X = power(A)

print("The dominant eigen value is {0: .5f} and its coreesponding eigen vector is {1}".format(E[0], X[0,]))
print("The second dominant eigen value is {0: .5f} and its coreesponding eigen vector is {1}".format(E[1], X[1,]))

a = -1
c = -1
b = 2
n = 5

#checking whether the eigen values and vectors satisfy the equation given in question" 
lam = [b + (2*np.sqrt(a*c)*np.cos(k*np.pi/(n+1))) for k in range(1,3)]
eig = [[2*((np.sqrt(c/a))**k)*np.sin(i*k*np.pi/(n+1)) for i in range(1,6) ] for k in range (1,3)] 

#eigen values checking
if ((lam == E).all):
    print("True - Eigen values match")

#eigen vectors checking
if ((X[0]/norm(X[0],X[0]) == eig[0]).all):
    print("True - First Eigen vector match")
elif ((X[1]/norm(X[1],X[1]) == eig[1]).all):
    print("True - Second Eigen vector match")
else:
    print("Doesn't match")



