import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '../')
from vaishakhi.reg import poly_reg
from vaishakhi.elimination_solvers import *
from vaishakhi.iterative_solvers import *

def chebyshev(x):
    phi = [1, 2*x-1, 8*x**2-8*x+1, 32*x**3-48*x**2+18*x-1] 
    return phi

def func(x,a,n,solve):

    if solve == "poly":
        return sum([a[i]*np.power(x,i) for i in range(n+1)])
    else: 
        return sum([a[i]*chebyshev(x)[i] for i in range(n+1)])        
            

n = 3
x,y = np.genfromtxt("ass2.txt", delimiter="    ", unpack=True, dtype = float)
A,b = poly_reg(x,y,np.ones(len(y)),n)
N = len(x)

#finding the fitting parameters Ax=b using elimination method and y from polynomial regression
a = lu_dolittle(A,b)
y_poly = func(x,a,n,"poly")

print("The fitting parameters for the polynomial fit are",a)

B = np.zeros((n+1,n+1))
for j in range(n+1):
    for k in range(n+1):
        B[j,k] = np.sum([(chebyshev(x[i])[k]*chebyshev(x[i])[j]) for i in range(N)])
    b[j] = np.sum([(chebyshev(x[i])[j]*y[i]) for i in range(N)])

#obtaining the fitted y using modified basis 
print("The fitting parameters for modified chebyshev are", lu_dolittle(B,b))
y_cheb = func(x,lu_dolittle(B,b),n,"cheb")

p1 = plt.plot(x,y,'o')
p2 = plt.plot(x,y_poly, 'b-')
p3 = plt.plot(x,y_cheb,'g-')
plt.legend(["Original","Polynomial fit", "Modified Chebyshev"],loc="lower right")
plt.savefig("q1.png")
print("The condition number for matrices for polynomial regression and modified chebyshev are {} and {} respectively".format(np.round(np.linalg.cond(A),4),np.round(np.linalg.cond(B),4)))
print("The lower conditional number of modified basis method tells us about the numerical stability and accuracy")


