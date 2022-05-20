import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, '../')
from vaishakhi.reg import poly_reg
from vaishakhi.elimination_solvers import *
from vaishakhi.iterative_solvers import *


def legendre(x):
    phi = [1, x, 0.5*(3*x**2-1), 0.5*(5*x**3-3*x), (35*x**4-30**2+3)/8, (63*x**5-70*x**3+15*x)/8, (231*x**6-315*x**4+105*x**2-5)/16]
    return phi

def func(x,a,n,solve):

    if solve == "poly":
        return sum([a[i]*np.power(x,i) for i in range(n+1)])
    else: 
        return sum([a[i]*legendre(x)[i] for i in range(n+1)])        
            

n = 6
x,y = np.genfromtxt("esem4fit.txt", delimiter="\t", unpack=True, dtype = float)
A,b = poly_reg(x,y,np.ones(len(y)),n)
N = len(x)

#finding the fitting parameters Ax=b using elimination method and y from polynomial regression
a = lu_dolittle(A,b)
y_poly = func(x,a,n,"poly")

print("The fitting is done for n = ", n)
print("The fitting parameters for the polynomial fit are",a)

B = np.zeros((n+1,n+1))
for j in range(n+1):
    for k in range(n+1):
        B[j,k] = np.sum([(legendre(x[i])[k]*legendre(x[i])[j]) for i in range(N)])
    b[j] = np.sum([(legendre(x[i])[j]*y[i]) for i in range(N)])

#obtaining the fitted y using modified basis 
print("The fitting parameters for modified Legendre are", lu_dolittle(B,b))
y_leg = func(x,lu_dolittle(B,b),n,"leg")

p1 = plt.plot(x,y,'o')
p2 = plt.plot(x,y_poly, 'b-')
p3 = plt.plot(x,y_leg,'g-')
plt.legend(["Original","Polynomial fit", "Modified Legendre"],loc="lower right")

plt.savefig("q2_fit.png")
plt.show()

"""The fitting is done for n =  6
The fitting parameters for the polynomial fit are [ 0.12177995 -0.027937   -0.5244683   0.09157037  0.74308968 -0.05297491 -0.17880208]
The fitting parameters for modified Legendre are [ 1.27657207e+01  4.30168584e-03 -2.95463087e-01  1.30837436e-02 1.14118550e-01 -6.72697222e-03 -1.23845597e-02]"""

