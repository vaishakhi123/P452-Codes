import sys
import numpy as np
from scipy import integrate
sys.path.insert(0, '../')
from vaishakhi.MLCG import *

n = 100000
ar = MLCG(m = 16381, a = 572, c = 0,initial = 0, final = 1,N = n)
integral = 0.0
initial = 0
final = 1
  
# function to calculate the sin of a particular value of x
def f(x):
    return np.exp(-x**2)

#probability function for importance sampling
def g(x,a):
    value = a*np.exp(-x)
    return value

#inverse function for variable y
def inv(y,a):
    return -np.log(1-y/a)
    
print("-------------------------Monte carlo integration----------------------------\n")  
for i in ar:
    integral += f(i)
ans = (final-initial)/float(n)*integral
print('The estimated value of the integral by monte carlo integration without important sampling is {}.'.format(np.round(ans,5)))

y = np.array(ar)
a = 1.582
x = inv(y,a)
q = f(x)/g(x,a)
print('The estimated value of the integral by monte carlo integration via important sampling using alpha*e^(-x) with alpha = 1.582 is {}.'.format(np.round(np.mean(q),5)))
print("alpha has been obtained by normalising the probability function from 0 to 1 which turned to be approximately 1.582")
