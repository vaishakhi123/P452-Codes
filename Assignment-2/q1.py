import sys
import numpy as np

sys.path.insert(0, '../')
from vaishakhi.reg import poly_reg
from vaishakhi.elimination_solvers import *
x,y = np.genfromtxt("ass2.txt", delimiter="    ", unpack=True, dtype = float)
A,b = poly_reg(x,y,np.ones(len(y)),3)

#finding the parameters Ax=b using elimination method 
 
a = lu_dolittle(A,b)
print("The fitting parameters for the polynomial fit are",a)
#print(np.polyfit(x, y, 3))

B = np.array([[1.0,-1.0,1.0,1.0],[0,2.0,-8.0,18.0],[0,0,8.0,48.0],[0,0,0,32.0]])
c = np.array([  0.57465867 ,  4.72586144 ,-11.12821778 ,  7.66867762])
print(back_substitution(B,c))
print(np.linalg.cond(A),np.linalg.cond(B))
