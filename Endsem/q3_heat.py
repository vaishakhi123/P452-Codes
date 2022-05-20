import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, '../')
from vaishakhi.iterative_solvers import multiply

dx = 0.1
dt = 0.0008
x = np.arange(0,2+dx,dx)
t = np.arange(0,4+dt,dt)
n = len(x)
m = len(t)
print(m)
alpha = dt/dx**2 
initialCond = 20*abs(np.sin(np.pi*x))
boundaryCond = [0,0]

T = np.zeros((n,m))
T[0,:] = boundaryCond[0]
T[-1,:] = boundaryCond[1]
T[:,0] = initialCond

def direct(T,n,m,alpha):
    for j in range (1,m):
        for i in range(1,n-1):
            T[i,j] = alpha*(T[i+1,j-1]+T[i-1,j-1]) + (1-2*alpha)*T[i,j-1]
    return T

def forward_mat(T,n,m,alpha):
    v = [1-2*alpha if i >0 and i<n-1 else alpha for i in range(n)]
    A = np.diag(v)
    for i in range(n-1):
        if i>0 and i<n-2:    
            A[i,i+1] = alpha
            A[i+1,i] = alpha
    #A[0,1], A[n-1,n-2] = alpha, alpha
    T_j = T[:,0]
    print(A,len(A))
    for j in range(1,m):
        T_j_1 = multiply(A,T_j)
        T[:,j] = T_j_1
        T_j = T_j_1
    return T

U = direct(T,n,m,alpha)
Y = forward_mat(T,n,m,alpha)
colorinterpolation = 1000
colourMap = plt.cm.jet #you can try: colourMap = plt.cm.coolwarm
values = [0,10,20,50,100,200,500]
for j in range(m):
    plt.plot(x,Y[:,j])
plt.legend([f't = {value} steps' for value in values])
[T, X] = np.meshgrid(t, x)
plt.contour(T, X, Y, colorinterpolation, cmap=colourMap)
plt.colorbar()
plt.xlabel('timme [s]')
plt.ylabel('distance [m]')
plt.savefig("q3_heat.png")
plt.show()

""" Boundary condition of the system restricts it to be at fixed point 0. With the evolution time, the temperature decreases, the system cools down"""


