import numpy as np

def leg(n):
    if n == 4:
        x,w = np.genfromtxt("leg_roots.txt", delimiter="  ", unpack=True, dtype = float,skip_footer = 11)
        
    elif n == 5: 
        x,w = np.genfromtxt("leg_roots.txt", delimiter="  ", unpack=True, dtype = float,skip_footer = 6,skip_header = 5)
        
    elif n == 6: 
        x,w = np.genfromtxt("leg_roots.txt", delimiter="  ", unpack=True, dtype = float,skip_header = 10)

    return x,w

def gauss(f,a,b,n):
    half = float(b-a)/2.
    mid = (a+b)/2.
    x,w = leg(n)
    result =  0.
    for i in range(n):
        result += w[i] * f(half*x[i] + mid)
    result *= half
    return result

def fun(x):
    return 1/np.sqrt((1+x**2))

for i in range(4,7,1):  
    print("The potential for 2 units rod at distance of 1 unit is",gauss(fun,0,2,i), "using gaussian quadrature with n =",i)
    
"""Output
The potential for 2 units rod at distance of 1 unit is 1.443515247597232 using gaussian quadrature with n = 4
The potential for 2 units rod at distance of 1 unit is 1.4436463874023353 using gaussian quadrature with n = 5
The potential for 2 units rod at distance of 1 unit is 1.443636333848491 using gaussian quadrature with n = 6
"""
