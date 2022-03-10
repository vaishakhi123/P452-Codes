import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def linear_reg(x,y,sig):
    n = len(x)
    S,Sx,Sxx,Sy,Sxy=0,0,0,0,0
    
    for i in range(n):
        S += 1/sig[i]**2
        Sx += x[i]/sig[i]**2
        Sy += y[i]/sig[i]**2
        Sxx += x[i]**2/sig[i]**2
        Sxy += x[i]*y[i]/sig[i]**2
    denominator = S*Sxx - Sx**2
    num1 = S*Sxy - Sx*Sy
    num2 = Sxx*Sy - Sx*Sxy
    m = num1/denominator
    c = num2/denominator
    var_m = S/denominator
    var_c = Sxx/denominator
    cov = -Sx/denominator
    r = -Sx/(np.sqrt(S*Sxx))
    
    return m, c, var_m, var_c

def exp_law(x,y,sig):
    n = len(x)
    m,c,var_m,var_c= linear_reg(x,np.log(y),1/np.sqrt(sig)) 
    power = m
    A0 = np.exp(c)
    err_A0 = A0*np.sqrt(var_c/(n-2))
    err_pow = np.sqrt(var_m/(n-2)) 
    return power,A0,err_pow , err_A0

def chi_square(x,y,sig,law):
    m,c,var_m,var_c = law(x,y,sig)
    n = len(x)
    chi = np.sum([((np.log(y[i])-np.log(c)-m*x[i])/np.log(sig[i]))**2 for i in range(n)])
    red_chi = chi/(n-2)
    return chi, red_chi

print("Exponential Law")
x,y,sig = np.genfromtxt("msfit.txt", delimiter="", unpack=True, dtype = None)
m,c,err_m,err_c = exp_law(x,y,sig)
y_new = m*x + np.log(c)
life = 0.693/m
err_life = life*(err_m/m)
print("Half Lifetime of radioactive material is {0} with error {1}".format(life,err_life))
print("The chi and reduced chi values are",chi_square(x,y,sig,exp_law))
print("The critical chi is found to be 15.057 which implies we do not have enough evidence to reject")
plt.plot(x,y_new)
plt.xlabel("time")
plt.ylabel("log(N)")
plt.show()
