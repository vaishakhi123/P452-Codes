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

def poly_reg(x,y,sig,n):
    N = len(x)
    A = np.zeros((n+1,n+1))
    b = np.zeros(n+1)
    for j in range(n+1):
        for k in range(n+1):
            A[j,k] = np.sum([(x[i]**j*x[i]**k)/(sig[i]**2) for i in range(N)])
        b[j] = np.sum([(x[i]**j*y[i])/(sig[i]**2) for i in range(N)])
    return A,b

def power_law(x,y,sig):
    m,c,var_m,var_c = linear_reg(np.log(x),np.log(y),np.log(sig)/y) 
    power = m
    A0 = np.exp(c) 
    err_A0 = A0*np.sqrt(var_c)
    err_pow = np.sqrt(var_m)
    return power,A0,err_pow , err_A0
   
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
x,y,sig = np.genfromtxt("msfit.txt", delimiter="\t", unpack=True, dtype = None)
#print(x,y,sig)
print(x,y)
print(exp_law(x,y,sig))
m,c,var_m,var_c = exp_law(x,y,sig)
#y_new = m*x + np.log(c)
popt, pcov = curve_fit(lambda t,a,b: a*np.exp(b*t),x,y,sigma=sig,p0=[0, 0])
#y_fit = np.log(popt[0])+popt[1]*x
print(popt,np.sqrt(np.diag(pcov)))
#print(chi_square(x,y,sig,exp_law))
#plt.plot(x,np.log(y),'.')
#plt.plot(x,y_new)
#plt.plot(x,y_fit)
#plt.show()

#print("Power Law")
#x,y,sig = np.genfromtxt("mock_data.txt", delimiter="        ", unpack=True, dtype = None, skip_header=27)
#print(x,y,sig)
#print(power_law(x,y,sig))
#popt, pcov = curve_fit(func2,x,y,sigma=sig,p0=[0, 0])
#print(popt,np.sqrt(np.diag(pcov)))

#x,y = np.genfromtxt("mock_data.txt", delimiter="   ", unpack=True, dtype = None, skip_header=36)
#print(linear_reg(x,y,np.sqrt(y)))
#popt, pcov = curve_fit(lambda m,x,c: m*x+c ,x,y,sigma=np.sqrt(y),p0=[0, 0])
#print(popt,pcov,np.sqrt(np.diag(pcov)))
#S = np.sum(1/sig**2)
#Sx = np.sum(x/sig**2)
#Sy = np.sum(x/sig**2)
 #   Sxx = np.sum(x*x/sig**2)
  #  Sxy = np.sum(x*y/sig**2)
