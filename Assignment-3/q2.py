from numpy import array,arange
import numpy as np
import matplotlib.pyplot as plt
# Constants
m = 511000     # Mass of electron: mc^2 in ev
hbar = 197  # Planck's constant over 2*pi
L = 1     # Width of well in nm
N = 1000
h = L/N

# Potential function
def V(x):
    return 0.0

def f(r,x,E):
    """ Schrodinger equation for infinite square potential well.

    r: array containing two values - wavefunction, psi and it's derivative, phi.
    E: Energy to solve TISE """
     
    psi = r[0]
    phi = r[1]
    dpsi = phi
    dphi = (2*m/hbar**2)*(V(x)-E)*psi
    return array([dpsi,dphi],float)

# Calculate the wavefunction for a particular energy
def solve(E):
    
    psifinal = 1
    psi = 0.0
    phi = 1.0 
    while np.abs(psifinal)>0.002:
        phi = 1
        x = 0
        xp = []
        r = array([psi,phi],float)
        psip =[]
        while x< L:
            k1 = h*f(r,x,E)
            k2 = h*f(r+0.5*k1,x+0.5*h,E)
            k3 = h*f(r+0.5*k2,x+0.5*h,E)
            k4 = h*f(r+k3,x+h,E)
            r += (k1+2*k2+2*k3+k4)/6
            x = x + h
            xp = xp + [x]
            psip = psip + [r[0]]
        psifinal = r[0]
        E = E + h
    return psip, E, xp

def norm(psip):
    Area = 0
    for i in range(len(psip)-1):
        dA  = (psip[i]**2)*h
        Area = Area + dA
    return np.sqrt(Area)

E0 = [0.0,1.2]
print("--------------------Solving infinte potential well of unit width using shooting method via RK4-------------------\n")
psi1,E,xp = solve(E0[0])
print("E =",np.round(E,5),"eV for ground state")
#plt.plot(xp,psi1/norm(psi1),label="Ground state")
psi2,E,xp = solve(E0[1])
print("E =",np.round(E,5),"eV for first excited state")
plt.plot(xp,psi2/norm(psi2),label="First excited state")
plt.xlabel("x")
plt.ylabel(r"$\psi(x)$")
plt.title("Infinite potential well using shooting method via RK4")
plt.legend(loc="best")
plt.savefig("q2_first.png")
plt.show()

#output is in q2_oytput.txt file and there are plots for first two lowest energy states for the normalised wave function

