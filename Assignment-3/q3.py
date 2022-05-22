# Simple Numerical Laplace Equation Solution using Finite Difference Method
import numpy as np
import matplotlib.pyplot as plt

# Set maximum iteration
maxIter = 500

# Set Dimension and delta
lenX = lenY = 20 #we set it rectangular
delta = 1

# Boundary condition
u_top = 0
u_bottom = 1
u_left = 0
u_right = 0

# Initial guess of interior grid
u_guess = 0.3

# Set colour interpolation and colour map
colorinterpolation = 50
colourMap = plt.cm.jet #you can try: colourMap = plt.cm.coolwarm

# Set meshgrid
X, Y = np.meshgrid(np.arange(0, lenX), np.arange(0, lenY))

# Set array size and set the interior value with Tguess
u = np.empty((lenX, lenY))
u.fill(u_guess)

# Set Boundary condition
u[(lenY-1):, :] = u_top
u[:1, :] = u_bottom
u[:, (lenX-1):] = u_right
u[:, :1] = u_left

# Iteration (We assume that the iteration is convergence in maxIter = 500)

for iteration in range(maxIter):
    for i in range(1,lenX-1, delta):
        for j in range(1, lenY-1, delta):
            u[i, j] = 0.25 * (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1])



# Configure the contour
plt.title("Contour of Potential")
plt.contourf(X, Y, u, colorinterpolation, cmap=colourMap)
plt.xlabel("X steps")
plt.ylabel("Y steps")
# Set Colorbar
plt.colorbar()

# Show the result in the plot window
plt.show()
plt.savefig("q3_laplace.png")
