import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import sys

sys.path.insert(0, '../')
from vaishakhi.MLCG import *


walkers = 500 # Example for 500 walkers
path = []
steps = 200
for walker in range(walkers):
    walker_path=[]
    position = [0,0]
    for i in range(steps):
        move = rand_uniform(16381,572,0,0,1)
        if move <= 0.5: 
            position[0]+= -1
            position[1]+= -1
        else: 
            position[0]+= 1
            position[1]+= 1
        distance = np.sqrt(position[0]**2 + position[1]**2)
        walker_path.append(distance)
    path.append(walker_path)
path = np.array(path)

rmsd = [np.sqrt((path[:, i]**2).mean()) for i in range(steps)] 


N = np.linspace(0,201,200)
plt.plot(N,np.power(N,1/2),'.')
plt.plot(rmsd)
plt.xlabel("Steps")
plt.ylabel("Average r.m.s. distance")
plt.savefig("q1_verify.png")
plt.show()

"""The average r.m.s. distance has been plotted and is verified that is approximately R_rms = (N)^1/2"""
