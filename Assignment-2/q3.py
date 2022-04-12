import sys

sys.path.insert(0, '../')
from vaishakhi.MLCG import *

initial = -1
final = 1 # gets the value of pi
n = 1000000
m = [1021,16381]
a = [65,572]

# function to calculate the sin of a particular
# value of x
def f(x):
    return 4*(1-x**2)

for j in range (0,2):
    ar = MLCG(m = m[j], a = a[j], c = 0,initial = initial,final = final, N = n)
    integral = 0.0  
    for i in ar:
        integral += f(i)
    ans = (final-initial)/float(n)*integral
    print ('The estimated volume of Steinmetz solid by Monte Carlo integration of integral 4(1-x\u00b2) from (-1,1) is {} for m = {} and a = {}.'.format(ans,m[j],a[j]))

st_points = 0
i =0

while i<n:
    
    x = rand_uniform(1021,65,0,-1,1)
    y = rand_uniform(1021,65,0,-1,1)
    z = rand_uniform(1021,65,0,-1,1)    
    #print(x,y,z)
    if(x**2 + y**2 <= 1) and (z**2 + y**2 <= 1):
        st_points+=1

    i+=1

vol = 8*st_points/n
print('\nThe estimated volume of Steinmetz solid by Monte Carlo i.e throwing points is', vol)  
