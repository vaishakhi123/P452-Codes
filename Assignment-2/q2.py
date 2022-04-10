import sys

sys.path.insert(0, '../')
from vaishakhi.MLCG import *

initial = 0
final = 1 # gets the value of pi
n = 10000


circlex = []
circley = []
squarex = []
squarey = []

i =0
x = MLCG(xo = 2, m = 16381, a = 572, c = 0, random_num = np.zeros(n), N=n)
y = MLCG(xo = 5, m = 16381, a = 572, c = 0, random_num = np.zeros(n), N=n)
while i<n:
    
    if(x[i]**2 + y[i]**2 <= 1):
        circlex.append(x)
        circley.append(y)
    else:
        squarex.append(x)
        squarey.append(y)
    i+=1
    
pi = 4*len(circlex)/float(n)
print('The estimated value calculated of pi by throwing points is', pi)   


# array of zeros of length N
ar = MLCG(xo = 5, m = 16381, a = 572, c = 0, random_num = np.zeros(n),N = n)
integral = 0.0
  
# function to calculate the sin of a particular
# value of x
def f(x):
    return np.sqrt(1-x**2)
  
for i in y:
    integral += f(i)
ans = (final-initial)/float(n)*integral
  
# prints the solution
print ('The estimated value calculated of pi by monte carlo integration is {}.'.format(ans*4))

#plt.plot(circlex, circley,'b.')
#plt.plot(squarex,squarey,'g.')
#plt.show()
