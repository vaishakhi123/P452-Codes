import sys

sys.path.insert(0, '../')
from vaishakhi.MLCG import *

initial = 0
final = 1 # gets the value of pi
n = 10000

ar = MLCG(xo = 5, m = 1021, a = 65, c = 0, random_num = np.zeros(n),N = n)
integral = 0.0
  
# function to calculate the sin of a particular
# value of x
def f(x):
    return 8*(1-x**2)
  
for i in ar:
    integral += f(i)
ans = (final-initial)/float(n)*integral
  
# prints the solution
print ('The estimated value calculated of integration is {}.'.format(ans))
