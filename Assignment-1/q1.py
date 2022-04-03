import sys

sys.path.insert(0, '../')
from vaishakhi.elimination_solvers import *
from vaishakhi.iterative_solvers import *

m =[4,1,6,1,4,1,6,1,3,3]
M = []
tol=1e-6
with open('matrix.txt') as file:
    for row_count in m:
        rows = islice(file, row_count) # read 'row_count' lines from the file
        rows = [row.split() for row in rows] # split every line in the file on whitespace
        rows = [[float(Fraction(cell)) for cell in row] for row in rows] # convert to int
        M.append(rows) # add to 'M'
A = np.array(M[6])
b = np.array(M[7]).ravel()
tol = 1e-06

#print(np.linalg.solve(A,b))

## Gauss Jordan Elimination method
print("----------Elimination methods----------\n")
print("\n-----1. Gauss Jordon elimination----\n")
tic1 = time.process_time()
x = gauss_elimination(A,b)
print("Time Taken",time.process_time()-tic1)
print("The solution for Ax = b is \n{0}".format(x))
error = A.dot(x)-b
print("With error \n", error)

## LU decomposition method
print("\n-----2. LU decomposition----\n")
tic1 = time.process_time()
x = lu_dolittle(A,b)
print("Time Taken",time.process_time()-tic1)
print("The solution for Ax = b is \n{0}".format(x))
error = A.dot(x)-b
print("With error \n", error)

print("\n \n----------Iterative methods----------\n")
## Gauss Seidal method
print("\n-----1. Gauss Seidal----\n")
tic1 = time.process_time()
x,l,r = gauss_seidal(A,b,tol)
print("Time Taken",time.process_time()-tic1)
print("The solution for Ax = b is \n{0} after {1} iterations".format(x,l))
error = A.dot(x)-b
print("With error \n", error)

## Jacobi method
print("\n-----2. Jacobi iteration ----\n")
tic1 = time.process_time()
x,l,r = jacobi(A,b,tol)
print("Time Taken",time.process_time()-tic1)
print("The solution for Ax = b is {0} after {1} iterations".format(x,l))
error = A.dot(x)-b
print("With error \n", error)
