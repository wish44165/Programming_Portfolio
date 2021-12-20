import numpy as np
import random
import copy
import warnings
warnings.filterwarnings("ignore")


"""
   parameters
"""
print('#### parameters ####')
N = [5, 10]
Omega = [1.1, 1.2, 1.8]
iterNum = int(1e9)
print('N =', N)
print('Omega = ', Omega)
print('Iteration number =', iterNum, '\n')

for n in N:
    print('---------------- n =', n, '----------------')
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i][j] = 1 / (i+1+j+1-1)
    
    b = np.ones((n,1))
    
    # diagonal matrix
    D = np.diag(np.diag(A))
    # lower triangular
    L = -np.tril(A, -1)
    # upper triangular
    U = -np.triu(A, 1)
    
    R = L+U
    L_ = D-L
    
    
    """
       jacobi method
    """
    print('---- Jacobi method ----')
    x = np.array([random.random() for i in range(n)]).reshape((n,1))
    for i in range(iterNum):
        x = np.linalg.inv(D)@(b+R@x)
    
    print('true value =\n', np.linalg.inv(A)@b, '\n')
    print('iterated value =\n', x,'\n')
    
    # check converge condition
    ews, evs = np.linalg.eig(-np.linalg.inv(D)@R)
    print('\n max eigenvalue =', np.max(np.abs(ews)), '\n')
    
    
    """
       gauss-seidel method
    """
    print('---- Gauss-Seidel method ----')
    x = np.array([random.random() for i in range(n)]).reshape((n,1))
    for i in range(iterNum):
        x = np.linalg.inv(L_)@(b+U@x)
        #print(x)

    print('true value =\n', np.linalg.inv(A)@b, '\n')
    print('iterated value =\n', x,'\n')

    # check converge condition
    ews, evs = np.linalg.eig(-np.linalg.inv(L_)@U)
    print('\n max eigenvalue =', np.max(np.abs(ews)), '\n')
    


"""
   SOR method
"""
print('-------- Successive over-relaxation --------')
for n in N:
    print('---------------- n =', n, '----------------')
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            A[i][j] = 1 / (i+1+j+1-1)
    
    b = np.ones((n,1))
    
    # diagonal matrix
    D = np.diag(np.diag(A))
    # lower triangular
    L = -np.tril(A, -1)
    # upper triangular
    U = -np.triu(A, 1)
    R = L+U
    L_ = L+D
    
    for w in Omega:
        print('-------- w =', w, '--------')
        x = np.array([random.random() for i in range(n)]).reshape((n,1))
        for i in range(iterNum):
            x = np.linalg.inv(D-w*L)@(w*b + (w*U + (1-w)*D)@x)
            
        print('w =', w, '\nx =\n', x)
