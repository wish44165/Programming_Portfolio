# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 05:32:02 2021

@author: yuhsi

department: IMM
student number: 309653012
name: yuhsi, Chen
"""

import numpy as np
from scipy import sparse as sps
from scipy.linalg import norm,eigh

from pydavidson import JDh

# ---- set up constants
N,N1=1000,10    #matrix dimension, non-zero elements each row.
sigma=0
k=10

# ---- construnct a Random Hermitian Matrix
mat = sps.coo_matrix((np.random.random(N1*N)+1j*np.random.random(N1*N),\
        (np.random.randint(0,N,N1*N), np.random.randint(0,N,N1*N))),shape=(N,N))
mat = mat.T.conj() + mat + sps.diags(np.random.random(N),0)*3
#print(mat)

n = 1000
A = np.eye(n) + 0.5 * np.eye(n, n, 1) + 0.5 * np.eye(n, n, -1)
for i in range(n):
    A[i][i] = i+1
sA = sps.csr_matrix(A)
#print(sA)

#solve it!
e,v=JDh(sA,k=k,v0=None,M=None,    #calculate k eigenvalues for mat, with no initial guess.
        tol=1e-9,maxiter=1000000,    #set the tolerence and maximum iteration as a stop criteria.
        which='SL',sigma=sigma,    #calculate selected(SL) region near sigma.
        #set up the solver for linear equation A*x=b(here, Jacobi-Davison correction function),
        #we use bicgstab(faster but less accurate than '(l)gmres') here.
        #preconditioning is used during solving this Jacobi-Davison correction function.
        linear_solver_maxiter=20,linear_solver='bicgstab',linear_solver_precon=True,
        iprint=1)   #medium information are printed to stdout, see advanced parameters in API.

print('Get eigenvalues %s'%e)
