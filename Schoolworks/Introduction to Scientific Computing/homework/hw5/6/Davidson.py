# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 03:12:57 2021

@author: yuhsi

department: IMM
student number: 309653012
name: yuhsi, Chen
"""

import numpy as np
import time

# ---- build a fake sparse symmetric matrix 
n = 2000
print('Dimension of the matrix',n,'*',n)
A = np.eye(n) + 0.5 * np.eye(n, n, 1) + 0.5 * np.eye(n, n, -1)
for i in range(n):
    A[i][i] = i+1

tol = 1e-9             # Convergence tolerance
mmax = 20              # Maximum number of iterations

# ---- setup the subspace trial vectors
k = 2000
print ('No. of start vectors:',k)
neig = 2000
print ('No. of desired Eigenvalues:',neig)
t = np.eye(n,k) # initial trial vectors
v = np.zeros((n,n)) # holder for trial vectors as iterations progress
I = np.eye(n) # n*n identity matrix
ritz = np.zeros((n,n))
f = np.zeros((n,n))

#-------------------------------------------------------------------------------
# Begin iterations  
#-------------------------------------------------------------------------------
start = time.time()
iter = 0
for m in range(k,mmax,k):
    iter = iter + 1
    print ("Iteration no:", iter)
    if iter==1:  # for first iteration add normalized guess vectors to matrix v
        for l in range(m):
            v[:,l] = t[:,l]/(np.linalg.norm(t[:,l]))
    # Matrix-vector products, form the projected Hamiltonian in the subspace
    T = np.linalg.multi_dot([v[:,:m].T,A,v[:,:m]]) # selects fastest evaluation order
    w, vects = np.linalg.eig(T) # Diagonalize the subspace Hamiltonian
    jj = 0
    s = w.argsort()
    ss = w[s]
    #***************************************************************************
    # For each eigenvector of T build a Ritz vector, precondition it and check
    # if the norm is greater than a set threshold.
    #***************************************************************************
    for ii in range(m): #for each new eigenvector of T
        f = np.diag(1./ np.diag((np.diag(np.diag(A)) - w[ii]*I)))
#        print (f)
        ritz[:,ii] = np.dot(f,np.linalg.multi_dot([(A-w[ii]*I),v[:,:m],vects[:,ii]]))
        if np.linalg.norm(ritz[:,ii]) > 1e-7 :
            ritz[:,ii] = ritz[:,ii]/(np.linalg.norm(ritz[:,ii]))
            v[:,m+jj] = ritz[:,ii]
            jj = jj + 1
    q, r = np.linalg.qr(v[:,:m+jj-1])
    for kk in range(m+jj-1):
        v[:,kk] = q[:,kk]
    for ii in range(neig):
        print (ss[ii])
    if iter==1: 
        check_old = ss[:neig]
        check_new = 1
    elif iter==2:
        check_new = ss[:neig]
    else: 
        check_old = check_new
        check_new = ss[:neig]
    check = np.linalg.norm(check_new - check_old)
    if check < tol:
        print('Block Davidson converged at iteration no.:',iter)
        break
end = time.time()

# ---- running time
print ('Block Davidson time:',end-start)
start = time.time()
eig, eigvecs = np.linalg.eig(A)
end = time.time() 
s = eig.argsort()
ss = eig[s]
print('Exact Diagonalization:')
for ii in range(neig):    
    print(ss[ii])
#print(ss[:neig])
print ('Exact Diagonalization time:',end-start)

# ---- top 3 eigenvalues
print(ss[-3:])
