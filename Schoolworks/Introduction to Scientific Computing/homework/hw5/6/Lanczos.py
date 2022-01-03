# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 01:03:27 2021

@author: yuhsi

department: IMM
student number: 309653012
name: yuhsi, Chen
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def Lanczos( A, v, m=100 ):
    n = len(v)
    if m>n: m = n;
    # from here https://en.wikipedia.org/wiki/Lanczos_algorithm
    V = np.zeros( (m,n) )
    T = np.zeros( (m,m) )
    vo   = np.zeros(n)
    beta = 0
    for j in range( m-1 ):
        w    = np.dot( A, v )
        alfa = np.dot( w, v )
        w    = w - alfa * v - beta * vo
        beta = np.sqrt( np.dot( w, w ) ) 
        vo   = v
        v    = w / beta 
        T[j,j  ] = alfa 
        T[j,j+1] = beta
        T[j+1,j] = beta
        V[j,:]   = v
    w    = np.dot( A,  v )
    alfa = np.dot( w, v )
    w    = w - alfa * v - beta * vo
    T[m-1,m-1] = np.dot( w, v )
    V[m-1]     = w / np.sqrt( np.dot( w, w ) ) 
    return T, V


# ---- generate matrix A
n = 2000; m=2000
A = np.eye(n) + 0.5 * np.eye(n, n, 1) + 0.5 * np.eye(n, n, -1)
for i in range(n):
    A[i][i] = i+1
#print(A)

# ---- full solve for eigenvalues for reference
t1 = time.time()
esA, vsA = np.linalg.eig( A )
t2 = time.time()

# ---- approximate solution by Lanczos
t3 = time.time()
np.random.seed(3)
v0   = np.random.rand( n ); v0 /= np.sqrt( np.dot( v0, v0 ) )
T, V = Lanczos( A, v0, m=m )
esT, vsT = np.linalg.eig( T )
VV = np.dot( V, np.transpose( V ) ) # check orthogonality
t4 = time.time()
#print "A : "; print A
print ("T : "); print (T)
print ("VV :"); print (VV)
print ("esA :"); print (np.sort(esA))
print ("esT : "); print (np.sort(esT))

# ---- plot eigenvalues distribution
#plt.plot( esA, np.ones(n)*0.2,  '+' )
#plt.plot( esT, np.ones(m)*0.1,  '+' )
#plt.ylim(0,1)
#plt.show()

# ---- three largest eigenvalues of A
esAA = esA[-3:]
esAA = esAA[::-1]
print('eig(A)', esAA)
print('eig(T)', esT[:3])

# ---- running time
print(esAA - esT[:3])
print('error =',np.sum((esAA - esT[:3])**2))
print('numpy time =', t2 - t1)
print('Lanczos time=', t4-t3)