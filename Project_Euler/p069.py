# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 16:56:26 2020

@author: ktpss
"""

from math import gcd
import numpy as np
import time

N = 10**6

def isPrime(n):
    if (n<2):
        return 0
    else:
        ct = 0
        for i in range(2, int(np.sqrt(n))+1):
            if (n%i != 0):
                ct = ct+1
        if (ct == (int(np.sqrt(n))-1)):
            return 1
        else:
            return 0

Dl = []
p = 1
for i in range(N):
    if (isPrime(i)==1):
        p = p*i
        if (p<N):
            Dl.append(p)
        else:
            break

t0 = time.time()
def totientFunction(n, tn, tphi):
    phi_n = 1
    for i in range(3, n, 2):
        if (n/phi_n<tphi):
            return 0
        if (gcd(n, i) == 1):
            phi_n = phi_n+1
    if (n/phi_n>tphi):
        return [n, n/phi_n]
    else:
        return 0

tn = 0
tphi = 0
for a in Dl:
    if (totientFunction(a, tn, tphi) != 0):
        tn = totientFunction(a, tn, tphi)[0]
        tphi = totientFunction(a, tn, tphi)[1]

print(tn, tphi)
print(time.time()-t0)

"""
t1 = time.time()
tn = 0
tphi = 0
for n in range(6, N+6, 6):
    phi_n = 1
    for i in range(3, n, 2):
        if (n/phi_n<tphi):
            break
        if (gcd(n, i) == 1):
            phi_n = phi_n+1
    if (n/phi_n>tphi):
        tn = n
        tphi = n/phi_n

print(tn, tphi)
print(time.time()-t1)
"""