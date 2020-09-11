# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 09:41:38 2020

@author: ktpss
"""

import numpy as np

def isPrime(n):
    if (n<=1):
        return 0
    if (n==2):
        return 1
    ct = 0
    for i in range(2, int(np.sqrt(n))+1):
        if (n%i == 0):
            break
        else:
            ct = ct+1
    if (ct == (int(np.sqrt(n))-1)):
        return 1
    else:
        return 0
    
lp = [2]

def isgoldbachConjecture(n):
    for i in range(np.max(lp)+1, n):
        if (isPrime(i) == 1):
            lp.append(i)
    for a in lp:
        for j in range(1, n):
            if (n == (a+2*(j**2))):
                return 1
            if ((a+2*j**2) > n):
                break
    return 0

for i in range(3, 10**9, 2):
    if (isPrime(i) == 0):
        if (isgoldbachConjecture(i) == 0):
            print(i)
            break