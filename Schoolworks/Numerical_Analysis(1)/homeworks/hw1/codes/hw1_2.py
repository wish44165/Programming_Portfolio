# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 08:56:22 2020

@author: ktpss
"""

import numpy as np
import time

N = 10**6
s = 0

a = np.ones(N)
b = np.ones(N)

def doubleSummation(s, n):
    for i in range(1, n+1):
        for j in range(1, i+1):
            s = s+a[i]*b[j]
    return s

def recursiveSummation(s, n):
    if (n==1):
        s = s+a[0]*b[0]
        return s
    else:
        for i in range(1, n+1):
            s = s+a[n]*b[i]
        return recursiveSummation(s, n-1)

print('--------double summation--------')
t1 = time.time()
print('sum =', doubleSummation(s, 2*10**3))
t2 = time.time()
print('time spent:', t2-t1)
print('\n--------recursive summation--------')
t3 = time.time()
print('sum =', recursiveSummation(s, 2*10**3))
t4 = time.time()
print('time spent:', t4-t3)