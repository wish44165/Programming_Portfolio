# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 18:35:33 2021

@author: yuhsi
"""

import numpy as np

def summation(i):
    r = 1/2
    return 1*(1-r**i) / (1-r)

n = 6
A = 2 * np.eye(n) - 1 * np.eye(n, n, 1) - 1 * np.eye(n, n, -1)

print('A =\n', A)

B = np.zeros((n, n))

v = np.ones((n, 1))
for i in range(int((n+1) / 2)):
    v[i] = summation(i+1)
    v[n - i - 1] = summation(i+1)
    
v *= 2**(n - int(n/10))

print('v =\n', v)
print('minimum element in vector Av is', np.min(A@v))