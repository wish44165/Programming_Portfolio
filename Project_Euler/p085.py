# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 20:50:03 2020

@author: ktpss
"""

import numpy as np

N = 2*10**6

def rectangleNum(a, b):
    ct = 0
    for i in range(0, a+1):
        for j in range(0, b+1):
            ct = ct+(a-i)*(b-j)
    return ct


for L in range(N):
    if (rectangleNum(L, L) > N):
        break

for L2 in range(N):
    if (rectangleNum(1, L2) > N):
        break

l = [0, 0, N]    # a, b, e
for a in range(1, L+1):
    for b in range(a, L2+1):
        e = np.abs(rectangleNum(a,b) - N)
        if (e<l[2]):
            l = [a, b, e]
        if (rectangleNum(a,b) > N):
            break

print(l[0]*l[1])