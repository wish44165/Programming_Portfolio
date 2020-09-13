# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 15:22:05 2020

@author: ktpss
"""

import numpy as np

N = 10**6

def isPrime(n):
    if (n<2):
        return 0
    if (n==2):
        return 1
    else:
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

lp = []
for i in range(N):
    if (isPrime(i) == 1):
        lp.append(i)

startNum = 0      
ctM = 0
sM = 0
for i in range(len(lp)-2):
    s = lp[i]
    ct = 0
    for j in range(i+1, len(lp)-1-i):
        s = s+lp[j]
        ct = ct+1
        if (s > N):
            break
        if (s in lp):
            if (ct>ctM):
                startNum = lp[i]
                ctM = ct
                sM = s

print(startNum, ctM+1, sM) #(start number, consecutive number, prime)