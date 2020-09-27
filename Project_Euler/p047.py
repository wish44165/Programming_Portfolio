# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 10:32:58 2020

@author: ktpss
"""

#import numpy as np
import time

Tl = 4

T = time.time()

"""
def isPrime(n):
    if (n<2):
        return 0
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
"""
        
def decompose(n):
    n0 = n
    l = []
    a = 1
    while a:
        for i in range(2, n0+1):
            if (n%i==0):
                if i not in l:
                    l.append(i)
                    if (len(l) > Tl):
                        return 0
                n = n/i
                break
        if (n==1):
            if (len(l)==Tl):
                return 1
            else:
                return 0
        
c = 0
n2 = 0
for i in range(2, 10**9):
    if (decompose(i)==1):
        if (i-n2 == 1):
            c = c+1
        else:
            c = 0
        n2 = i
    if (c==Tl-1):
        break

print(i-Tl+1)
print(time.time() - T)