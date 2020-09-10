# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:09:22 2020

@author: ktpss
"""

import numpy as np

def isPrime(n):
    if (n==1):
        return 0
    if (n==2):
        return 1
    ct = 0
    for i in range(2, int(np.sqrt(n))+1):
        if (n%i == 0):
            break
        else:
            ct = ct+1
    if (ct == int(np.sqrt(n))-1):
        return 1
    else:
        return 0

def ltrTruncate(n):
    a = 1
    while a:
        if (len(str(n)) > 1):
            n = int(str(n)[1:])
            a = isPrime(n)
        if (len(str(n)) == 1):
            a = 0
    if (len(str(n)) == 1) and (isPrime(n) == 1):
        return 1
    else:
        return 0
    
def rtlTruncate(n):
    a = 1
    while a:
        if (len(str(n)) > 1):
            n = int(str(n)[:-1])
            a = isPrime(n)
        if (len(str(n)) == 1):
            a = 0
    if (len(str(n)) == 1) and (isPrime(n) == 1):
        return 1
    else:
        return 0

truncatablePL = []
n = 0
for i in range(11, 10**6):
    if (isPrime(i) == 1):
        if ((ltrTruncate(i)+rtlTruncate(i)) == 2):
            truncatablePL.append(i)
            n = n+1
    if (n == 11):
        break

print(sum(truncatablePL))
