# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 13:22:00 2020

@author: ktpss
"""

import numpy as np

N = 10**4

def pentagon(n):
    return n*(3*n-1)/2

def ipentagon(pn):
    n = (1+np.sqrt(1+24*pn))/6
    if (n == int(n)):
        return 1
    else:
        return 0

#3*n**2 - n = 2*pn
#3*n**2 - n - 2*pn = 0
#n = (1+-np.sqrt((-1)**2-4*3*(-2*pn)))/2*3 = (1+-np.sqrt(1+24*pn))/6
#n = (1+np.sqrt(1+24*pn))/6

# a*(3*a-1)/2
# b*(3*b-1)/2
# (3*a**2-a - (3*b**2-b))/2 = c*(3*c-1)/2
# 3*(a**2-b**2)-(a-b) = c*(3*c-1)
# (3*a**2-a + (3*b**2-b))/2 = d*(3*d-1)/2
# 3*(a**2+b**2)-(a+b) = d*(3*d-1)

D = 10**100
for i in range(1, N):
    for j in range(i+1, N):
        d = pentagon(j)-pentagon(i)
        if (d>D):
            break
        if (ipentagon(d)==1):
            s = pentagon(i)+pentagon(j)
            if (ipentagon(s)==1):
                if (d<D):
                    D = d
                    print(i, j, D)

print(D)