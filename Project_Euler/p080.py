# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 22:32:07 2021

@author: ktpss
"""

import numpy as np
from decimal import*

getcontext().prec = 200
#print(np.sqrt(Decimal(2)))
totalSum = 0
#ct = 0
for item in range(2, 100):
    if int(np.sqrt(item))!=np.sqrt(item):
        #ct+=1
        totalSum+=np.sum([int(np.sqrt(item))]+[int(str(np.sqrt(Decimal(item)))[i]) for i in range(2, len(str(np.sqrt(Decimal(item)))))][:99])
print(totalSum)

"""
import math

for i in range(2, 90):
    sum+=int(str("%.100f" %(math.sqrt(2)))[i])
"""


"""
import numpy as np

# s = a**2+b
# np.sqrt(s) = a+1/(2a+1/(2a+...))

def plusA(frac, a):    #frac: [n, d]
    return [frac[0]+frac[1]*a, frac[1]]

def plus2A(frac, a):
    return [frac[0]+frac[1]*2*a, frac[1]]

def reverse(frac):
    return [frac[1], frac[0]]

def squareRoots(n):
    a = int(np.sqrt(n))
    b = n-a**2
    frac = [1, 2*a]
    for i in range(50):
        frac = reverse(plus2A(frac, a))
    frac = plusA(frac, a)
    return frac[0]/frac[1]

print(squareRoots(2))
"""


"""
decimalDigits = 100

# T: target, a: LHS, q: UP, T = r: remainder, s: supplement

# 1. find max_a s.t. a**2<T
# 2. q = a
# 3. r-=a*q
# 4. r*=100
# 5. a+=q
# 6. a*=10
# 7. find q s.t. (a+q)*q<r
# 8. a+=(2*q)
# 9.(=3.) r-=a*q

def squreRoot(T):
    sum = 0
    a = int(np.sqrt(T))
    q = a
    print(q)
    r = T
    for i in range(decimalDigits):
        r-=a*q
        r*=100
        a+=q
        a*=10
        for i in range(10):
            if (a+i)*i>r:
                break
        q = i-1
        print(q)
        sum+=q
        a+=q
    return sum

print(squreRoot(2))
"""