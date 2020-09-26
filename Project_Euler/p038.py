# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 11:07:18 2020

@author: ktpss
"""

Pl = [1,2,3,4,5,6,7,8,9]
def ispandigitalNum(n):
    l = []
    for i in range(len(n)):
        l.append(int(n[i]))
    if (sorted(l)==Pl):
        return 1
    else:
        return 0

def pandigitalNum(n):
    s = str(n)
    for i in range(2, 6):
        s = s+str(n*i)
        if (len(s)>9):
            return 0
        if (ispandigitalNum(s)==1):
            return int(s)
    if (ispandigitalNum(s) != 1):
        return 0

PN = 0

for i in range(1, 10**5):        
    if (pandigitalNum(i)>PN):
        PN = pandigitalNum(i)

print(PN)