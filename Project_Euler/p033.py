# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 11:52:55 2020

@author: ktpss
"""

def digitCancel(n, d):
    if (int(str(n)[1])+int(str(d)[1]) == 0):
        return 0
    else:
        nl = []
        dl = []
        for i in range(2):
            nl.append(str(n)[i])
            dl.append(str(d)[i])
        for a in nl:
            if a in dl:
                nl.remove(a)
                dl.remove(a)
        if (len(nl) == 1):
            n2 = int(nl[0])
            d2 = int(dl[0])
            if (d2 < n2):
                return 0
            if ((n/d) == (n2/d2)):
                return 1
            else:
                return 0
        else:
            return 0
        
def gcd(a, b):
    if (b == 0):
        return a
    else:
        return gcd(b, a%b)
   
N = []
D = []
for n in range(10, 100):
    for d in range(n+1, 100):
        if (digitCancel(n, d) == 1):
            N.append(n)
            D.append(d)

PN = 1
PD = 1
for i in range(len(D)):
    PN = PN*N[i]
    PD = PD*D[i]
    
A = PD/gcd(PN, PD)
print(A)