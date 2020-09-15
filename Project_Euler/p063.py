# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:36:55 2020

@author: ktpss
"""

N = 10**3

ct = 0
for n in range(1, 10):
    for m in range(1, N):
        l = len(str(n**m))
        if (l > m):
            break
        elif (l==m):
            ct = ct+1

print(ct)