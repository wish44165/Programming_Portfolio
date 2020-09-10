# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 21:01:44 2020

@author: ktpss
"""

import copy

f = open('p081_matrix.txt', 'r')

L = []
for x in f:
    l = x.split(',', len(x))
    for i in range(len(l)):
        l[i] = int(l[i])
    L.append(l)

f.close()

# a[0], b[0] a[1], c[0] b[1] a[2], ...
# L[0][0], L[1][0] L[0][1], L[2][0] L[1][1] L[0][2], ..., L[79][0]-L[0][79],

# upper triangle
Ut = []

for i in range(len(L)):
    diag = []
    for j in range(i+1):
        diag.append(L[i-j][j])
    Ut.append(diag)

minUt = copy.deepcopy(Ut)    # use copy.copy will also change the value in Ut

for i in range(len(Ut)):
    for j in range(i+1):
        if (i==0):
            minUt[i][j] = minUt[0][0]
        else:
            if (j==0):
                minUt[i][j] = minUt[i-1][j]+minUt[i][j]
            elif (j==i):
                minUt[i][j] = minUt[i-1][j-1]+minUt[i][j]
            else:
                minUt[i][j] = min(minUt[i-1][j-1], minUt[i-1][j])+minUt[i][j]

# L[79][1]-L[1][79], L[79][2]-L[2][79], ..., L[79][78] L[78][79], L[79][79]
# 79 1 78 2 77 3 - 1 79

# lower triangle
Lt = []

for i in range(len(L)-1):
    diag = []
    for j in range(len(L)-i-1):
        diag.append(L[len(L)-1-j][i+j+1])
    Lt.append(diag)

minLt = [minUt[-1]]
for i in range(len(Lt)):
    minLt.append(Lt[i])

for i in range(1, len(minLt)):
    for j in range(len(minLt)-i):
        minLt[i][j] = min(minLt[i-1][j], minLt[i-1][j+1])+minLt[i][j]

print(minLt[-1][-1])