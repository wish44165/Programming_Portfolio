# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 21:25:31 2021

@author: ktpss
"""

goalNum = 10**4

def palindromic(x):
    x = str(x)
    checkNum = int(len(str(x))/2)
    ct = 0
    for i in range(checkNum):
        if str(x)[i] == str(x)[-i-1]:
            ct+=1
    if ct==checkNum:
        return 1
    else:
        return 0

def ReverseAdd(x):
    return int(str(x)[::-1])+x

"""
#check
x = 10677
for i in range(53):
    x = ReverseAdd(x)

print(x)
"""

LychrelNum = 0
for i in range(goalNum):
    x = i
    for j in range(50):
        x = ReverseAdd(x)
        if palindromic(x)==1:
            break
        if j==49:
            LychrelNum+=1

print(LychrelNum)