# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 13:37:44 2021

@author: ktpss
"""

def plusOne(frac):    #frac: [n, d]
    return [frac[0]+frac[1], frac[1]]

def minusOne(frac):
    return [frac[0]-frac[1], frac[1]]

def plusTwo(frac):
    return [frac[0]+frac[1]*2, frac[1]]

def reciprocal(frac):
    return [frac[1], frac[0]]

ct = 0
checkL = [1,2]
for i in range(1000):
    checkL = plusOne(checkL)
    #print(checkL)
    if len(str(checkL[0]))!=len(str(checkL[1])):
        ct+=1
    checkL = reciprocal(plusTwo(minusOne(checkL)))
    
    
print(ct)
    
"""
print(plusOne([1,2]))

print(plusOne(reciprocal(plusTwo([1,2]))))

print(plusOne(reciprocal(plusTwo(reciprocal(plusTwo([1,2]))))))
"""