# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 22:42:43 2020

@author: ktpss
"""

import numpy as np

tolerateValue = 10**(-3)
N = 21**(1/3)

#1
def p1(n):
    p = 1
    print('01 : ', p, ' ', np.abs(p-N)/N)
    ct = 0
    while ct<n:
        p = (20*p+21/p**2)/21
        if (ct<8):
            print(str(0)+str(ct+2),': ', p, ' ', np.abs(p-N)/N)
        else:
            print(ct+2,': ', p, ' ', np.abs(p-N)/N)
        if (np.abs(p-N)/N < tolerateValue):
            break
        ct = ct+1
            

#2
def p2(n):
    p = 1
    print('01 : ', p, ' ', np.abs(p-N)/N)
    ct = 0
    while ct<n:
        p = p - (p**3-21)/(3*p**2)
        if (ct<8):
            print(str(0)+str(ct+2),': ', p, ' ', np.abs(p-N)/N)
        else:
            print(ct+2,': ', p, ' ', np.abs(p-N)/N)
        if (np.abs(p-N)/N < tolerateValue):
            break
        ct = ct+1

#3
def p3(n):
    p = 1
    print('01 : ', p, ' ', np.abs(p-N)/N)
    ct = 0
    while ct<n:
        p = p - (p**4-21*p)/(p**2-21)
        if (ct<8):
            print(str(0)+str(ct+2),': ', p, ' ', np.abs(p-N)/N)
        else:
            print(ct+2,': ', p, ' ', np.abs(p-N)/N)
        if (np.abs(p-N)/N < tolerateValue):
            break
        ct = ct+1


#4
def p4(n):
    p = 1
    print('01 : ', p, ' ', np.abs(p-N)/N)
    ct = 0
    while ct<n:
        p = (21/p)**(1/2)
        if (ct<8):
            print(str(0)+str(ct+2),': ', p, ' ', np.abs(p-N)/N)
        else:
            print(ct+2,': ', p, ' ', np.abs(p-N)/N)
        if (np.abs(p-N)/N < tolerateValue):
            break
        ct = ct+1
        
n = 40
print('\ntolerate value =', tolerateValue)
print('\n------------------------(a)------------------------')
p1(n)
print('\n------------------------(b)------------------------')
p2(n)
print('\n------------------------(c)------------------------')
p3(n)
print('\n------------------------(d)------------------------')
p4(n)