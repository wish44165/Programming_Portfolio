# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 07:36:50 2020

@author: ktpss
"""

#from scipy.misc import derivative
import numpy as np

TOL = 10**(-6)

def f(x):
    return x*((x-1)**2)

def Df(x):
    return (x-1)**2 + 2*x*(x-1)
#print(derivative(f, x, dx=1e-6))
    
def D2f(x):
    return 2*(x-1)+2*((x-1)+x)

print('--------Newton method on f--------')
x_0 = 1.5
x = 0
xt = 0
for i in range(10**3):
    xt = x
    if (i==0):
        x = x_0-(f(x_0)/Df(x_0))
        print(x)
    else:
        x = x-(f(x)/Df(x))
        print(x)
    if (np.abs(x-xt)<TOL) or (np.abs(x-xt)==0):
        break

print('\n--------Newton method on derivative of f--------')
x_0 = 1.5
x = 0
xt = 0
for i in range(10**3):
    xt = x
    if (i==0):
        x = x_0-(Df(x_0)/D2f(x_0))
        print(x)
    else:
        x = x-(Df(x)/D2f(x))
        print(x)
    if (np.abs(x-xt)<TOL) or (np.abs(x-xt)==0):
        break

print('\nSince f(1)=0 and f\'(1)=0, f has repeated root at x=1.')