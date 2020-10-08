# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 10:36:46 2020

@author: ktpss
"""

import numpy as np
#from scipy.misc import derivatives

def f_a(x):
    return 2*x*np.cos(2*x) - (x-2)**2

def Df_a(x):
    return 2*(np.cos(2*x)+x*2*(-np.sin(2*x))) - 2*(x-2)

def f_b(x):
    return np.e**x - 3*x**2

def Df_b(x):
    return np.e**x - 6*x

"""
x = 10**2
print(derivative(f_a, x, dx=1e-6))
print(Df_a(x))
print(derivative(f_b, x, dx=1e-6))
print(Df_b(x))
"""

def newtonMethod_a(p_0, TOL):
    x2 = 10**12
    x = p_0
    while 1:
        x = x-(f_a(x)/Df_a(x))
        if (np.abs(x2-x)<TOL):
            return x
        else:
            x2 = x
            
def newtonMethod_b(p_0, TOL):
    x2 = 10**12
    x = p_0
    while 1:
        x = x-(f_b(x)/Df_b(x))
        if (np.abs(x2-x)<TOL):
            return x
        else:
            x2 = x
    
def secantMethod_a(p_0, p_1, TOL):
    x0 = p_0
    x1 = p_1
    x2 = 10**10
    while 1:
        temp = x1
        x2 = x1 - f_a(x1)*((x1-x0)/(f_a(x1)-f_a(x0)))
        if (np.abs(x2-x1)<TOL):
            return x2
        x1 = x2
        x0 = temp
        
def secantMethod_b(p_0, p_1, TOL):
    x0 = p_0
    x1 = p_1
    x2 = 10**10
    while 1:
        temp = x1
        x2 = x1 - f_b(x1)*((x1-x0)/(f_b(x1)-f_b(x0)))
        if (np.abs(x2-x1)<TOL):
            return x2
        x1 = x2
        x0 = temp

print('------------------------(a)------------------------')
print('----------------Newton Method----------------')
p_0 = input('input p0 in range [2,3]: ')    #2.5
TOL = input('input TOL: ')    #10**(-6)
print('x = ', newtonMethod_a(float(p_0), float(TOL)))
p_0 = input('input p0 in range [3,4]: ')    #3.5
TOL = input('input TOL: ')    #10**(-6)
print('x = ', newtonMethod_a(float(p_0), float(TOL)))

print('\n----------------Secant Method----------------')
p_0 = input('input p0 in range [2,3]: ')    #2
p_1 = input('input p1 in range [2,3]: ')    #3 
TOL = input('input TOL: ')    #10**(-6)
print('x = ', secantMethod_a(float(p_0), float(p_1), float(TOL)))
p_0 = input('input p0 in range [3,4]: ')    #3
p_1 = input('input p1 in range [3,4]: ')    #4
TOL = input('input TOL: ')    #10**(-6)
print('x = ', secantMethod_a(float(p_0), float(p_1), float(TOL)))

print('\n\n------------------------(b)------------------------')
print('----------------Newton Method----------------')
p_0 = input('input p0 in range [0,1]: ')    #0.5
TOL = input('input TOL: ')    #10**(-6)
print('x = ', newtonMethod_b(float(p_0), float(TOL)))
p_0 = input('input p0 in range [3,4]: ')    #3.5
TOL = input('input TOL: ')    #10**(-6)
print('x = ', newtonMethod_b(float(p_0), float(TOL)))
p_0 = input('input p0 in range [6,7]: ')    #6.5
TOL = input('input TOL: ')    #10**(-6)
print('x = ', newtonMethod_b(float(p_0), float(TOL)))

print('\n----------------Secant Method----------------')
p_0 = input('input p0 in range [0,1]: ')    #0
p_1 = input('input p1 in range [0,1]: ')    #1 
TOL = input('input TOL: ')    #10**(-6)
print('x = ', secantMethod_b(float(p_0), float(p_1), float(TOL)))
p_0 = input('input p0 in range [3,4]: ')    #3
p_1 = input('input p1 in range [3,4]: ')    #4 
TOL = input('input TOL: ')    #10**(-6)
print('x = ', secantMethod_b(float(p_0), float(p_1), float(TOL)))
p_0 = input('input p0 in range [6,7]: ')    #6
p_1 = input('input p1 in range [6,7]: ')    #7 
TOL = input('input TOL: ')    #10**(-6)
print('x = ', secantMethod_b(float(p_0), float(p_1), float(TOL)))