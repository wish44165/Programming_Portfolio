# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 20:45:28 2021

@author: ktpss

student number: 309653012
name: yuhsi, Chen
"""

import numpy as np
import matplotlib.pyplot as plt

perNum = [20, 12, 6, 2]

def Lagrange(rangeL, x, y):
    l = []
    for a in rangeL:
        yprod = 0
        for i in range(len(y)):
            prod = 1
            for j in range(len(x)):
                if j!=i:
                    prod*=(a-x[j])/(x[i]-x[j])
            yprod += y[i]*prod
        l.append(yprod)
    return l

def absoluteError(x_c, y_c):
    return np.abs(np.sqrt(x_c**2+y_c**2)-1)

#Demo
xd = np.array([0, 1, 2, 3, 4])
yd = np.array([21, 24, 24, 18, 16])
r = np.arange(0,4,0.01)
plt.figure(figsize=(6,6))
plt.scatter(xd,yd,label='five data points',color='k')
for t, v in zip(xd, yd):
    plt.text(t, v, '({}, {})'.format(t, v))
plt.title('Lagrange Demo')
plt.plot(r, Lagrange(r,xd,yd),label='Lagrange interpolation')
plt.legend()
plt.show()
    
# circle
theta = np.linspace(0, 2*np.pi, 100)
radius = 1
a = radius*np.cos(theta)
b = radius*np.sin(theta)

for i in range(len(perNum)):
    fig = plt.figure(figsize=(6,6))
    plt.plot(a, b, color='g', label='circle')
    
    # Lagrange
    x = [a[:len(a)//2][i] for i in range(0, len(a[:len(a)//2]), perNum[i])]
    y = [b[:len(b)//2][i] for i in range(0, len(b[:len(b)//2]), perNum[i])]
    plt.scatter(x, y, color='r', label='upper interpolated points')
    
    x2 = [a[len(a)//2:][i] for i in range(0, len(a[len(a)//2:]), perNum[i])]
    y2 = [b[len(b)//2:][i] for i in range(0, len(b[len(b)//2:]), perNum[i])]
    plt.scatter(x2, y2, color='k', label='lower interpolated points')
    
    t = np.linspace(np.min([np.min(x),np.min(x2)]), np.max([np.max(x),np.max(x2)]), 200)
    
    #estimate Error
    E = 0
    tE = np.linspace(-.95, .95, 20)
    AE1 = Lagrange(tE, x, y)
    AE2 = Lagrange(tE, x2, y2)
    for k in range(len(tE)):
        E+=absoluteError(tE[k], AE1[k])
        E+=absoluteError(tE[k], AE2[k])
    
    
    plt.plot(t, Lagrange(t, x, y), color='b', label='upper Lagrange')
    plt.plot(t, Lagrange(t, x2, y2), color='y', label='lower Lagrange')
    #plt.axis([-2, 2, -2, 2])
    plt.legend()
    plt.title('Lagrange Interpolation, N=%s' %str(int(len(a)/perNum[i])))
    plt.show()
    
    print(E) #Absolute Error