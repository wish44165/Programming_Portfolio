# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 22:58:56 2021

@author: ktpss
"""

import numpy as np
import matplotlib.pyplot as plt

# variables
mu = 260
T = 2
y_0 = 4

# exact solution
def y(t):
    return 4*np.e**(-260*t) + t**3

# numerical solution
def forwardEuler(y, t, h):
    return (1-h*mu)*y + h*mu*t**3 + 3*h*t**2

def backwardEuler(y, t_i, t_j, h):
    return (y + h*mu*t_i**3 + 3*h*t_j**2) / (1 + h*mu)

def trapezoidalRule(y, t_i, t_j, h):
    return ((1 - h*mu/2)*y + (h/2)*(mu*(t_i**3+t_j**3) + 3*(t_i**2+t_j**2))) / (1 + h*mu/2)

# exact graph
h = 0.001
tn = np.arange(0, 2+h, h)
yn = []
for t in tn:
    yn.append(y(t))
plt.figure(1)
plt.plot(tn, yn)

# numerical graphs
yf = [y_0]
yb = [y_0]
yt = [y_0]
for i in range(np.size(tn)-1):
    yf.append(forwardEuler(yf[i], tn[i], h))
    yb.append(backwardEuler(yb[i], tn[i], tn[i+1], h))
    yt.append(trapezoidalRule(yt[i], tn[i], tn[i+1], h))
plt.figure(2)
plt.plot(tn, yf)
plt.figure(3)
plt.plot(tn, yb)
plt.figure(4)
plt.plot(tn, yt)
   

