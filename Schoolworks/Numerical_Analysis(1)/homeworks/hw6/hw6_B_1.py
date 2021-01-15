# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:34:19 2021

@author: ktpss
"""

import numpy as np
import matplotlib.pyplot as plt

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

# time steps
k = [7, 1, 1]

h = []
for a in k:
    h.append(2**(-a))

print(' ----solutions----    ----errors----')
fig, axs = plt.subplots(3, 2)
T = []
Y = []
for i in range(len(h)):
    # exact graph
    T.append(np.arange(0, 2+h[i], h[i]))
    yn = []
    for t in T[i]:
        yn.append(y(t))
    Y.append(yn)
    #plt.figure(i)
    ps, = axs[i, 0].plot(T[i], yn)
    plt.tight_layout()

# numerical graphs
yf = [y_0]
yb = [y_0]
yt = [y_0]

for i in range(len(T[0])-1):
    yf.append(forwardEuler(yf[i], T[0][i], h[0]))
    
for i in range(len(T[1])-1):
    yb.append(backwardEuler(yb[i], T[1][i], T[1][i+1], h[1]))

for i in range(len(T[2])-1):
    yt.append(trapezoidalRule(yt[i], T[2][i], T[2][i+1], h[2]))
#plt.figure(0)
pf, = axs[0,0].plot(T[0], yf)
axs[0,0].set_title('Forward Euler method, k = 7')
axs[0,0].set_xlabel('t')
axs[0,0].set_ylabel('y')
axs[0,0].legend([ps, pf], ["exact", "FM"], loc = 'best')
#plt.figure(1)
pb, = axs[1,0].plot(T[1], yb)
axs[1,0].set_title('Backward Euler method, k = 1')
axs[1,0].set_xlabel('t')
axs[1,0].set_ylabel('y')
axs[1,0].legend([ps, pb], ["exact", "BM"], loc = 'best')
#plt.figure(2)
pt, = axs[2,0].plot(T[2], yt)
axs[2,0].set_title('Trapezoidal Rule, k = 1')
axs[2,0].set_xlabel('t')
axs[2,0].set_ylabel('y')
axs[2,0].legend([ps, pt], ["exact", "TR"], loc = 'best')

# global error
ef = []
eb = []
et = []
for i in range(len(T[0])):
    ef.append(np.abs(yf[i] - Y[0][i]))

    
for i in range(len(T[1])):
    eb.append(np.abs(yb[i] - Y[1][i]))
    
for i in range(len(T[2])):
    et.append(np.abs(yt[i] - Y[2][i]))

E = [ef, eb, et]

Title = ['Forward Euler method, k = 7', 'Backward Euler method, k = 1', 'Trapezoidal Rule, k = 1']

for i in range(3, 6):
    axs[i-3, 1].plot(T[i-3], E[i-3])
    axs[i-3, 1].set_title(Title[i-3])
    axs[i-3, 1].set_xlabel('t')
    axs[i-3, 1].set_ylabel('y')
    plt.tight_layout()