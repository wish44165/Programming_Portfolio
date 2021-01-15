# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 22:08:14 2021

@author: ktpss
"""

import numpy as np
import matplotlib.pyplot as plt

# exact sol.
def y(t):
    return 4*np.e**(-260*t) + t**3

# numerical sol.
def f(t, y):
    return -260*(y-t**3) + 3*t**2

def EM(y, h, f):
    return y + h*f


# exact
h1 = 0.01
tn = np.arange(0, 2+h1, h1)
yn = []
for t in tn:
    yn.append(y(t))
plt.figure(1)
plt.plot(tn,yn)

# numerical
# FE
yf = [4]
for i in range(np.size(tn)-1):
    yf.append(EM(yf[i], h1, f(tn[i], yf[i])))
plt.figure(2)
plt.plot(tn, yf)

