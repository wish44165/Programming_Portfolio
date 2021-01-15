# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 19:08:59 2021

@author: ktpss
"""

import numpy as np
import matplotlib.pyplot as plt

mu = 0.1
T = 0.1
y_0 = 0.1

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
k = []
for kk in range(1,10):
    k.append(kk)

H = []
for a in k:
    H.append(10**(-a))


E = [[], [], []]


#fig, axs = plt.subplots(3, 2)

Rh = []
Ref = []
Reb = []
Ret = []

h1 = 0
h2 = 0
ef1 = 0
ef2 = 0
eb1 = 0
eb2 = 0
et1 = 0
et2 = 0
for h in H:


    # exact graph
    tn = np.arange(0, T+h, h)
    yn = []
    for t in tn:
        yn.append(y(t))
    #ps, = axs[i, 0].plot(T[i], yn)
    #plt.tight_layout()
    
    # numerical graphs
    yf = [y_0]
    yb = [y_0]
    yt = [y_0]
    
    for i in range(len(tn)-1):
        yf.append(forwardEuler(yf[i], tn[i], h))
        
    for i in range(len(tn)-1):
        yb.append(backwardEuler(yb[i], tn[i], tn[i+1], h))
    
    for i in range(len(tn)-1):
        yt.append(trapezoidalRule(yt[i], tn[i], tn[i+1], h))

    # global error
    ef = []
    eb = []
    et = []
    for i in range(len(yn)):
        ef.append(np.abs(yf[i] - yn[i]))
        
    for i in range(len(yn)):
        eb.append(np.abs(yb[i] - yn[i]))
        
    for i in range(len(yn)):
        et.append(np.abs(yt[i] - yn[i]))
    
    E[0].append(np.max(ef))
    E[1].append(np.max(eb))
    E[2].append(np.max(et))
    
    
    # Ratio
    h1 = h2
    h2 = h
    ratio_h = h1/h2
    Rh.append(ratio_h)
    
    ef1 = ef2
    ef2 = np.max(ef)
    ratio_ef = ef1/ef2
    Ref.append(ratio_ef)

    eb1 = eb2
    eb2 = np.max(eb)
    ratio_eb = eb1/eb2
    Reb.append(ratio_eb)

    et1 = et2
    et2 = np.max(et)
    ratio_et = et1/et2
    Ret.append(ratio_et)

Ord = [[],[],[]]
for i in range(len(Rh)):
    Ord[0].append(np.log(Ref[i])/np.log(Rh[i]))
    Ord[1].append(np.log(Reb[i])/np.log(Rh[i]))
    Ord[2].append(np.log(Ret[i])/np.log(Rh[i]))


fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
#fig.suptitle('Horizontally stacked subplots')
ax1.loglog(H, Ref, 'bo', markersize=3)
ax1.set_xlabel('h')
ax1.set_ylabel('Error')
ax1.set_title('Forward Euler method')
ax2.loglog(H, Reb, 'bo', markersize=3)
ax2.set_xlabel('h')
ax2.set_ylabel('Error')
ax2.set_title('Backward Euler method')
ax3.loglog(H, Ret, 'bo', markersize=3)
ax3.set_xlabel('h')
ax3.set_ylabel('Error')
ax3.set_title('Trapezoidal Rule')
plt.tight_layout()
