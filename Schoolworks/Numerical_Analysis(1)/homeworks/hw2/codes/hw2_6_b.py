# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:59:41 2020

@author: ktpss
"""

import numpy as np
import matplotlib.pyplot as plt

# l0 = [x0,y0,z0,u0,v0,..., x1,y1,z1,u1,v1,..., ...]
# lp = [x0+,y0+,z0+,u0+,v0+,..., x1+,y1+,z1+,u1+,v1+,..., ...]
# lm = [x1-,y1-,z1-,u1-,v1-,..., x2-,y2-,z2-,u2-,v2-,..., ...]

def Bézier(dim, n, l0, lp, lm):    # dimension, number of data, data, left guidepoint, right guidepoint
    l = []
    for i in range(n):
        for j in range(i*dim, (i+1)*dim):    
            l.append(l0[j])    # a0,b0,c0,...
            l.append(3*(lp[j]-l0[j]))    # a1,b1,c1,...
            l.append(3*(l0[j]+lm[j]-2*lp[j]))    # a2,b2,c2,...
            l.append(l0[j+dim]-l0[j]+3*lp[j]-3*lm[j])    # a3,b3,c3,...
    return l

print('\n--------result of algorithm for Bézier curve--------\n')
Bl = Bézier(2, 1, [1,1,6,2], [1.5,1.25], [7,3])
print('coefficients: ', Bl)
print('x(t) = ((%.4f*t + %.4f)*t + %.4f)*t + %.4f' %(Bl[3], Bl[2], Bl[1], Bl[0]))
print('y(t) = ((%.4f*t + %.4f)*t + %.4f)*t + %.4f' %(Bl[-1], Bl[-2], Bl[-3], Bl[-4]))


t = np.arange(-1, 2, 0.01)
plt.show()
plt.plot((((-11.5000*t + 15.0000)*t + 1.5000)*t + 1.0000), (((-4.2500*t + 4.5000)*t + 0.7500)*t + 1.0000))
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.title('t in range (-1,2)')

t = np.arange(-.7, 1.5, 0.01)
plt.show()
plt.plot((((-11.5000*t + 15.0000)*t + 1.5000)*t + 1.0000), (((-4.2500*t + 4.5000)*t + 0.7500)*t + 1.0000))
plt.xlabel('x(t)')
plt.ylabel('y(t)')
plt.title('t in range (-0.7,1.5)')