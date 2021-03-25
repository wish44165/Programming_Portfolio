# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 16:24:00 2021

@author: ktpss

student number: 309653012
name: yuhsi, Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import bezier

iterNum = 7

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


def Bezier(itemsX, itemsY, t, iterNum):
    ct = 0
    l1 = itemsX
    ll1 = itemsY
    plt.plot(l1, ll1, color='b')
    while ct<iterNum:
        l2 = []
        ll2 = []
        for i in range(len(l1)-1):
            l2.append((1-t)*l1[i] + t*l1[i+1])
            ll2.append((1-t)*ll1[i] + t*ll1[i+1])
        l1 = l2
        ll1 = ll2
        plt.plot(l2, ll2, color='b')
        ct+=1
    #print(l1,ll1)
    return [(1-t)*l1[0] + t*l1[-1], (1-t)*ll1[0] + t*ll1[-1]]


with open('NACA_6712_81pts.txt', 'r') as file:
    line = file.readlines()[1:]
file.close()
data_x = [float(line[i].split(' ')[2]) for i in range(len(line))]
data_y = [float(line[i].split(' ')[-1][:-1]) for i in range(len(line))]


# verify data
with open('NACA_6712_201pts.txt', 'r') as file:
    line2 = file.readlines()[1:]
file.close()
verify_x = [float(line2[i].split(' ')[2]) for i in range(int(len(line2)/2)-1)] + [float(line2[99].split(' ')[1])] + [float(line2[100 + i].split(' ')[2]) for i in range(int(len(line2)/2)+1)]
verify_y = [float(line2[i].split(' ')[-1][:-1]) for i in range(len(line2))]


plt.figure(figsize=(10,2))
plt.scatter(data_x, data_y, color='k', label='data points')

r = np.linspace(0, 1, 100)

plt.plot(r, Lagrange(r, data_x[:int(len(data_x))//2], data_y[:int(len(data_y))//2]), color='b', label='upper Lagrange')
plt.plot(r, Lagrange(r, data_x[int(len(data_x))//2:], data_y[int(len(data_y))//2:]), color='y', label='lower Lagrange')
plt.legend()
#plt.axis([.04, .08, -.03, -.02])
plt.show()


plt.figure(figsize=(10,2))
plt.scatter(data_x, data_y, color='k', label='data points')

cs = CubicSpline(data_x[:int(len(data_x))//2][::-1], data_y[:int(len(data_y))//2][::-1], bc_type='natural')
cs2 = CubicSpline(data_x[int(len(data_x))//2:], data_y[int(len(data_y))//2:],bc_type='natural')
plt.plot(r, cs(r), color='b',  label='upper natural spline')
plt.plot(r, cs2(r), color='y',  label='lower natural spline')
plt.legend()
#plt.axis([.04, .08, -.03, -.02])
plt.show()


plt.figure(figsize=(10,2))
plt.scatter(data_x, data_y, color='k', label='data points')

for i in range(10):
    plt.scatter(Bezier(data_x, data_y, i*0.1 , iterNum)[0], Bezier(data_x, data_y, i*0.1 , iterNum)[1], color='b')
plt.plot(Bezier(data_x, data_y, i*0.1 , iterNum)[0], Bezier(data_x, data_y, i*0.1 , iterNum)[1], color='b', label='Bezier curve')
plt.legend()
#plt.axis([.04, .08, -.03, -.02])
plt.show()




# Error
# Lagrange
#EL = 0
#for i in range(len(verify_x)):
#    EL += np.abs(Lagrange([verify_x[i]], data_x, data_y)[0] - verify_y[i])
    #print(np.abs(Lagrange([verify_x[i]], data_x, data_y)[0] - verify_y[i]))
EL = np.sum([np.abs(Lagrange([verify_x[i]], data_x, data_y)[0] - verify_y[i]) for i in range(len(verify_x))])
print(EL)

# Natural Cubic Spline
verify_x1 = verify_x[:100][::-1]
verify_x2 = verify_x[101:]
verify_y1 = verify_y[:100][::-1]
verify_y2 = verify_y[101:]
EN = np.sum([np.abs(cs(verify_x1[i]) - verify_y1[i]) for i in range(len(verify_x1))]) + np.sum([np.abs(cs2(verify_x2[i]) - verify_y2[i]) for i in range(len(verify_x2))])
print(EN)

# Bezier
#EB = 0
nodes = np.asfortranarray([data_x, data_y,])
curve = bezier.Curve(nodes,degree=len(data_x)-1)
#for i in range(len(verify_x)):
#    EB += np.abs(float(curve.evaluate_hodograph(verify_x[i])[1]) - verify_y[i])
    #print(np.abs(float(curve.evaluate_hodograph(verify_x[i])[1]) - verify_y[i]))
EB = np.sum([np.abs(float(curve.evaluate_hodograph(verify_x[i])[1]) - verify_y[i]) for i in range(len(verify_x))])
print(EB)