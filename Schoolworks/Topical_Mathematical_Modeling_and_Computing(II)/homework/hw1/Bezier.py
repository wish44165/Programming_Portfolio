# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:43:00 2021

@author: ktpss

student number: 309653012
name: yuhsi, Chen
"""

import numpy as np
import matplotlib.pyplot as plt

perNum = [20, 12, 6, 2]
iterNum = [int(100/a)-1 for a in perNum]

def Bezier(itemsX, itemsY, t, iterNum):
    ct = 0
    l1 = itemsX
    ll1 = itemsY
    plt.plot(l1, ll1)
    while ct<iterNum:
        l2 = []
        ll2 = []
        for i in range(len(l1)-1):
            l2.append((1-t)*l1[i] + t*l1[i+1])
            ll2.append((1-t)*ll1[i] + t*ll1[i+1])
        l1 = l2
        ll1 = ll2
        plt.plot(l2, ll2)
        ct+=1
    #print(l1,ll1)
    return [(1-t)*l1[0] + t*l1[-1], (1-t)*ll1[0] + t*ll1[-1]]

def BezierC(itemsX, itemsY, t, iterNum):
    ct = 0
    l1 = itemsX
    ll1 = itemsY
    #plt.plot(l1, ll1)
    while ct<iterNum:
        l2 = []
        ll2 = []
        for i in range(len(l1)-1):
            l2.append((1-t)*l1[i] + t*l1[i+1])
            ll2.append((1-t)*ll1[i] + t*ll1[i+1])
        l1 = l2
        ll1 = ll2
        #plt.plot(l2, ll2)
        ct+=1
    #print(l1,ll1)
    return [(1-t)*l1[0] + t*l1[-1], (1-t)*ll1[0] + t*ll1[-1]]

def absoluteError(x_c, y_c):
    return np.abs(np.sqrt(x_c**2+y_c**2)-1)

# Demo
"""
for i in range(10):
    plt.scatter(Bezier([1,2,3], [1, 2, 1],i*0.1 , 2)[0], Bezier([1,2,3], [1, 2, 1],i*0.1 , 2)[1], color='b', s=100)
plt.title('3 data points')
plt.show()
for i in range(10):
    plt.scatter(Bezier([1,2,3,4], [1, 2, 2,1],i*0.1 , 2)[0], Bezier([1,2,3,4], [1, 2, 2,1],i*0.1 , 2)[1], color='b', s=100)
plt.title('4 data points')
plt.show()
"""
plt.figure(figsize=(6,6))
xd = np.array([0, 1, 2, 3, 4])
yd = np.array([21, 24, 24, 18, 16])
plt.scatter(xd, yd, label='five data points', color='k')
for t, v in zip(xd, yd):
    plt.text(t, v, '({}, {})'.format(t, v))
for i in range(10):
    plt.scatter(Bezier(xd, yd, i*0.1 , 3)[0], Bezier(xd, yd, i*0.1 , 3)[1], color='b', s=100)
plt.scatter(Bezier(xd, yd, i*0.1 , 3)[0], Bezier(xd, yd, i*0.1 , 3)[1], color='b', s=100, label='estimated points')
plt.title('Bezier Curves Demo')
plt.legend()
plt.show()


# circle
theta = np.linspace(0, 2*np.pi, 100)
radius = 1

# upper circle
a = radius*np.cos(theta)
b = radius*np.sin(theta)

for k in range(len(perNum)):
    fig = plt.figure(figsize=(6,6))
    plt.plot(a, b, color='g', label='circle')
    
    
    plt.scatter([a[i] for i in range(0, len(a), perNum[k])], [b[i] for i in range(0, len(b), perNum[k])], color='k', label='data points')
    
    #plt.scatter(a,b)
    for j in range(10):
        #Bezier([a[i] for i in range(0, len(a), perNum)], [b[i] for i in range(0, len(b), perNum)], j*0.1, iterNum)
        plt.scatter(Bezier([a[i] for i in range(0, len(a), perNum[k])], [b[i] for i in range(0, len(b), perNum[k])], j*0.1, iterNum[k])[0],Bezier([a[i] for i in range(0, len(a), perNum[k])], [b[i] for i in range(0, len(b), perNum[k])], j*0.1, iterNum[k])[1],color='b',s=100)
    plt.scatter(Bezier([a[i] for i in range(0, len(a), perNum[k])], [b[i] for i in range(0, len(b), perNum[k])], j*0.1, iterNum[k])[0],Bezier([a[i] for i in range(0, len(a), perNum[k])], [b[i] for i in range(0, len(b), perNum[k])], j*0.1, iterNum[k])[1],color='b',s=100, label='estimated points')
        
    #plt.axis([-2, 2, -2, 2])
    plt.title('Bezier, N=%s' %str(int(len(a)/perNum[k])))
    plt.legend()
    plt.show()
    

# https://spencermortensen.com/articles/bezier-circle/
c = 0.551915024494
x1 = [0,c,1,1]
y1 = [1,1,c,0]
x2 = [1,1,c,0]
y2 = [0,-c,-1,-1]
x3 = [0,-c,-1,-1]
y3 = [-1,-1,-c,0]
x4 = [-1,-1,-c,0]
y4 = [0,c,1,1]
X = [x1,x2,x3,x4]
Y = [y1,y2,y3,y4]


fig = plt.figure(figsize=(6,6))
for i in range(len(X)-1):
    plt.scatter(X[i],Y[i],color='k')
    for t, v in zip(X[i],Y[i]):
        plt.text(t, v, '({}, {})'.format(t, v))
plt.scatter(X[3],Y[3],color='k',label='data points')
for t, v in zip(X[3],Y[3]):
    plt.text(t, v, '({}, {})'.format(t, v))
    
E = 0
for j in range(len(X)):
    for i in range(10):
        plt.scatter(BezierC(X[j], Y[j], i*0.1, 3)[0], BezierC(X[j], Y[j], i*0.1, 3)[1],color='b',s=40)
        E+=absoluteError(BezierC(X[j], Y[j], i*0.1, 3)[0], BezierC(X[j], Y[j], i*0.1, 3)[1])
plt.scatter(BezierC(X[j], Y[j], i*0.1, 3)[0], BezierC(X[j], Y[j], i*0.1, 3)[1],color='b',s=40, label='estimated points')

plt.title('Bezier curves, N=16')
plt.legend()
plt.show()

#Absolute Error for N=16
print(E)