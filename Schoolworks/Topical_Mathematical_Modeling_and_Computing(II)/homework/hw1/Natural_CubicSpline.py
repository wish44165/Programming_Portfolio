# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:00:32 2021

@author: ktpss

student number: 309653012
name: yuhsi, Chen
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

perNum = [20, 12, 6, 2]

def absoluteError(x_c, y_c):
    return np.abs(np.sqrt(x_c**2+y_c**2)-1)


"""
x = np.array([-1.5, -.2, 1, 5, 10, 15, 20])
y = np.array([-1.2, 0, .5, 1, 1.2, 2, 1])
plt.scatter(x,y)
"""

"""
x = np.array([0, 1, 2, 3, 4])
y = np.array([21, 24, 24, 18, 16])

A = np.array([[4, 3, 2], [-2, 2, 3], [3, -5, 2]])
B = np.array([25, -10, -4])
X = np.linalg.inv(A).dot(B)

print(X)
"""

"""
A = np.array([[4, 3, 2], [-2, 2, 3], [3, -5, 2]])
B = np.array([25, -10, -4])
X2 = np.linalg.solve(A,B)

print(X2)
"""

#Example
x = np.array([0, 1, 2, 3, 4])
y = np.array([21, 24, 24, 18, 16])

A = np.zeros((16, 16))
A[0][3] = 1
for i in range(1,5):
    A[1][i-1] = 1
    A[2][i+3] = 1
    A[3][i+3] = 2**(4-i)
    A[4][i+7] = 2**(4-i)
    A[5][i+7] = 3**(4-i)
    A[6][i+11] = 3**(4-i)
    A[7][i+11] = 4**(4-i)
    A[8][i-1] = 4-i
    A[8][i+3] = i-4
A[9][4] = 12
A[9][5] = 4
A[9][6] = 1
A[9][8] = -12
A[9][9] = -4
A[9][10] = -1
A[10][8] = 27
A[10][9] = 6
A[10][10] = 1
A[10][12] = -27
A[10][13] = -6
A[10][14] = -1
A[11][0] = 6
A[11][1] = 2
A[11][4] = -6
A[11][5] = -2
A[12][4] = 12
A[12][5] = 2
A[12][8] = -12
A[12][9] = -2
A[13][8] = 18
A[13][9] = 2
A[13][12] = -18
A[13][13] = -2
A[14][1] = 2
A[15][12] = 24
A[15][13] = 2
    
B = np.zeros((16,1))
B[0][0] = 21
B[1][0] = 24
B[2][0] = 24
B[3][0] = 24
B[4][0] = 24
B[5][0] = 18
B[6][0] = 18
B[7][0] = 16
X = np.linalg.solve(A,B)

#print(A)
#print(B)
#print(X)



def f1(x):
    return X[0][0]*x**3 + X[1][0]*x**2 + X[2][0]*x+X[3][0]
def f2(x):
    return X[4][0]*x**3 + X[5][0]*x**2 + X[6][0]*x+X[7][0]
def f3(x):
    return X[8][0]*x**3 + X[9][0]*x**2 + X[10][0]*x+X[11][0]
def f4(x):
    return X[12][0]*x**3 + X[13][0]*x**2 + X[14][0]*x+X[15][0]

x1 = np.linspace(0,1)
x2 = np.linspace(1,2)
x3 = np.linspace(2,3)
x4 = np.linspace(3,4)
plt.figure(figsize=(6,6))
plt.scatter(x,y,label='five data points',color='k')
for t, v in zip(x, y):
    plt.text(t, v, '({}, {})'.format(t, v))

plt.plot(x1,f1(x1),label='first part')
plt.plot(x2,f2(x2),label='second part')
plt.plot(x3,f3(x3),label='third part')
plt.plot(x4,f4(x4),label='forth part')

plt.title('Natural Cublic Spline Demo')
plt.legend()
plt.show()




# circle
theta = np.linspace(0, 2*np.pi, 100)
radius = 1
a = radius*np.cos(theta)
b = radius*np.sin(theta)

for i in range(len(perNum)):
    
    fig = plt.figure(figsize=(6,6))
    
    # upper circle points
    x = [a[:len(a)//2][i] for i in range(0, len(a[:len(a)//2]), perNum[i])][::-1]
    y = [b[:len(b)//2][i] for i in range(0, len(b[:len(b)//2]), perNum[i])][::-1]
    
    #lower ciricle points
    x2 = [a[len(a)//2:][i] for i in range(0, len(a[len(a)//2:]), perNum[i])]
    y2 = [b[len(b)//2:][i] for i in range(0, len(b[len(b)//2:]), perNum[i])]
    
    #CubicSpline package
    xs = np.arange(-1, 1, 0.01)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, y, 'o', color='r', label='upper interpolated points')
    ax.plot(x2, y2, 'o', color='k', label='lower interpolated points')
    
    cs = CubicSpline(x,y,bc_type='natural')
    ax.plot(xs, cs(xs), label='upper natural spline')
    
    cs2 = CubicSpline(x2,y2,bc_type='natural')
    ax.plot(xs, cs2(xs), label='lower natural spline')
    #ax.plot(xs, cs(xs, 1), label="S'")
    #ax.plot(xs, cs(xs, 2), label="S''")
    #ax.plot(xs, cs(xs, 3), label="S'''")
    #ax.set_xlim(-0.5, 9.5)
    
    #estimate Error
    E = 0
    tE = np.linspace(-.95, .95, 20)
    AE1 = CubicSpline(x, y, bc_type='natural')(tE)
    AE2 = CubicSpline(x2, y2, bc_type='natural')(tE)
    for k in range(len(tE)):
        E+=absoluteError(tE[k], AE1[k])
        E+=absoluteError(tE[k], AE2[k])
    
    ax.legend()
    plt.title('CubicSpline, N=%s' %str(int(len(a)/perNum[i])))
    plt.show()
    
    print(E) #Absolute Error