# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 02:36:15 2022

@author: ktpss
"""


import numpy as np

np.random.seed(13)

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
         (I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)

def example():
    dt = 1.0/30
    valueQ = 0.1
    valueR = 0.01
    F = np.array([[  1,  0, dt,  0],
                  [  0,  1,  0, dt],
                  [  0,  0,  1,  0],
                  [  0,  0,  0,  1]])
    H = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0]]).reshape(2, 4)
    Q = np.array([[valueQ, 0.0, 0.0, 0.0],
                  [0.0, valueQ, 0.0, 0.0],
                  [0.0, 0.0, valueQ, 0.0],
                  [0.0, 0.0, 0.0, valueQ]])
    R = np.array([[valueR, 0.0],
                 [0.0, valueR]]).reshape(2, 2)
    
    
    # measurement
    m_x = np.linspace(-10, 10, 100)
    m_y = - (m_x**2 + 2*m_x - 2)  + np.random.normal(0, 2, 100)
    measurements = np.array([[xi, yi] for i, (xi, yi) in enumerate(zip(m_x, m_y))])
    
    kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
    predictions = []
    gains = []
    prepredictions = []
    for z in measurements:
        predictions.append(list(np.dot(H,  kf.predict()).reshape(1,2)[0]))
        #print('before', kf.x)
        #print('====',[float(np.dot(H,  kf.predict())[0]), float(np.dot(H,  kf.predict())[1])],'====')
        #print('====',np.dot(H,  kf.predict()).reshape(1,2),'====')
        
        kf.update(z.reshape(2,1))
        #print('after', (kf.x))
        gains.append([kf.x[0], kf.x[1]])
        
        for ii in range(10):
            kf.predict()
        prepredictions.append(list(np.dot(H,  kf.predict()).reshape(1,2)[0]))
    #print(np.array(predictions)[:,0])
    #print(gains)



    import matplotlib.pyplot as plt
    num = 7
    plt.scatter(measurements[:num,0], measurements[:num,1], label = 'Measurements', color='b')
    #plt.scatter(np.array(predictions)[:num,0], np.array(predictions)[:num,1], label = 'Kalman Filter Prediction', color='g')
    #plt.scatter(np.array(gains)[:num,0], np.array(gains)[:num,1], label = 'Gain', color='r')
    plt.scatter(np.array(prepredictions)[:num,0], np.array(prepredictions)[:num,1], label = 'Kalman Filter pre-Prediction', color='k')
    #print(measurements[:10])
    #print(predictions[:10])
    #print(gains[:10])
    #plt.xlim(-10, 10)
    #plt.ylim(-125, 10)
    
    #plt.plot(, label='updated')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    example()