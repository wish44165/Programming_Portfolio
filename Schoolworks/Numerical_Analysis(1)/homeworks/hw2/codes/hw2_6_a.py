# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:20:30 2020

@author: ktpss
"""

def neville(x_eva, xl, fl, k):
    if (k==1):
        for i in range(len(fl)):
            fl[i] = round(fl[i], 8)
            print('p_{%d%d}(%.4f) = '% (i, i, x_eva), fl[i])
        #print(fl)
        print('')
    pl = []
    for i in range(len(xl)-k):
        pl.append(round((((x_eva - xl[i+k])*fl[i] - (x_eva - xl[i])*fl[i+1]) / (xl[i] - xl[i+k])), 8))
        print('p_{%d%d}(%.4f) = '% (i, i+k, x_eva), pl[i])
    #print(pl)
    print('')
    k = k+1
    if (k<len(xl)):
        return neville(x_eva, xl, pl, k)

#p01 = ((x-x1)*p00 - (x-x0)*p11) / (x0-x1)
                #p02 = ((x-x2)*p01 - (x-x1)*p12) / (x0-x2)
#p12 = ((x-x2)*p11 - (x-x1)*p22) / (x1-x2)
#p23 = ((x-x3)*p22 - (x-x2)*p33) / (x2-x3)
                

n = input('enter the number of data points: ')
n = int(n)
xl = []
fl = []
for i in range(n):
    xl.append(float(input('enter x_%d: ' %i)))
    fl.append(float(input('enter y_%d: ' %i)))

#xl = [1, 1.3, 1.6, 1.9, 2.2]
#fl = [0.7651977, 0.6200860, 0.4554022, 0.2818186, 0.1103623]

x_eva = input('enter the evaluate point x: ')
x_eva = float(x_eva)

print('\n--------result of Neville\'s algorithm--------\n')
neville(x_eva, xl, fl, 1)