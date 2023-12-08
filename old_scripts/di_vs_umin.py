#/usr/bin/env python

import pdb
import numpy as np
#from scipy.optimize import fsolve
from scipy.optimize import newton
from scipy.optimize import root_scalar


#def func(c1, args=(di,)):
#    return np.tan(np.sqrt(Rmc*c1)) - np.sqrt(c1/Rmc)*\
#        (np.cosh(c1/2)*cosh(Ha/2))/((np.tanh(Ha/2)-np.tanh(c1/2))**(-1) - c1*di*np.sinh(Ha/2))


def func(x, di, Rmc, Ha):
    return np.tan(abs(np.sqrt(Rmc*x))) - abs(np.sqrt(x/Rmc))*\
        (np.cosh(x/2)*np.cosh(Ha/2))/((np.tanh(Ha/2)-np.tanh(x/2))**(-1) - x*di*np.sinh(Ha/2))


N = int(100)

di_arr = np.linspace(0.001, 0.05, N)*1.0
um_arr = np.zeros((N, ))
c1_arr = np.zeros((N, ))

Ha  = 100
Rmc = 1000

for i in range(N):
    #c1 = root_scalar(func, bracket=(0.0001, 0.2), method='brentq', args=(di_arr[i], Rmc, Ha))
    c1 = root_scalar(func, x0 = 0.05, x1 = 0.58, args=(di_arr[i], Rmc, Ha))
    #c1 = root_scalar(func, x0=0.001, method='newton', args=(di_arr[i], Rmc, Ha))
    #c1 = newton(func, x0=0.1, args=(di_arr[i], Rmc, Ha))
    #print(c1)
    print(c1.root)
    #um = 1 - np.cosh(c1.root/2)/np.cosh(Ha/2) 
    #c1_arr[i] = c1.root
    #um_arr[i] = um

pdb.set_trace()

#c1 = root_scalar(func, bracket=(0.001, 0.2), method='toms748', args=(0.01,1000, 100))



