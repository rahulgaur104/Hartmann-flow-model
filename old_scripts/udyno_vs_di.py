#/usr/bin/env python

import pdb
import numpy as np
#from scipy.optimize import fsolve
from scipy.optimize import newton
from scipy.optimize import root_scalar
from matplotlib import pyplot as plt

#def func(c1, args=(di,)):
#    return np.tan(np.sqrt(Rmc*c1)) - np.sqrt(c1/Rmc)*\
#        (np.cosh(c1/2)*cosh(Ha/2))/((np.tanh(Ha/2)-np.tanh(c1/2))**(-1) - c1*di*np.sinh(Ha/2))


def func(c1, dc, Re, Rmc, Ha):
    di =  (Re/Ha*1/np.cosh(c1/2) - np.sqrt(c1/Rmc)*(np.tanh(Ha/2) - np.tanh(c1/2))/np.tanh(abs(np.sqrt(Rmc * c1)*dc)))/(c1*(np.tanh(c1/2) + np.tanh(Ha/2)))
        
    u1 = 1 - np.cosh(c1/2)/np.cosh(Ha/2)
    u2 =  (np.cosh(-c1*0.05) - np.cosh(c1/2)*np.cosh(Ha*0.05)/np.cosh(Ha/2))
    return di, u2

N = 1000
ud_arr = np.zeros((N, ))
c1_arr = np.logspace(-3, 1, N)

Re = 50
Ha = 200
Rmc = 1000
dc = 0.02


plt.figure()
ax = plt.gca()

di0, ud0 = func(c1_arr, dc, Re, Rmc, Ha)
plt.plot(di0, ud0, '-r', linewidth=2)


ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=18)
plt.yscale('log')
plt.xscale('log')
plt.xlim([0.0002, 0.3])

plt.ylabel(r"$u_{\mathrm{dyno}}$", fontsize=20)
plt.xlabel(r"$d_i$", fontsize=20)


di1, ud1 = func(c1_arr, dc, Re, Rmc, 2*Ha)
plt.plot(di1, ud1, '-g', linewidth=2)

di2, ud2 = func(c1_arr, dc, Re, Rmc, 4*Ha)
plt.plot(di2, ud2, '-b', linewidth=2)


plt.tight_layout()
plt.show()


#plt.figure()
#di, ud = func(c1_arr, dc, Re, Rmc, Ha)
#plt.plot(di, ud, '-r', linewidth=2)
#
#di, ud = func(c1_arr, dc, 2*Re, 1*Rmc, Ha)
#plt.plot(di, ud, '-g', linewidth=2)
#
#di, ud = func(c1_arr, dc, 4*Re, 1*Rmc, Ha)
#plt.plot(di, ud, '-b', linewidth=2)
#
#plt.yticks(fontsize=18)
#plt.xticks(fontsize=18)
#plt.ylabel(r"$u_{\mathrm{dyno}}$", fontsize=20)
#plt.xlabel(r"$d_i$", fontsize=20)
#
##plt.yscale('log')
##plt.xscale('log')
##plt.xlim([0.0002, 0.2])
#plt.xlim([0.0002, 0.3])
#plt.ylim([1, 2])
#
#plt.tight_layout()
#plt.show()



#plt.figure()
#ax = plt.gca()
#di, ud = func(c1_arr, dc, Re, Rmc, Ha)
#plt.plot(di, ud, '-r', linewidth=2)
#
#di, ud = func(c1_arr, 2*dc, Re, 1*Rmc, Ha)
#plt.plot(di, ud, '-g', linewidth=2)
#
#di, ud = func(c1_arr, 4*dc, Re, 1*Rmc, Ha)
#plt.plot(di, ud, '-b', linewidth=2)
#
#pdb.set_trace()
#
##plt.yticks(fontsize=18)
##plt.xticks(fontsize=18)
#ax.tick_params(axis='both', which='major', labelsize=18)
#ax.tick_params(axis='both', which='minor', labelsize=18)
#plt.ylabel(r"$u_{\mathrm{dyno}}$", fontsize=20)
#plt.xlabel(r"$d_i$", fontsize=20)
#
#plt.yscale('log')
#plt.xscale('log')
##plt.xlim([0.0002, 0.2])
#plt.xlim([0.0002, 0.3])
#plt.ylim([1, 2])
#
#plt.tight_layout()
#plt.show()


pdb.set_trace()

#c1 = root_scalar(func, bracket=(0.001, 0.2), method='toms748', args=(0.01,1000, 100))



