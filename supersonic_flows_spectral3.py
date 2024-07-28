#!/usr/bin/env python3
"""
This function solves the channel flow equations in a rectangular
domain using a spectral solver written using the dedalus2 package

∂uy/∂t = F0(1 - exp(-c0 * t)) + ∂By/∂z + (1/Re) * ∂²uy/∂z²

∂By/∂t = ∂uy/∂z + (1/Rm) * ∂²By/∂z²

Specifically, we create Figure 4 of the paper with this script
(uybar vs t)
"""
from dedalus import public as de
from dedalus.core.operators import integrate
import numpy as np

import matplotlib
#matplotlib.use('TkAgg')  # or 'Qt5Agg' or any other interactive backend

from matplotlib import pyplot as plt
import sys

if sys.argv[1] is None:
    Nz = int(128)
else:
    Nz = int(eval(sys.argv[1]))

len_d_i = 1
d_i_array = np.linspace(0.01, 1, len_d_i)

# Define the domain
nz = Nz  # Number of grid points
Lz = 1.0  # Length of the domain in the z-direction

# Create basis and domain using Chebyshev basis
z_basis = de.Chebyshev('z', nz, interval=(0, Lz))
domain = de.Domain([z_basis], grid_dtype=np.float64)

E_0 = -1.0
B_0 = 1.0
F_0 = 0.5
c_0 = 0.01
S_c = 1e4
Ha = 500
Re = 50
Rm = Ha**2/Re
d_c = 5e-3
fac0 = np.sqrt(c_0*S_c)
fac1 = np.sqrt(c_0/S_c)

# Define the problem
problem = de.IVP(domain, variables=['uy', 'By', 'dz_By', 'dz_uy'])
problem.parameters['F_0'] = F_0  # Forcing term
problem.parameters['c_0'] = c_0  
problem.parameters['Re'] = Re  # Reynolds number
problem.parameters['Rm'] = Rm  # Magnetic Reynolds number
problem.parameters['S_c'] = S_c  # Magnetic Reynolds number of the perfect conductor
problem.parameters['E_0'] = E_0
problem.parameters['B_0'] = B_0
problem.parameters['fac0'] = fac0
problem.parameters['fac1'] = fac1
problem.parameters['d_c'] = d_c
problem.parameters['d_i'] = d_i_array[0]


## Boundary values as a function of time
#Byb = "((1-exp(-10000*t))*(S_c*E_0*d_c + B_0) - (exp(-c_0*t)-exp(-10000*t))*(E_0/fac1*sin(fac0*d_c) + B_0*cos(fac0*d_c)))"
#Exb = "(1-exp(-10000*t))*E_0 + (exp(-c_0*t)-exp(-10000*t))*(-E_0*cos(fac0*d_c) + B_0*fac1*sin(fac0*d_c) + c_0*d_i*(E_0/fac1*sin(fac0*d_c) + B_0*cos(fac0*d_c)))"

# Boundary values as a function of time
Byb = "((S_c*E_0*d_c + B_0) - exp(-c_0*t)*(E_0/fac1*sin(fac0*d_c) + B_0*cos(fac0*d_c)))"
Exb = "E_0 + exp(-c_0*t)*(-E_0*cos(fac0*d_c) + B_0*fac1*sin(fac0*d_c) + c_0*d_i*(E_0/fac1*sin(fac0*d_c) + B_0*cos(fac0*d_c)))"

problem.add_equation("dt(uy)  - dz(By) - (1/Re)*dz(dz_uy) = F_0*(1-exp(-c_0*t))")
problem.add_equation("dt(By)  - dz(uy) - (1/Rm)*dz(dz_By) = 0")
problem.add_equation("dz_By - dz(By) = 0")
problem.add_equation("dz_uy - dz(uy) = 0")

problem.add_bc(f"left(By) = -1*{Byb}")  # Dirichlet condition at left boundary
problem.add_bc(f"right(uy) + 1/Rm*right(dz_By) = -{Exb}")  # Mixed condition at right boundary
problem.add_bc(f"left(uy) + 1/Rm*left(dz_By)  = -{Exb}")  # Mixed condition at left boundary
problem.add_bc(f"right(By) = {Byb}")  # Dirichlet condition at right boundary

# Set up the solver
ts = de.timesteppers.SBDF2  # Time-stepper
#ts = de.timesteppers.RK443  # Time-stepper

solver = problem.build_solver(ts)

Byb_t0 = (S_c*E_0*d_c + B_0) - (E_0/fac1*np.sin(fac0*d_c) + B_0*np.cos(fac0*d_c))
Exb_t0 = E_0 + (-E_0*np.cos(fac0*d_c) + B_0*fac1*np.sin(fac0*d_c)) + c_0*d_i_array[0]*(E_0/fac1*np.sin(fac0*d_c) + B_0*np.cos(fac0*d_c))

# Initial conditions or state setup
uy = solver.state['uy']
By = solver.state['By']
dz_uy = solver.state['dz_uy']
dz_By = solver.state['dz_By']

uy['g'] = np.zeros_like(uy['g'])  # Initial condition for uy
By['g'] = np.zeros_like(By['g'])  # Initial condition for uy
dz_uy['g'] = np.zeros_like(dz_uy['g'])  # Initial condition for dz_uy
dz_By['g'] = np.zeros_like(dz_By['g'])  # Initial condition for dz_By

z = domain.grid(0)

# Simulation parameters
solver.stop_sim_time = 500  # Simulation stop time
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

dt = 0.01  # Time step size

# Initialize lists to store results
uy_list = []
By_list = []
uy_bar_list = []
times = []

plot_counter = int(0)
mask = np.where((z >= 0.25) & (z <= 0.75), 1, 0)

# Main simulation loop
while solver.ok:

    #pdb.set_trace()
    ## Integrate uy over the z domain
    #total_uy = 2*integrate(solver.state['uy']*mask, 'z')  # Integrate uy over z

    total_uy = 2*np.trapz(solver.state['uy']["g"]*mask, z*mask)  # Integrate uy over z
    uy_bar_list.append(total_uy / Lz)


    if solver.iteration % 1000 == 0 or (solver.iteration <= 1000 and solver.iteration % 50 == 0): 
    #if solver.iteration > 20000 and solver.iteration < 20002:
        print('Completed iteration', solver.iteration)
        # Store the current state and time
        uy_list.append(np.copy(uy['g']))
        By_list.append(np.copy(By['g']))
        times.append(solver.sim_time)


    solver.step(dt)  # Advance the solution by one time step

x_1 = z.copy() - 0.5
#nz_analytical = 1001
nz_analytical = len(x_1)
x_1 = np.linspace(-0.5, 0.5, nz_analytical)

# Analytical solution
B_1 = 1/np.cosh(-0.5*c_0) * (1*(E_0 - c_0 * d_i_array[0] * B_0)* np.cos(fac0*d_c) - fac1 * (B_0 + S_c * d_i_array[0] * E_0) * np.sin(fac0 * d_c) - F_0/c_0)
B_2 = Re/Ha*(B_1 * np.sinh(-c_0 * 0.5) - (1 / fac1 * E_0 * np.sin(fac0 * d_c) + B_0 * np.cos(fac0 * d_c)))

uy_steady_state = -E_0 - Re/Ha * ((np.exp(-Ha*(x_1+0.5)) + np.exp(Ha*(x_1-0.5)))/(1 - np.exp(-Ha))) * (S_c*E_0*d_c + B_0 + 0.5*F_0) 

len1 = len(uy_bar_list)
uy_total_analytical = np.zeros((len1, nz_analytical))

t0 = np.linspace(0, solver.stop_sim_time, len1)  # Simulation stop time

for i in range(len1):
    uy_total_analytical[i, :] = uy_steady_state + np.exp(-c_0*t0[i])*(F_0/c_0 + B_1 * np.cosh(-c_0 * x_1) - B_2 * (np.exp(Ha * (x_1 - 0.5)) + np.exp(-Ha * (x_1 + 0.5)))/(1 - 0.))

mask = np.where((x_1 >= -0.25) & (x_1 <= 0.25), 1, 0)
mask_2D = np.vstack([mask for i in range(len1)])
#pdb.set_trace()
uy_bar_total_analytical = 2*np.trapz(uy_total_analytical * mask_2D, x_1 * mask, axis=1)

# Convert lists to arrays for easier indexing
uy_array = np.array(uy_list)
By_array = np.array(By_list)
z_array = domain.grid(0)

uy_bar = np.array(uy_bar_list)
len1 = len(uy_bar)

# Mean flow vs time
t = np.linspace(0, solver.stop_sim_time, len1)  # Simulation stop time

fig, ax1 = plt.subplots(figsize=(6, 4))

# These are in unitless percentages of the figure size. (0,0 is bottom left)
left, bottom, width, height = [0.39, 0.57, 0.5, 0.38]
ax2 = fig.add_axes([left, bottom, width, height])


# Mean flow vs time
ax1.plot(t, uy_bar, '-or', ms=0.5)
ax1.plot(t0, uy_bar_total_analytical, '-k', ms=0.5)
ax1.axvspan(50, t[-1], color='black', alpha=0.1)

ax1.set_xlabel(r"$t$", fontsize=20)
ax1.set_ylabel(r"$\bar{u}_y$", fontsize=20, labelpad=-2)
ax1.tick_params(axis='both', which='major', labelsize=18)
ax1.legend(["numerical", "analytical"], fontsize=18, loc='lower right', framealpha=0.5)

ax1.set_xlim([-5, 500])


inset_range_index = 1000

# Inset plot
ax2.plot(t[:inset_range_index], uy_bar[:inset_range_index], '-or', ms=1.0, linewidth=0.5)
ax2.plot(t0[:inset_range_index], uy_bar_total_analytical[:inset_range_index], '-k', ms=0.5)

# Remove labels in the inset
ax2.tick_params(axis='both', which='major', labelsize=18)
ax2.set_xticks([0, t[inset_range_index]])
ax2.set_yticks([0, 2])

# Set specific limits for ticks in the inset
ax2.set_xticklabels([0, t[inset_range_index]])
ax2.set_yticklabels([0, 2])

print(len1, len(uy_bar), len(uy_bar_total_analytical))

plt.tight_layout()
#plt.savefig(f"uybar_vs_t_Ha{Ha}_zero_initial_mask_test.png", dpi=400)
plt.savefig(f"uybar_vs_t_Ha{Ha}_nonzero_initial_mask.png", dpi=400)
#plt.savefig(f"uybar_vs_t_Ha{Ha}_nonzero_initial_mask_d_i{d_i_array[0]}.png", dpi=400)
#plt.savefig(f"uybar_vs_t_Ha{Ha}_zero_initial_mask_d_i{d_i_array[0]}.png", dpi=400)
#plt.savefig(f"uybar_vs_t_Ha{Ha}_nonzero_initial_mask_d_i{d_i_array[0]}.png", dpi=400)


## Mean flow vs time
#t = np.linspace(0, solver.stop_sim_time, len1)  # Simulation stop time
#plt.plot(t, uy_bar, '-or', ms=0.5)
#plt.plot(t0, uy_bar_total_analytical, '-k', ms=0.5)
#print(uy_bar_total_analytical[-1])
#plt.axvspan(30, t[-1], color='black', alpha=0.1)
##plt.text(20, 3, "acceleration", fontsize=18)
##plt.text(20, 2.3, "deceleration", fontsize=18)
#
##np.save(f"time_zero_initial_Ha{Ha}.npy",t) 
##np.save(f"uy_bar_Ha{Ha}_zero_initial.npy", uy_bar) 
#
##np.save(f"time_nonzero_initial_Ha{Ha}.npy",t) 
##np.save(f"uy_bar_Ha{Ha}_nonzero_initial.npy", uy_bar) 
#
##window = 500
##uy_bar_rolling = np.convolve(uy_bar, np.ones((window,)), 'valid') / window
##plt.plot(t[window-1:], uy_bar_rolling, '-k', linewidth=3)
#plt.xlabel(r"$t$", fontsize=20)
#plt.ylabel(r"$\bar{u}_y$", fontsize=20, labelpad=-2)
##plt.xscale("log")
#plt.xticks(fontsize=18)
#plt.yticks(fontsize=18)
#plt.legend(["numerical", "analytical"], fontsize=18)
#plt.tight_layout()
##plt.savefig(f"uybar_vs_t_Ha{Ha}_zero_initial.png", dpi=400)
##plt.savefig(f"uybar_vs_t_Ha{Ha}_nonzero_initial.png", dpi=400)
#
##plt.savefig(f"uybar_vs_t_Ha{Ha}_zero_initial_mask.png", dpi=400)
#plt.savefig(f"uybar_vs_t_Ha{Ha}_nonzero_initial_mask.png", dpi=400)
#
##plt.tight_layout()
##plt.show()
#


