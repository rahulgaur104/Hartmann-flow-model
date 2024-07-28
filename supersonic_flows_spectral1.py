#!/usr/bin/env python3
"""
This function solves the channel flow equations in a rectangular
domain using a spectral solver written using the dedalus2 package

∂uy/∂t = F0(1 - exp(-c0 * t)) + ∂By/∂z + (1/Re) * ∂²uy/∂z²

∂By/∂t = ∂uy/∂z + (1/Rm) * ∂²By/∂z²

Specifically, we create Figure 6 of the paper with this script
(uy vs uy_analytical)
"""
from dedalus import public as de
from dedalus.core.operators import integrate
import numpy as np

import matplotlib
#matplotlib.use('TkAgg')  # or 'Qt5Agg' or any other interactive backend

from matplotlib import pyplot as plt
import pdb

import sys

if sys.argv[1] is None:
    Nz = int(128)
else:
    Nz = int(eval(sys.argv[1]))

len_d_i = 1
d_i_array = np.linspace(0.01, 1, len_d_i)

# For appendix comparison
plot_indices = [49]

# Define the domain
nz = Nz  # Number of grid points
Lz = 1.0  # Length of the domain in the z-direction

# Create basis and domain using Chebyshev basis
z_basis = de.Chebyshev('z', nz, interval=(0, Lz))
domain = de.Domain([z_basis], grid_dtype=np.float64)

E_0 = -1.0
B_0 = 1
F_0 = 0.5
c_0 = 0.01
S_c = 1e4
Ha = 50
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


# Boundary values as a function of time
Byb = "((1-exp(-10000*t))*(S_c*E_0*d_c + B_0) - (exp(-c_0*t)-exp(-10000*t))*(E_0/fac1*sin(fac0*d_c) + B_0*cos(fac0*d_c)))"
Exb = "(1-exp(-10000*t))*E_0 + (exp(-c_0*t)-exp(-10000*t))*(-E_0*cos(fac0*d_c) + B_0*fac1*sin(fac0*d_c) + c_0*d_i*(E_0/fac1*sin(fac0*d_c) + B_0*cos(fac0*d_c)))"

## Boundary values as a function of time
#Byb = "((S_c*E_0*d_c + B_0) - exp(-c_0*t)*(E_0/fac1*sin(fac0*d_c) + B_0*cos(fac0*d_c)))"
#Exb = "E_0 + exp(-c_0*t)*(-E_0*cos(fac0*d_c) + B_0*fac1*sin(fac0*d_c) + c_0*d_i*(E_0/fac1*sin(fac0*d_c) + B_0*cos(fac0*d_c)))"

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
# Initialize a counter for the plot indices
next_plot_index = plot_indices[plot_counter]  # Get the first plot index


# Main simulation loop
while solver.ok:
    if solver.iteration % 1000 == 0 or (solver.iteration <= 1000 and solver.iteration % 50 == 0): 
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

len1 = len(uy_list)
uy_total_analytical = np.zeros((len1, nz_analytical))

t0 = np.linspace(0, solver.stop_sim_time, len1)  # Simulation stop time

for i in range(len1):
    uy_total_analytical[i, :] = uy_steady_state + np.exp(-c_0*t0[i])*(F_0/c_0 + B_1 * np.cosh(-c_0 * x_1) - B_2 * (np.exp(Ha * (x_1 - 0.5)) + np.exp(-Ha * (x_1 + 0.5)))/(1 - 0.))


# Convert lists to arrays for easier indexing
uy_array = np.array(uy_list)
By_array = np.array(By_list)
z_array = domain.grid(0)

#pdb.set_trace()

fig = plt.figure(figsize=(5, 4))
plt.plot(z_array-Lz/2, uy_array[-1], '-or', ms = 3, linewidth =2, alpha=1.0, label=f't=500, numerical')
plt.plot(z_array-Lz/2, uy_total_analytical[-1], '-ob', ms = 3, linewidth =2, alpha=1.0, label=f't=500, analytical')
plt.xlabel(r"$z$", fontsize=20)
plt.ylabel(r"$u_y$", fontsize=20, labelpad=-5)
#plt.yscale("symlog")
plt.legend(fontsize=18)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig(f"uy_vs_z_Ha{Ha}_num_analytic_comp.png", dpi=400)
#plt.show()




