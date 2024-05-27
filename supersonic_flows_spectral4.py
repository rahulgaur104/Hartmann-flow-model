#!/usr/bin/en python3
"""
This function solves the channel flow equations in a rectangular
domain using a spectral solver written using the dedalus2 package

∂uy/∂t = F0(1 - exp(-c0 * t)) + ∂By/∂z + (1/Re) * ∂²uy/∂z²

∂By/∂t = ∂uy/∂z + (1/Rm) * ∂²By/∂z²

Specifically, we create Figure 5 of our paper with this script
(uybar/E_0 vs d_i as a function of B_0)
"""
from dedalus import public as de
from dedalus.core.operators import integrate
import numpy as np

from matplotlib import pyplot as plt
import pdb


len_d_i = 5
d_i_array = np.linspace(0.01, 3, len_d_i)

plot_indices = [5, 10, 20, 50]

uy_bar = np.zeros((len(plot_indices), len_d_i))
plot_counter = 0

for j in range(len_d_i):
    # Define the domain
    nz = 128  # Number of grid points
    Lz = 1.0  # Length of the domain in the z-direction
    
    # Create basis and domain using Chebyshev basis
    z_basis = de.Chebyshev('z', nz, interval=(0, Lz))
    domain = de.Domain([z_basis], grid_dtype=np.float64)
    
    E_0 = -10
    B_0 = 1.
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
    problem.parameters['F_0'] = 0.5  # Forcing term
    problem.parameters['c_0'] = c_0  
    problem.parameters['Re'] = Re  # Reynolds number
    problem.parameters['Rm'] = Rm  # Magnetic Reynolds number
    problem.parameters['S_c'] = S_c  # Magnetic Reynolds number of the perfect conductor
    problem.parameters['E_0'] = E_0
    problem.parameters['B_0'] = B_0
    problem.parameters['fac0'] = fac0
    problem.parameters['fac1'] = fac1
    problem.parameters['d_c'] = d_c
    problem.parameters['d_i'] = d_i_array[j]
    

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
    
    problem.add_bc(f"left(By) = -1*{Byb}")  # Dirichlet condition on the left boundary
    problem.add_bc(f"right(uy) + 1/Rm*right(dz_By) = -{Exb}")  # Mixed condition on the right boundary
    problem.add_bc(f"left(uy) + 1/Rm*left(dz_By)  = -{Exb}")  # Mixed condition on the left boundary
    problem.add_bc(f"right(By) = {Byb}")  # Dirichlet condition on the right boundary
    
    # Set up the solver
    ts = de.timesteppers.SBDF2  # Time-stepper
    #ts = de.timesteppers.RK443  # Time-stepper
    
    solver = problem.build_solver(ts)
 
    Byb_t0 = (S_c*E_0*d_c + B_0) - (E_0/fac1*np.sin(fac0*d_c) + B_0*np.cos(fac0*d_c))
    Exb_t0 = E_0 + (-E_0*np.cos(fac0*d_c) + B_0*fac1*np.sin(fac0*d_c)) + c_0*d_i_array[j]*(E_0/fac1*np.sin(fac0*d_c) + B_0*np.cos(fac0*d_c))
   
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
    times = []
 
    plot_counter = int(0)
    # Initialize a counter for the plot indices
    next_plot_index = plot_indices[plot_counter]  # Get the first plot index

    # Main simulation loop
    while solver.ok:
        # Check if the current iteration matches one of the plot indices
        if int(solver.iteration / 1000) == int(next_plot_index):
            print('Completed iteration', solver.iteration)

            lower_lim = 0.25
            upper_lim = 0.75
            mask = np.where((z >= lower_lim) & (z <= upper_lim), 1, 0)
            total_uy = np.trapz(solver.state['uy']["g"]*mask, z*mask)  # Integrate uy over z
            uy_bar[plot_counter, j] = total_uy / (upper_lim-lower_lim)

            # Update the counter and get the next plot index, if any left
            plot_counter += 1
            if plot_counter < len(plot_indices):
                next_plot_index = plot_indices[plot_counter]
            else:
                break  # Exit loop if all plot indices have been processed

        solver.step(dt)  # Advance the solution by one time step

    print(f"d_i_idx {j} done, {plot_counter}")

   
    # Convert lists to arrays for easier indexing
    uy_array = np.array(uy_list)
    By_array = np.array(By_list)
    z_array = domain.grid(0)
    
    print(f"d_i_idx {j} done")

colors = ['r', 'g', 'b', 'k']

fig = plt.figure(figsize=(5, 4))

for i in range(len(plot_indices)):
    plt.plot(d_i_array, uy_bar[i, :], colors[i])


plt.xlabel(r"$d_i$", fontsize=20)
plt.ylabel(r"$\bar{u}_y/E_{\mathrm{v}0}$", fontsize=20, labelpad=-1)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend([r"$t = 50$", r"$t = 100$",r"$t = 200$",r"$t = 500$"], fontsize=18)
plt.tight_layout()
plt.savefig(f"uybar_vs_d_i_E_0{E_0}_B_0{B_0}_{Ha}_mask.png", dpi=400)


