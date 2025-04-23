""" Langevin (or Metropolis) simulation of a quartic potential """

import sys, os, datetime
import numpy as np
# import jax
import jax.numpy as jnp
# from joblib import Parallel, delayed

from Functions.basic_functions_bayesian import run_Metropolis, langevin_sampling

#%%

if_langevin = int(sys.argv[1])
starting_point = float(sys.argv[2])
n_steps = int(sys.argv[3])
seed = int(sys.argv[4])
kT = float(sys.argv[5])

if if_langevin: gamma = 1e-1
else: dx = 1e-1

dir_name = 'quartic'

#%%
# define the quartic potential
quartic_potential = lambda x : 1/30*(0.2*x**4 - 4*x**2 + 2*x)

starting_point = starting_point*np.ones(1)

# my_energy_function_quartic = lambda x : jnp.sum(quartic_potential(x))

if if_langevin:
    out_sim = langevin_sampling(quartic_potential, starting_point, n_steps, seed=seed, kT=kT)
else:
    out_sim = run_Metropolis(starting_point, dx, quartic_potential, n_steps=n_steps,
            i_print=int(1e5), seed=seed)

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

id_code = datetime.datetime.now().strftime('%d_%H_%M_%S_%f')

np.save(dir_name + '/' + id_code + '_traj.npy', out_sim[0])
np.save(dir_name + '/' + id_code + '_ene.npy', out_sim[1])
