""" Distribution of the exit times from a simple quadratic well """

import sys, os, datetime
import numpy as np
# import jax
import jax.numpy as jnp
# from joblib import Parallel, delayed

# from basic_functions_bayesian import run_Metropolis, langevin_sampling
from Functions.basic_functions_gaussian import sample_and_group

#%%

depth = float(sys.argv[1])
seed = int(sys.argv[2])
n_steps = int(sys.argv[3])

#%%

my_energy_function_simple = lambda x : -depth*jnp.exp(-x**2)

sampling_pars = {'n_steps': n_steps, 'starting_point': np.ones(1),
    'which_sampling': 'Langevin'}
    # 'which_sampling': 'Metropolis', 'dx': 0.05}

group_pars = {'if_diff': False, 'threshold': 50, 'value': 0.}
# group_pars = {'threshold': 200}

tolerance = 2e-2*depth  # for Group_points function

results = sample_and_group(my_energy_function_simple, **sampling_pars, seed=seed, **group_pars, tolerance=tolerance)

#%%

dir_name = 'exit_times'

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

id_code = datetime.datetime.now().strftime('%d_%H_%M_%S_%f')
subdir_name = dir_name + '/' + id_code
os.makedirs(subdir_name)

traj = results[depth][seed][0][0]
np.save(subdir_name + '/traj.npy', traj)

ene = results[depth][seed][0][1]
np.save(subdir_name + '/ene.npy', traj)

np.save(subdir_name + 'whs_first.npy', results[depth][seed][1].whs_first)
np.save(subdir_name + 'whs_len.npy', results[depth][seed][1].whs_len)
np.save(subdir_name + 'whs_flat.npy', np.array(results[depth][seed][1].whs_flat))

