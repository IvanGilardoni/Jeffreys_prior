#%% Script to run Metropolis sampling and return where it gets stuck in the plateau region

import sys, os, datetime, pandas
import numpy as np
# import matplotlib.pyplot as plt
# from IPython.display import clear_output

# from basic_functions_gaussian import compute_depth, compute_depth_analytical, flatten, my_group_fun, loss_fun
# from basic_functions_bayesian import compute_single, compute, run_Metropolis

from basic_functions_gaussian import loss_fun, my_group_fun, compute_depth, compute_depth_analytical
from basic_functions_bayesian import run_Metropolis

#%% input values

alpha = 0.5

sigma = 0.1
gexp = 0.3
sigma_exp = float(sys.argv[1])

n_frames = np.int64(1e4)

n_steps = np.int64(1e6)
step_length = 5.

seed = int(sys.argv[2])

dir_name = 'Metropolis_sampling'

#%%

rng = np.random.default_rng(seed)
rng_fixed = np.random.default_rng(1)

if not os.path.exists(dir_name):
    os.makedirs(dir_name)

id_code = datetime.datetime.now().strftime('%d_%H_%M_%S_%f')
subdir_name = dir_name + '/' + id_code

os.makedirs(subdir_name)

infos = {'id_code': id_code, 'alpha': alpha, 'sigma': sigma, 'gexp': gexp, 'sigma_exp': sigma_exp,
    'n_frames': n_frames, 'n_steps': n_steps, 'seed': seed}

pandas.DataFrame(list(infos.values()), index=list(infos.keys())).T.to_csv(subdir_name + '/input_pars', index=False)

#%%

p0 = np.ones(n_frames)/n_frames
g = rng_fixed.normal(0, sigma, size=n_frames)

dV_num = compute_depth(n_frames, sigma, gexp, sigma_exp, alpha, delta_lambda=100).dV
dV_th = compute_depth_analytical(n_frames, sigma, gexp, sigma_exp, alpha).dV

def energy_function(x, p0, g, gexp, sigma_exp, alpha, if_jeffreys = False):

    out = loss_fun(x, p0, g, gexp, sigma_exp, alpha, if_cov=if_jeffreys)

    if if_jeffreys:
        energy = out[0]
        cov = out[-1]
        jeff = np.log(np.linalg.det(cov))
        energy -= jeff

    else:
        energy = out

    return np.array([energy])

# dV[sigma_exp] = compute_depth_analytical(n, sigma, gexp, sigma_exp, alpha).dV

# out[sigma_exp] = compute_depth(n, np.array([g]), gexp, sigma_exp, alpha, if_scan=True, delta_lambda=100)
# out_compute = compute(out[sigma_exp].scan_lambdas, p0, g, gexp, sigma_exp, alpha)

#%%

result_values = {'dV_num': dV_num, 'dV_th': dV_th}

my_energy_function = lambda x : energy_function(x, p0, g, gexp, sigma_exp, alpha, False)

out_Metropolis = run_Metropolis(np.zeros(1), step_length, my_energy_function, n_steps=n_steps)

np.save(subdir_name + '/lambdas_noJef', out_Metropolis[0])
np.save(subdir_name + '/energies_noJef', out_Metropolis[1])
result_values['accept_noJef'] = out_Metropolis[2]

my_energy_function = lambda x : energy_function(x, p0, g, gexp, sigma_exp, alpha, True)

out_J_Metropolis = run_Metropolis(np.zeros(1), step_length, my_energy_function, n_steps=n_steps)

np.save(subdir_name + '/lambdas_Jef', out_J_Metropolis[0])
np.save(subdir_name + '/energies_Jef', out_J_Metropolis[1])
result_values['accept_Jef'] = out_J_Metropolis[2]

dif = np.ediff1d(out_Metropolis[1])
group = my_group_fun(dif, 0, 0.05)
wh = np.where(np.array(group[1]) > 100)[0]
if not (len(wh) == 0): ind = group[0][wh[0]]
else: ind = None

result_values['diverg_frame'] = ind

temp = pandas.DataFrame(list(result_values.values()), index=list(result_values.keys())).T
temp.to_csv(subdir_name + '/result_values', index=False)

