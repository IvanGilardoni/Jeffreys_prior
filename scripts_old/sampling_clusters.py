# # Bayesian sampling of ensembles (independent clusters)

# %%

import os
import sys
import numpy as np
import pandas
import datetime
# import jax
# import jax.numpy as jnp

# %%

from basic_functions import run_Metropolis, block_analysis

def energy_fun(new_weights, original_weights, if_Jeffreys, upto: int = 4):
    """
    Given the new set of weights `new_weights` and the original ones `original_weights`,
    it computes the Kullback-Leibler divergence from the original set of weights,
    which corresponds to the loss function, including the Jeffreys prior if `if_Jeffreys`.
    Also, it computes as quantity of interest the population in the first part of the set of weights over `upto`. 
    
    Returns:
        - energy
        - quantity
    """

    # sys_names = data.properties.system_names
    # assert len(sys_names) == 1, 'this script works only for 1 molecular system'
    # sys_name = sys_names[0]

    # chi2 = compute_chi2(data.mol[sys_name].ref, new_weights, data.mol[sys_name].g, data.mol[sys_name].gexp)[3]
    dkl = np.sum(new_weights*np.log(new_weights/original_weights))

    # energy = 1/2*chi2 + alpha*dkl
    energy = dkl

    # quantity: population in the first frame/cluster over `upto` total clusters
    quantity = np.sum(new_weights[:(len(new_weights)//upto)])
    
    # av_g = unwrap_2dict(out.av_g)[0] 

    if if_Jeffreys:
        # Jef_prior = 1/np.sqrt(np.prod(weights))
        # energy = -np.log(Jef_prior)

        # more efficient?
        energy += 1/2*np.sum(np.log(new_weights))
    
    return energy, quantity

def proposal(weights, dx = 0.01):
    """ Based on the stick-breaking process """

    if not np.sum(weights) == 1: weights = weights/np.sum(weights)

    x = np.cumsum(weights)[:-1]
    
    x += dx*np.random.normal(size=len(x))
    x = np.mod(x, 1)
    x = np.concatenate((np.sort(x), [1]))

    weights_new = np.concatenate(([x[0]], np.ediff1d(x)))

    return weights_new

def double(weights, n: int = 2):
    """ double if n = 2, else three times and so on """

    weights_double = np.zeros(len(weights)*n)

    for i in range(len(weights)):
        for j in range(n):
            weights_double[n*i + j] = weights[i]

    return weights_double

# %%
size = 4
seed = 1

delta_w = 1e-2

n_steps = int(1e5)

# n_times = 1
n_times = int(sys.argv[1])

# if_Jeffreys = True
if_Jeffreys = int(sys.argv[2])

# %%
rng = np.random.RandomState(seed=1)

weights_orig = rng.uniform(0, 1, size=size)
weights_orig = weights_orig/np.sum(weights_orig)

weights_double = double(weights_orig, n_times)

proposal_full = {'fun': proposal, 'args': ([delta_w])}
energy_fun_full = {'fun': energy_fun, 'args': ([weights_orig, if_Jeffreys, size])}

out = run_Metropolis(weights_orig, proposal=proposal_full, energy_function=energy_fun_full, n_steps=n_steps)

av_accept = out[3]

# %%

blocks = block_analysis(out[2], delta=10)

opt_epsilon = blocks[2]
epsilons = blocks[3]
smooth_epsilons = blocks[4]

# %%

av_pop = [blocks[0], blocks[1], blocks[2]]

cols = ['n_times', 'if_Jeffreys', 'av. accept', 'av. value', 'std', 'block error']
temp = pandas.DataFrame([n_times, if_Jeffreys, av_accept] + av_pop, index=cols, columns=[0]).T

s = datetime.datetime.now()
date = s.strftime('%Y_%m_%d_%H_%M_%S_%f')

temp.to_csv('Result/values_%s.txt' % date)


