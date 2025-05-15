"""
Script to perform Bayesian sampling of the optimal ensembles parametrized by lambda coefficients
(Ensemble Refinement only or Force-Field Fitting only).
"""

import os
import datetime
import sys
import numpy as np
import jax
import jax.numpy as jnp
# import matplotlib.pyplot as plt
import pandas

import sys
sys.path.append('../loss_function_complete/')

from MDRefine.MDRefine import load_data, normalize_observables, minimizer, loss_function, unwrap_2dict
from Functions.basic_functions_bayesian import run_Metropolis, local_density

def flat_lambda(lambdas):

    flatten_lambda = []
    for name_mol in data.properties.system_names:
        flatten_lambda = flatten_lambda + list(
            np.hstack([lambdas[name_mol][k] for k in data.mol[name_mol].n_experiments.keys()]))

    flatten_lambda = np.array(flatten_lambda)

    return flatten_lambda

#%% 0. global values

stride = int(sys.argv[1])  # stride for the frames
alpha = float(sys.argv[2])  # hyperparameter value for Ensemble Refinement loss function
beta = None # float(sys.argv[2])

if_normalize = int(sys.argv[3])  # True (1) if you want to normalize the observables, False (0) otherwise
if_reduce = int(sys.argv[4])  # True if you want to take just 2 observables (it works only for `AAAA` and `backbone1_gamma_3J`)

dx = float(sys.argv[5])  # standard deviation of the normal distribution for the proposal
if_Jeffreys = int(sys.argv[6])  # boolean variable (True if you take into account the Jeffreys prior, False otherwise)
n_steps = int(sys.argv[7])  # n. of steps in the Metropolis sampling

#%% 1. load data

infos = {'global': {
    'path_directory': '../loss_function_complete/DATA',
    'system_names': ['AAAA'],  # , 'CAAU'],  # , 'CCCC', 'GACC', 'UUUU', 'UCAAUC'],
    'g_exp': ['backbone1_gamma_3J'],  # , 'backbone2_beta_epsilon_3J', 'sugar_3J'],#, 'NOEs'],# , ('uNOEs', '<')],
    'forward_qs': ['backbone1_gamma'],  # , 'backbone2_beta_epsilon','sugar'],
    # 'obs': ['NOEs'],#, 'uNOEs'],
    'forward_coeffs': 'original_fm_coeffs'}}

def forward_model_fun(fm_coeffs, forward_qs, selected_obs=None):

    # 1. compute the cosine (which is the quantity you need in the forward model;
    # you could do this just once before loading data)
    forward_qs_cos = {}

    for type_name in forward_qs.keys():
        forward_qs_cos[type_name] = jnp.cos(forward_qs[type_name])

    # if you have selected_obs, compute only the corresponding observables
    if selected_obs is not None:
        for type_name in forward_qs.keys():
            forward_qs_cos[type_name] = forward_qs_cos[type_name][:,selected_obs[type_name+'_3J']]

    # 2. compute observables (forward_qs_out) through forward model
    forward_qs_out = {
        'backbone1_gamma_3J': fm_coeffs[0]*forward_qs_cos['backbone1_gamma']**2 + fm_coeffs[1]*forward_qs_cos['backbone1_gamma'] + fm_coeffs[2]}  # ,
        # 'backbone2_beta_epsilon_3J': fm_coeffs[3]*forward_qs_cos['backbone2_beta_epsilon']**2 + fm_coeffs[4]*forward_qs_cos['backbone2_beta_epsilon'] + fm_coeffs[5],
        # 'sugar_3J': fm_coeffs[6]*forward_qs_cos['sugar']**2 + fm_coeffs[7]*forward_qs_cos['sugar'] + fm_coeffs[8] }

    return forward_qs_out

infos['global']['forward_model'] = forward_model_fun

infos['global']['names_ff_pars'] = ['sin alpha', 'cos alpha']

def ff_correction(pars, f):
    out = jnp.matmul(pars, (f[:, [0, 6]] + f[:, [1, 7]] + f[:, [2, 8]]).T)
    return out

def ff_correction_hexamers(pars, f):
    out = jnp.matmul(pars, (f[:, [0, 10]] + f[:, [1, 11]] + f[:, [2, 12]] + f[:, [3, 13]] + f[:, [4, 14]]).T)
    return out

infos['global']['ff_correction'] = ff_correction
infos['UCAAUC'] = {'ff_correction': ff_correction_hexamers}

data = load_data(infos, stride=stride)

print(data)

#%% 2. normalize observables
# then, find optimal solution at given alpha

if if_normalize:

    list_name_mol = list(data.mol.keys())

    for name_mol in list_name_mol:
        out = normalize_observables(data.mol[name_mol].gexp, data.mol[name_mol].g, weights=data.mol[name_mol].weights)

        data.mol[name_mol].g = out[0]
        data.mol[name_mol].gexp = out[1]
        data.mol[name_mol].normg_mean = out[2]
        data.mol[name_mol].normg_std = out[3]

if if_reduce:

    s = 'backbone1_gamma_3J'

    assert list(data.mol.keys()) == ['AAAA']
    assert list(data.mol['AAAA'].g.keys()) == [s]

    data.mol['AAAA'].gexp[s] = data.mol['AAAA'].gexp[s][:2, :]
    data.mol['AAAA'].g[s] = data.mol['AAAA'].g[s][:, :2]
    # data.mol['AAAA'].normg_mean[s] = data.mol['AAAA'].normg_mean[s][:2]
    # data.mol['AAAA'].normg_std[s] = data.mol['AAAA'].normg_std[s][:2]
    data.mol['AAAA'].n_experiments[s] = 2

if alpha is not None:
    result = minimizer(data, alpha=alpha)

    lambdas = result.min_lambdas
    x0 = flat_lambda(lambdas)

else:
    assert beta is not None
    result = minimizer(data, regularization={'force_field_reg': 'KL divergence'}, beta=beta)

    x0 = result.pars

#%% 3. run Metropolis sampling

def proposal(x0, dx=0.01):
    x_new = x0 + dx*np.random.normal(size=len(x0))
    return x_new

proposal_full = {'fun': proposal, 'args': ([dx])}

if alpha is not None:

    def energy_fun_ER(lambdas, if_Jeffreys):  # energy_fun output must be a tuple, 2nd output None if no quantities are computed
        """ there are some inner variables previously defined but non as function input, like alpha """
        
        out = loss_function(np.zeros(2), data, regularization=None, alpha=alpha, fixed_lambdas=lambdas, if_save=True)
        
        energy = out.loss_explicit

        av_g = unwrap_2dict(out.av_g)[0] + [np.float(out.D_KL_alpha['AAAA'])]

        if if_Jeffreys:
            name_mol = list(out.weights_new.keys())[0]
            measure, cov = compute_sqrt_det(data.mol[name_mol].g, out.weights_new[name_mol], if_cholesky=True)
            energy -= np.log(measure)
        
        return energy, av_g

    energy_function = {'fun': energy_fun_ER, 'args': ([if_Jeffreys])}

else:

    name_mols = data.properties.system_names
    assert len(name_mols) == 1, 'this script works only for 1 molecular system'
    name_mol = name_mols[0]

    ff_correction = data.mol[name_mol].ff_correction
    fun_forces = jax.jacfwd(ff_correction, argnums=0)

    def energy_fun_FFF(pars, if_Jeffreys):
        """ there are some inner variables previously defined but non as function input, like beta """

        out = loss_function(pars, data, regularization={'force_field_reg': 'KL divergence'}, beta=beta, if_save=True)
        
        energy = out.loss  # which is loss_explicit if alpha is infinite

        av_g = unwrap_2dict(out.av_g)[0] + [np.float(out.reg_ff['AAAA'])]

        if if_Jeffreys:
            name_mol = list(out.weights_new.keys())[0]
            measure, cov = compute_sqrt_det((fun_forces, pars, data.mol[name_mol].f), out.weights_new[name_mol])
            energy -= np.log(measure)
        
        return energy, av_g

    energy_function = {'fun': energy_fun_FFF, 'args': ([if_Jeffreys])}

sampling = run_Metropolis(x0, proposal_full, energy_function, n_steps=n_steps)

#%% 4. save output

s = datetime.datetime.now()
date = s.strftime('%Y_%m_%d_%H_%M_%S_%f')

path = 'Result_' + str(date)

if not os.path.exists(path): os.mkdir(path)
else: print('possible overwriting')

if alpha is not None:
    values = {'stride': stride, 'alpha ER': alpha, 'normalize?': if_normalize, 'reduce?': if_reduce, 'Jeffreys?': if_Jeffreys, 'dlambda': dx, 'n_steps': n_steps, 'av. acceptance': sampling[-1]}
else:
    values = {'stride': stride, 'beta FFF': beta, 'normalize?': if_normalize, 'reduce?': if_reduce, 'Jeffreys?': if_Jeffreys, 'dlambda': dx, 'n_steps': n_steps, 'av. acceptance': sampling[-1]}

temp = pandas.DataFrame(list(values.values()), index=list(values.keys()), columns=[date]).T
temp.to_csv(path + '/par_values')

np.save(path + '/trajectory', sampling[0])
np.save(path + '/energy', sampling[1])

if type(sampling[2]) is not float:  # if float, it is the average acceptance
    np.save(path + '/quantities', sampling[2])
