"""
Script to perform Bayesian sampling of the optimal ensembles parametrized by lambda coefficients
(Ensemble Refinement only or Force-Field Fitting only).
"""

import os, datetime, sys, argparse, time
import numpy as np
import jax
import jax.numpy as jnp
import pandas

sys.path.append('../loss_function_complete/')

from MDRefine.MDRefine import load_data, normalize_observables, minimizer, loss_function, unwrap_2dict
from Functions.basic_functions_bayesian import run_Metropolis, local_density, Saving_function

def save_dict_to_txt(my_dict, txt_path, sep : str=' '):
    """
    Save a dictionary as a txt file with column names given by indicization of dict keys.
    Each item value should be 0- or 1-dimensional (either int, float, np.ndarray or list),
    not 2-dimensional or more.
    """

    header = []
    values = []

    for key, arr in my_dict.items():
        if (type(arr) is int) or (type(arr) is float):
            header.append(key)
            values.append(arr)
        else:
            # assert ((type(arr) is np.ndarray) and (len(arr.shape) == 1)) or (type(arr) is list), 'error on element with key %s' % key
            # you could also have jax arrays, so manage as follows:

            try:
                l = len(arr.shape)
            except:
                l = 0
            assert (l == 1) or (type(arr) is list), 'error on element with key %s' % key
            
            # you should also check that each element in the list is 1-dimensional
            for i, val in enumerate(arr, 1):
                header.append(f"{key}_{i}")
                values.append(val)

    with open(txt_path, 'w') as f:
        f.write(sep.join(header) + '\n')
        f.write(sep.join(str(v) for v in values) + '\n')

    return

def flat_lambda(lambdas):

    flatten_lambda = []
    for name_mol in data.properties.system_names:
        flatten_lambda = flatten_lambda + list(
            np.hstack([lambdas[name_mol][k] for k in data.mol[name_mol].n_experiments.keys()]))

    flatten_lambda = np.array(flatten_lambda)

    return flatten_lambda

#%% 0. input values

parser = argparse.ArgumentParser()

parser.add_argument('--jobid', type=str, required=False, default=1, help="SLURM job ID")  # optional job_id

# parser.add_argument('params', nargs=7, type=float, help='List of parameters')
parser.add_argument('stride', type=int, help='Example: 1')
parser.add_argument('alpha', type=float, help='Example: 10.')  # hyperparameter value for Ensemble Refinement loss function
# parser.add_argument('beta', type=float, help='Example: 10.')  # only for force-field fitting
parser.add_argument('if_normalize', type=int, help='1 True, 0 False; if you want to normalize observables')
parser.add_argument('if_reduce', type=int, help='1 True, 0 False; if you want to take just 2 observables')
parser.add_argument('if_onebyone', type=int, help='1 True, 0 False; if the proposal move is done cycling on each coordinate')
parser.add_argument('dx', type=float, help='Example: 0.2')  # standard deviation of the normal distribution for the proposal
parser.add_argument('which_measure', type=int, help='0 for plain sampling on lambdas, 1 for Jeffreys, 2 for Dirichlet, 3 for average')
# `which_measure` rather than `if_jeffreys`
parser.add_argument('n_steps', type=int, help='n. steps in the Metropolis sampling')
parser.add_argument('--seed', type=int, required=False, default=np.random.randint(1000), help='seed (random state)')

args = parser.parse_args()

print(f"Running job with SLURM ID: ${args.jobid}")

# seed = np.random.randint(1000)
rng = np.random.default_rng(args.seed)

#%% 1. load data

infos = {'global': {
    'path_directory': '../loss_function_complete/DATA',
    'system_names': ['AAAA'],  # , 'CAAU'],  # , 'CCCC', 'GACC', 'UUUU', 'UCAAUC'],
    'g_exp': ['backbone1_gamma_3J', 'backbone2_beta_epsilon_3J', 'sugar_3J'],  # , 'NOEs'],# , ('uNOEs', '<')],
    'forward_qs': ['backbone1_gamma', 'backbone2_beta_epsilon','sugar'],
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
            forward_qs_cos[type_name] = forward_qs_cos[type_name][:, selected_obs[type_name+'_3J']]

    # 2. compute observables (forward_qs_out) through forward model
    forward_qs_out = {
        s + '_3J' : fm_coeffs[0]*forward_qs_cos[s]**2 + fm_coeffs[1]*forward_qs_cos[s] + fm_coeffs[2] for s in forward_qs_cos.keys()}
    
        # that is:
        # 'backbone1_gamma_3J': fm_coeffs[0]*forward_qs_cos['backbone1_gamma']**2 + fm_coeffs[1]*forward_qs_cos['backbone1_gamma'] + fm_coeffs[2],
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
# infos['UCAAUC'] = {'ff_correction': ff_correction_hexamers}

data = load_data(infos, stride=args.stride)

#%% 2. normalize observables
# then, find optimal solution at given alpha

if args.if_normalize:

    list_name_mol = list(data.mol.keys())

    for name_mol in list_name_mol:
        out = normalize_observables(data.mol[name_mol].gexp, data.mol[name_mol].g, weights=data.mol[name_mol].weights)

        data.mol[name_mol].g = out[0]
        data.mol[name_mol].gexp = out[1]
        data.mol[name_mol].normg_mean = out[2]
        data.mol[name_mol].normg_std = out[3]

if args.if_reduce:

    assert list(data.mol.keys()) == ['AAAA'], 'ensure you have loaded just AAAA molecule'

    s = 'backbone1_gamma_3J'
    my_mol = data.mol['AAAA']
    # to avoid repeating `data.mol['AAAA']` you can act on `my_mol`, with the same results on `data.mol['AAAA']`
    # because this does not create a copy, just a reference to the original object!

    assert s in list(my_mol.g.keys()), 'ensure you have loaded just one kind of observables'

    my_mol.gexp = {s: my_mol.gexp[s]}
    my_mol.g = {s: my_mol.g[s]}
    my_mol.names = {s: my_mol.names[s]}
    my_mol.ref = {s: my_mol.ref[s]}
    my_mol.forward_qs = {s[:-3]: my_mol.forward_qs[s[:-3]]}
    my_mol.n_experiments = {s: my_mol.n_experiments[s]}

    my_mol.gexp[s] = my_mol.gexp[s][:2, :]
    my_mol.g[s] = my_mol.g[s][:, :2]
    my_mol.names[s] = my_mol.names[s][:2]
    my_mol.normg_mean[s] = my_mol.normg_mean[s][:2]
    my_mol.normg_std[s] = my_mol.normg_std[s][:2]
    my_mol.n_experiments[s] = 2

    # just a final check
    assert (len(list(data.mol['AAAA'].g.keys())) == 1) and (len(data.mol['AAAA'].names[s]) == 2), 'error in reduction'

if args.alpha is not None:
    result = minimizer(data, alpha=args.alpha)

    lambdas = result.min_lambdas
    x0 = flat_lambda(lambdas)

else:
    assert args.beta is not None
    result = minimizer(data, regularization={'force_field_reg': 'KL divergence'}, beta=args.beta)

    x0 = result.pars

#%% 3. define proposal and energy_fun

if not args.if_onebyone: proposal = args.dx
else: proposal = ('one-by-one', args.dx)

if args.alpha is not None:

    def energy_fun_ER(lambdas, which_measure):  # energy_fun output must be a tuple, 2nd output None if no quantities are computed
        """ there are some inner variables previously defined but non as function input, like alpha """
        
        out = loss_function(np.zeros(2), data, regularization=None, alpha=args.alpha, fixed_lambdas=lambdas, if_save=True)
        
        energy = out.loss_explicit

        av_g = unwrap_2dict(out.av_g)[0] + [np.float64(out.D_KL_alpha['AAAA']), out.loss_explicit]

        if (args.which_measure in [1, 2, 3]):
            name_mol = list(out.weights_new.keys())[0]
            
            if args.which_measure == 1 : which_measure = 'jeffreys'
            elif args.which_measure == 2 : which_measure = 'dirichlet'
            else: which_measure = 'average'
            
            measure, cov = local_density(data.mol[name_mol].g, out.weights_new[name_mol], which_measure=which_measure)
            energy -= np.log(measure)
        
        return energy, av_g

    energy_function = lambda x : energy_fun_ER(x, args.which_measure)

else:

    name_mols = data.properties.system_names
    assert len(name_mols) == 1, 'this script works only for 1 molecular system'
    name_mol = name_mols[0]

    ff_correction = data.mol[name_mol].ff_correction
    fun_forces = jax.jacfwd(ff_correction, argnums=0)

    def energy_fun_FFF(pars, if_Jeffreys):  # TO BE FIXED!!
        """ there are some inner variables previously defined but non as function input, like beta """

        out = loss_function(pars, data, regularization={'force_field_reg': 'KL divergence'}, beta=args.beta, if_save=True)
        
        energy = out.loss  # which is loss_explicit if alpha is infinite

        av_g = unwrap_2dict(out.av_g)[0] + [np.float64(out.reg_ff['AAAA'])]

        if if_Jeffreys:
            name_mol = list(out.weights_new.keys())[0]
            measure, cov = local_density((fun_forces, pars, data.mol[name_mol].f), out.weights_new[name_mol])
            energy -= np.log(measure)
        
        return energy, av_g

    energy_function = lambda x : energy_fun_FFF(x, args.if_Jeffreys)

#%% 4. saving folders
if args.if_reduce: path = 'Results_sampling_ER_reduced'
else: path = 'Results_sampling_ER'
if not os.path.exists(path): os.mkdir(path)

s = datetime.datetime.now()
date = s.strftime('%Y_%m_%d_%H_%M_%S_%f')
path = path + '/Result_' + str(date)

if not os.path.exists(path): os.mkdir(path)
else:
    print('possible overwriting')
    sys.exit()

if args.if_normalize:
    for name_mol in data.mol.keys():
        save_dict_to_txt(data.mol[name_mol].normg_mean, path + '/normg_mean_%s_%i' % (name_mol, args.stride))
        save_dict_to_txt(data.mol[name_mol].normg_std, path + '/normg_std_%s_%i' % (name_mol, args.stride))

#%% 5. run_Metropolis and save output results

t0 = time.time()

saving = Saving_function(vars(args), t0, date, path)

sampling = run_Metropolis(x0, proposal, energy_function, n_steps=args.n_steps, seed=args.seed, saving=saving)

