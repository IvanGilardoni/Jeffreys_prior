"""
Sample a hyper-ensemble of (canonical) ensembles by using a suitable uninformative prior,
namely, a prescription on how to count the ensembles (to measure distances between ensembles).

Basic functions.
"""

import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

def local_density(variab, weights, which_measure = 'jeffreys'):
    """
    This function computes the local density of ensembles in the cases of Ensemble Refinement or Force-Field Fitting.
    
    This density can be defined through the Jeffreys ``uninformative" prior (`which_measure = 'jeffreys'`):
    in these two cases, the Jeffreys prior is given by the square root of the determinant of the covariance matrix
    (of the observables in Ensemble Refinement or the generalized forces in Force-Field Fitting,
    where the generalized forces are the derivatives of the force-field correction with respect to the fitting coefficients).
    
    It includes also the possibility for the computation of the local density of ensembles with plain Dirichlet
    if `which_measure == 'dirichlet'`, or with the variation of the average observables if 
    `which_measure == 'average'`.

    Since we are always dealing with a real-value, symmetric and semi-positive definite matrix,
    its determinant is computed through the Cholesky decomposition (which is faster for big matrices):
    `triang` is such that `metric = triang * triang.T`, so `sqrt(det metric) = det(triang)`.

    Parameters
    -----------
    
    variab : numpy.ndarray, dict or tuple
        For Ensemble Refinement, `variab` is either the dictionary `data.mol[name_mol].g` to be unwrapped
        or directly the numpy array with the observables defined in each frame.
        
        For Force-Field Fitting and `which_measure == 'jeffreys' or 'dirichlet'`, `variab` is the tuple `(fun_forces, pars, f)` where:
            - `fun_forces` is the function for the gradient of the force-field correction with respect to `pars`
            (defined through Jax as `fun_forces = jax.jacfwd(ff_correction, argnums=0)` where `ff_correction = data.mol[name_mol].ff_correction`;
            you can compute it just once at the beginning of the MC sampling);
            - `pars` is the numpy.ndarray of parameters for the force-field correction;
            - `f` is the numpy.ndarray `data.mol[name_mol].f` with the terms required to compute the force-field correction.
        If `which_measure = 'average'`, then the observables are required, too, and `variab` is the tuple `(fun_forces, pars, f, g)`.

        See documentation of `MDRefine` at https://www.bussilab.org/doc-MDRefine/MDRefine/index.html for further details
        about the `data` object.

    weights : numpy.ndarray
        Numpy array with the normalized weights of each frame; this is the probability distribution
        at which you want to compute the Jeffreys prior, corresponding to the local density of ensembles.

    which_measure: str
        String variable, chosen among: `jeffreys`, `dirichlet` or `average`, indicating the prescription
        for the local density of ensembles (Jeffreys prior, plain Dirichlet, average observables).
    -----------

    Returns
    -----------

    measure : float
        The local density of ensembles at the given distribution `weights`, computed as specified by `which_measure`
        (Jeffreys prior by default).
    
    cov : numpy.ndarray
        The metric tensor for the chosen metrics defined by `which_measure` if `which_measure = 'jeffreys'` or `'dirichlet'`;
        the covariance matrix if `which_measure = 'average'`.
    """

    if which_measure == 'jeffreys' or (which_measure == 'average' and type(variab) is not tuple):
        # in this case, the density is given by computing the variance-covariance matrix of values
        # (either forces or observables)

        if type(variab) is tuple: values = variab[0](variab[1], variab[2])
        else:
            if type(variab) is dict: values = np.hstack([variab[s] for s in variab.keys()])
            elif type(variab) is np.ndarray and len(variab.shape) == 1: values = np.array([variab]).T
            else: values = variab

        av_values = np.einsum('ti,t->i', values, weights)
        cov = np.einsum('ti,tj,t->ij', values, values, weights) - np.outer(av_values, av_values)

        # exploit the Cholesky decomposition:
        # metric = triang*triang.T, so sqrt(det metric) = det(triang)
        try:  # it may happen: `Matrix is not positive definite` (zero due to round-off errors)
            triang = np.linalg.cholesky(cov)
            density = np.prod(np.diag(triang))
        except:
            density = np.sqrt(np.linalg.det(cov))

        if which_measure == 'average': density = density**2

        return density, cov

    elif which_measure == 'average' and type(variab) is tuple:
        # in this case, we are sampling Force-Field Fitting with 'average' measure of ensembles
        # so we have to compute the covariance matrix of observables and forces;
        # then, since it is not a square matrix in general, you cannot compute its det,
        # but you have to compute the sqrt of det (C.T C)
        
        assert len(variab) == 4

        forces = variab[0](variab[1], variab[2])
        
        if type(variab[3]) is dict: g = np.hstack([variab[s] for s in variab.keys()])
        else: g = variab[3]

        av_forces = np.einsum('ti,t->i', forces, weights)
        av_g = np.einsum('ti,t->i', g, weights)
        cov = np.einsum('ti,tj,t->ij', forces, g, weights) - np.outer(av_forces, av_g)
        
        metric = np.einsum('ji,ki->jk', cov, cov)
        triang = np.linalg.cholesky(metric)
        density = np.prod(np.diag(triang))

        return density, cov

    else:
        assert which_measure == 'dirichlet', 'error on `which_measure`'

        if type(variab) is tuple: values = variab[0](variab[1], variab[2])
        else:
            if type(variab) is dict: values = np.hstack([variab[s] for s in variab.keys()])
            else: values = variab

        av_values = np.einsum('ti,t->i', values, weights)
        metric = np.einsum('ti,tj,t->ij', values, values, weights**2) + np.sum(weights**2)*np.outer(av_values, av_values)
        met = np.einsum('i,tj,t->ij', av_values, values, weights**2)
        metric -= met + met.T

        try:
            triang = np.linalg.cholesky(metric)
            density = np.prod(np.diag(triang))
        except:
            density = np.sqrt(np.linalg.det(metric))
            if np.isnan(density): print('density is NaN because metric has been evaluated as ', metric)

        return density, metric

def local_density_old(variab, weights, if_cholesky = True, which_measure = 'jeffreys'):
    """ old implementation of `local_density`: it does not work for every case of Force-Field Fitting """
    
    if type(variab) is tuple: values = variab[0](variab[1], variab[2])
    else:
        try: values = np.hstack([variab[s] for s in variab.keys()])
        except : values = variab
    
    av_forces = np.einsum('ti,t->i', values, weights)
    
    # print('a', np.einsum('ti,tj,t->ij', values, values, weights) - np.outer(av_forces, av_forces))
    # print('b', np.einsum('ti,tj,t->ij', values, values, weights**2) + (len(weights) - 2)*np.outer(av_forces, av_forces))

    if which_measure == 'jeffreys' or which_measure == 'average':
        metric = np.einsum('ti,tj,t->ij', values, values, weights) - np.outer(av_forces, av_forces)
    
    elif which_measure == 'dirichlet': None
        # wrong!
        # metric = np.einsum('ti,tj,t->ij', values, values, weights**2) + (len(weights) - 2)*np.outer(av_forces, av_forces)

    if which_measure == 'average':
        metric = np.einsum('jk,jl->kl', metric, metric)
    
    if not if_cholesky: measure = np.sqrt(np.linalg.det(metric))
    else:  # alternatively, exploit the Cholesky decomposition
        # print(np.linalg.det(metric), metric)
        triang = np.linalg.cholesky(metric)
        measure = np.prod(np.diag(triang))

    return measure, metric

class compute_single:
    """
    Compute basic quantities of the ensemble parametrized by `lambdas` as the optimal solution
    of Ensemble Refinement with max. entropy principle.
    It works both for the 1d case (single observable) or the multi-dimensional case (more observables);
    in the first case the instance variables are all `float` except for the reweighted ensemble`P`,
    while in the second case there are also `numpy.ndarray` variables (when referring to more observables).

    Parameters
    ---------------
    
    lambdas : float or numpy.float64 or numpy.ndarray
        Value(s) of `lambdas` coefficient(s).
    
    P0 : numpy.ndarray
        The reference probability distribution (ensemble) `P0(x)`.
    
    g : numpy.ndarray
        The numpy array with the observables `g(x)`; its shape is (M, N)
        where M is the n. of observables and N the n. of frames.
        If `type(lambdas) is float`, then `g` can be just a 1d array of length N.
    
    gexp, sigma : float or numpy.ndarray
        The corresponding experimental values and associated uncertainties.
    
    alpha : float
        The `alpha` hyperparameter of Ensemble Refinement.
    """

    def __init__(self, lambdas, P0, g, gexp, sigma, alpha):

        if type(lambdas) in [float, np.float64, int]: lambdas = np.array([lambdas])
        if len(g.shape) == 1: g = np.array([g])
        
        P0 = P0/np.sum(P0)

        self.P = P0
        """`numpy.ndarray` reporting the new (reweighted) probability distribution."""

        if len(g.shape) == 1: self.P *= np.exp(-lambdas*g)  # np.exp(-np.dot(lambda_i, g[np.newaxis, 0]))
        else: self.P *= np.exp(-np.dot(lambdas, g))
        
        self.Z = np.sum(self.P)
        """`float` reporting the partition function `Z lambda`."""
        self.P = self.P/self.Z
        
        self.av_g = np.dot(g, self.P)
        """`float` or `numpy.ndarray` with the average value of the observables $\langle g \rangle$ at given `lambdas`."""

        self.av_g2 = np.dot(g**2, self.P)
        """`float` or `numpy.ndarray` with the average value of the observables squared $\langle g^2 \rangle$ at given `lambdas`."""

        self.var_g = self.av_g2 - self.av_g**2
        """`float` or `numpy.ndarray` with the variance of the observables at given `lambdas`."""

        # if type(self.var_g) is float:
        #     if self.var_g < 0:
        #         if self.var_g > -1e-6:  self.var_g = 0
        #         else: print('error'); return

        # else:
        #     if np.any(self.var_g) < 0:
        #         wh = np.where(self.var_g > 0)[0]
        #         if self.var_g[wh] > -1e-6:  self.var_g[wh] = 0
        #         else: print('error'); return
        
        self.std_g = np.sqrt(self.var_g)
        """`float` or `numpy.ndarray` with the standard deviation of the observables at given `lambdas`."""

        if len(self.var_g) == 1:
            self.var_g = float(self.var_g)
            self.std_g = float(self.std_g)
        
        
        self.chi2 = ((self.av_g - gexp)/sigma)**2
        """`float` or `numpy.ndarray` with the $\chi^2$ at given `lambdas`."""
    
        self.Srel = np.log(self.Z) + np.dot(lambdas, self.av_g)
        """`float` with the relative entropy at given `lambdas`."""

        self.lossf = 1/2*np.sum(self.chi2) - alpha*self.Srel
        """`float` with the loss function at given `lambdas`."""

        self.gamma = alpha*(np.log(self.Z) + np.dot(lambdas, gexp) + 1/2*alpha*(np.dot(lambdas, sigma))**2)
        """`float` with the value of Gamma function at given `lambdas`."""

        # if len(lambdas) == 1: self.jeffreys = self.std_g
        out = local_density(g.T, self.P)
        self.jeffreys = out[0]
        """ `float` reporting the local density of ensembles measured by the Jeffreys prior.
        Notice that in 1d it is given directly by the standard deviation of the observables,
        while in multi-dimension you have to take into account also the covariance between different observables,
        resulting in the square root of the determinant of the variance-covariance matrix."""
        
        self.cov = out[1]
        """ `numpy.ndarray` with the covariance matrix given by `local_density` with `which_measure = 'jeffreys'`."""

        self.dirichlet = local_density(g.T, self.P, which_measure='dirichlet')[0]
        """`float` with the local density of ensembles measured by the (plain) Dirichlet prior."""

        self.avg_density = local_density(g.T, self.P, which_measure='average')[0]
        """`float` with the local density of ensembles measured by the variation of the average values of the observables."""

def compute(my_lambdas, P0, g, gexp, sigma, alpha):
    """
    Do `compute_single` for every lambda specified by `my_lambdas`, in order to get a 1d or 2d grid
    of results as a function of lambda. In the 1d case, it stacks `compute_single` results, then
    rearrange as a single dictionary. In the 2d case, it does not rearrange.

    Parameters:
    -----------
    my_lambdas : numpy.ndarray or tuple
        In the 1d case, `my_lambdas` is a 1d numpy array
    
    P0, g : numpy.ndarray
        Original probability distribution and observables (see `compute_single`).
    
    gexp, sigma : float or numpy.ndarray
        Experimental values and uncertainties.

    alpha : float
        Hyperparameter value for Ensemble Refinement.
    """

    if not type(my_lambdas) is tuple:  # namely, 1d case

        results = []

        for lambda_i in my_lambdas:
            out = compute_single(lambda_i, P0, g, gexp, sigma, alpha)
            results.append(vars(out))
        
        # rearrange as a single dict (namely, invert the two indices of results)
        
        results_dict = {}

        for s in results[0].keys():
            # try: results_dict[s] = np.array([results[i][s][0] for i in range(len(results))])
            # except:
            results_dict[s] = np.array([results[i][s] for i in range(len(results))])

        return results_dict

    elif len(my_lambdas) == 2:

        results = {}

        for l1 in my_lambdas[0]:
            results[l1] = {}
            
            for l2 in my_lambdas[1]:
                results[l1][l2] = vars(compute_single(np.array([l1, l2]), P0, g, gexp, sigma, alpha))
        return results

        # then, to pass from results to results_dict in the 2d case, you can use sth like:
        # loss = [[results[l1][l2]['lossf'] for l1 in lambdas] for l2 in lambdas]
        # and analogously for the other keys

    else: return None

def run_Metropolis(x0, proposal, energy_function, quantity_function = lambda x: None, *, kT = 1.,
    n_steps = 100, seed = 1, i_print = 10000, if_tqdm = True):
    """
    This function runs a Metropolis sampling algorithm.
    
    Parameters
    -----------

    x0 : numpy.ndarray
        Numpy array for the initial configuration.
    
    proposal : function or float
        Function for the proposal move, which takes as input variables just the starting configuration `x0`
        and returns the new proposed configuration (trial move of Metropolis algorithm).
        Alternatively, float variable for the standard deviation of a (zero-mean) multi-variate Gaussian variable
        representing the proposed step (namely, the stride).

    energy_function : function
        Function for the energy, which takes as input variables just a configuration (`x0` for instance)
        and returns its energy; `energy_function` can return also some quantities of interest,
        defined on the input configuration.
    
    quantity_function : function
        Function used to compute some quantities of interest on the initial configuration.
        If `energy_function` has more than one output, `quantity_function` is ignored and the quantities
        of interest are the 2nd output of `energy_function` (in this way, they are computed together with
        the energy).
        Notice that `quantity_function` does not support other input parameters beyond the configuration;
        otherwise, you can use `energy_function`.

    kT : float
        Temperature of the Metropolis sampling algorithm.

    n_steps : int
        Number of steps of Metropolis.
    
    seed : int
        Seed for the random generation.
    
    i_print : int
        How many steps to print an indicator of the running algorithm (current n. of steps).
    -----------

    Returns
    -----------
    traj : numpy.ndarray
        Numpy array with the trajectory.

    ene : numpy.ndarray
        Numpy array with the energy.

    av_alpha : float
        Average acceptance.

    quantities : numpy.ndarray, optional
        Numpy array with the quantities of interest (if any).
    """

    if energy_function is None:
        # energy_function = {'fun': lambda x : 0, 'args': ()}
        energy_function = lambda x : 0

    if type(proposal) is float:
        
        proposal_stride = proposal
        # def fun_proposal(x0, dx = 0.01):
        #     x_new = x0 + dx*np.random.normal(size=len(x0))
        #     return x_new

        # proposal = {'fun': fun_proposal, 'args': ([proposal])}

        def proposal(x0):
            x_new = x0 + proposal_stride*np.random.normal(size=len(x0))
            return x_new

    rng = np.random.default_rng(seed)

    x0_ = +x0  # in order TO AVOID OVERWRITING!
    
    traj = []
    ene = []
    quantities = []
    av_alpha = 0

    traj.append([])
    traj[-1] = +x0_

    # energy_function may have more than one output
    # out = energy_function['fun'](x0_, *energy_function['args'])
    out = energy_function(x0_)
    u0 = out[0]

    if len(out) == 2:
        print('Warning: the quantities of interest are given by energy_function')  #  and not by quantity_function')
        q0 = out[1]  # if `energy_function` has more than one output, the second one is the quantity of interest
    else: q0 = quantity_function(x0_)
    
    ene.append([])
    ene[-1] = +u0

    quantities.append([])
    quantities[-1] = q0

    counter = range(n_steps)
    if if_tqdm: counter = tqdm(counter)

    for i_step in counter:

        x_try = +proposal(x0_)  # proposal['fun'](x0_, *proposal['args'])

        out = energy_function(x_try)  # energy_function['fun'](x_try, *energy_function['args'])
        u_try = out[0]

        alpha = np.exp(-(u_try - u0)/kT)
        
        if alpha > 1: alpha = 1
        if alpha > rng.random():
            av_alpha += 1
            x0_ = +x_try
            u0 = +u_try

            if len(out) == 2: q0 = out[1]
            else: q0 = quantity_function(x0_)
        
        # traj.append(x0_)
        # to avoid overwriting!
        traj.append([])
        traj[-1] = +x0_
        
        ene.append([])
        ene[-1] = +u0

        quantities.append([])
        quantities[-1] = q0

        if (not if_tqdm) and (np.mod(i_step, i_print) == 0): print(i_step)

    av_alpha = av_alpha/n_steps
    
    if quantities[0] is None: return np.array(traj), np.array(ene), av_alpha
    else: return np.array(traj), np.array(ene), av_alpha, np.array(quantities)

def langevin_sampling(energy_fun, starting_x, n_iter : int = 10000, gamma : float = 1e-1,
    dt : float = 5e-3, kT : float = 1., seed : int = 1, if_tqdm: bool = True):
    """
    Perform a Langevin sampling of `energy_fun` at temperature `kT` (with Euler-Maruyama scheme).
    
    Parameters
    ----------
    energy_fun : function
        The energy function, written with `jax.numpy` in order to do automatic differentiation
        through `jax.grad` (this requires `energy_fun` to return a scalar value and not an array,
        otherwise you should use `jax.jacfwd` for example; to this aim, you can do 
        `jnp.sum(energy_fun(x))`).
    
    starting_x : numpy.ndarray
        The starting configuration of the Langevin sampling.
    
    n_iter : int
        Number of iterations.
    
    gamma : float
        Friction coefficient.
    
    dt : float
        Time step.
    
    kT : float
        The temperature.
    
    Seed : int
        Integer value for the seed.
    """

    jax_energy_fun = lambda x : jnp.sum(energy_fun(x))  # to use jax.grad rather than jax.jacfwd

    rng = np.random.default_rng(seed)
    grad = jax.grad(jax_energy_fun)

    sigma = np.sqrt(2*kT*gamma)
    step_width = sigma*np.sqrt(dt)

    traj = []
    ene_list = []
    force_list = []

    # x = jnp.array(starting_x)
    x = +starting_x
    force = -grad(x)

    traj.append(x)
    ene_list.append(jax_energy_fun(x))
    force_list.append(force)

    counter = range(n_iter)
    if if_tqdm: counter = tqdm(counter)
    
    for i in counter:
        r = rng.normal(size=len(x))
        x += gamma*force*dt + step_width*r
        force = -grad(x)

        traj.append(x)
        ene_list.append(jax_energy_fun(x))
        force_list.append(force)

    # check: steps not too big!!
    dif = np.ediff1d(traj)
    mean = np.mean(dif)
    std = np.std(dif)
    check = {'dif': dif, 'mean': mean, 'std': std}

    traj = np.array(traj)
    if len(x) == 1: traj = traj[:, 0]

    return traj, np.array(ene_list), force_list, check

def block_analysis(x, size_blocks = None, n_conv = 50):
    """
    This function performs the block analysis of a (correlated) time series `x`, cycling over different block sizes.
    It includes also a numerical search of the optimal estimated error `epsilon`, by smoothing `epsilon` and searching
    for the first time it decreases, which should correspond to a plateau region.

    Parameters
    -----------

    x : numpy.ndarray
        Numpy array with the time series of which you do block analysis.

    size_blocks : list, int or None
        The list with the block sizes used in the analysis; you can either pass an integer value,
        in this case the list of sizes is given by `np.arange(1, np.int64(size/2) + size_blocks, size_blocks)`;
        further, if `size_blocks` is `None`, the list of sizes is `np.arange(1, np.int64(size/2) + 1, 1)`.

    n_conv : int
        Length (as number of elements in the block-size list) of the kernel used to smooth the epsilon function
        (estimated error vs. block size) in order to search for the optimal epsilon, corresponding to the plateau.
    -----------

    Returns
    -----------
    
    mean : float
        Mean value of the time series.

    std : float
        Standard deviation of the time series (assuming independent frames).

    opt_epsilon : float
        Optimal estimate of the associated error epsilon.

    epsilon : list
        List with the associated error `epsilon` for each block size.

    smooth : numpy.ndarray
        Smoothing of the `epsilon` function.

    n_blocks : list
        List with the number of blocks in the time series, for each analysed block size.

    size_blocks : list
        List with the block sizes initially defined.
    """

    size = len(x)
    mean = np.mean(x)
    std = np.std(x)/np.sqrt(size)

    if size_blocks is None: size_blocks = np.arange(1, np.int64(size/2) + 1, 1)
    elif type(size_blocks) is int: size_blocks = np.arange(1, np.int64(size/2) + size_blocks, size_blocks)
    else: assert type(size_blocks) is list, 'incorrect size_blocks'

    n_blocks = []
    epsilon = []

    for size_block in size_blocks:

        n_block = np.int64(size/size_block)
        
        # a = 0 
        # for i in range(n_block):
        #     a += (np.mean(x[(size_block*i):(size_block*(i+1))]))**2
        # 
        # epsilon.append(np.sqrt((a/n_blocks[-1] - mean**2)/n_blocks[-1]))

        block_averages = []
        for i in range(n_block): block_averages.append(np.mean(x[(size_block*i):(size_block*(i+1))]))
        block_averages = np.array(block_averages)

        n_blocks.append(n_block)
        epsilon.append(np.sqrt((np.mean(block_averages**2) - np.mean(block_averages)**2)/n_block))

    # find the optimal epsilon: smooth the epsilon function and find the first time it decreases
    kernel = np.ones(n_conv)/n_conv
    smooth = np.convolve(epsilon, kernel, mode='same')
    diff = np.ediff1d(smooth)
    wh = np.where(diff < 0)
    opt_epsilon = smooth[wh[0][0]]
    
    return mean, std, opt_epsilon, epsilon, smooth, n_blocks, size_blocks
