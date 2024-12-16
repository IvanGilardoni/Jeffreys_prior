"""
Simulate long trajectories with Metropolis for different particle numbers and
compute their static structure factor.
Basic functions.
"""

import numpy as np

rng = np.random.default_rng(1)
# rng is defined here so that one can explicitely call `proposal`;
# then, you can select the seed for `run_Metropolis`.

def compute_Sk(n_sites, n_particles, traj, stride = 1, if_fast = True):
    """
    Compute the static structure factor S(k).
    
    Parameters:
        - n_sites: integer
            n. of sites of the lattice, used to make the array of values for k;
        - n_particles: integer
            n. of particles in the system, used in the equation for S(k);
        - traj: numpy array
            the trajectory (M x N) where M is the n. of frames and N is the n. of particles;
            `traj[i, j]` is the position of particle n. j at frame n. i;
        - stride: integer
            stride between frames in the trajectory employed to get S(k);
        - if_fast: boolean
            if True, compute the static structure factor using a sped-up algorithm;
            if False, use the "basic" algorithm and returns also the "Sk values" for each particle.
            To run faster, sum firtly (namely, in the inner loop) over the frames, this will reduce the n. of computations.
    Return:
        - ks: numpy array
            array of k values;
        - S_k: numpy array
            array of S(k) static structure factor;
        - if if_fast, S_k_list: numpy array
            array (n_sites x M') where n_sites is the n. of lattice sites, equal to the length of k,
            and M' is the n. of employed frames in the trajectory; S_k_list[i] is the static structure
            factor computed for the i-th frame in the trajectory (which is then averaged over frames
            to get S_k).
    """
    
    print('computing Sk...')

    # ks = 2*np.pi/n_sites*np.arange(n_sites)
    ks = np.pi/n_sites*np.arange(n_sites)  # since it is periodic S(2\pi - k) = S(k)
    # ks = np.arange(0, np.pi, 0.05)  # no sense to make thicker stride than the lattice site

    if not if_fast:

        S_k_list = []

        for i, xs in enumerate(traj[::stride]):

            # if np.mod(i, 100) == 0: print(i)

            S_k_list.append([])
            
            for k in ks:
                rho_k = np.sum(np.exp(1j*k*xs))
                S_k = np.abs(rho_k)**2

                S_k_list[-1].append(S_k)

        S_k_list = 1/n_particles*np.array(S_k_list)
        S_k = np.mean(S_k_list, axis=0)

        print('done')

        return ks, S_k, S_k_list
    
    else:

        S_k = []

        # a_iij = np.einsum('jk,ji->jik', traj, -traj)  # this does not do what I want

        # a_tij = np.zeros((traj.shape[0], traj.shape[1], traj.shape[1]))
        # for i1 in range(traj.shape[1]):
        #     for i2 in range(traj.shape[1]):
        #         for t in range(traj.shape[0]):
        #             a_tij[t, i1, i2] = traj[t, i1] - traj[t, i2]

        a_tij = traj[:, None, :] - traj[:, :, None]  # this is as the above 5 lines but faster

        for k in ks:
            val = np.einsum('ik->', np.einsum('jik->ik', np.exp(1j*k*a_tij)))/(n_particles*np.shape(traj)[0])
            S_k.append(np.real_if_close(val))

        return ks, S_k

def proposal(x0, n_sites, val = 0.3, which_strategy = 0):
    """
    Propose a move (to Metropolis algorithm) for a 1d lattice with PBCs as it is an ideal gas
    (where each particle occupy a single site).
    The way I implemented `run_Metropolis` requires the proposal move to be symmetric.

    Parameters:
        - x0: numpy array
            the starting configuration, an array of length M where M is the n. of particles
            in the system; x0[i] is the position (lattice site) occupied by particle n. i;
        - n_sites: integer
            the number of sites in the lattice;
        - val: float
            2*val is the probability for each particle to move;
            the remaining probability 1 - 2*val is the one to be in the same site;
            you can select val = M/n_particles so that at each step M particles are moved on average
            (useful when the particles start to saturate the full available space).
        - which_strategy: integer
            strategy used for the move: 0 for moving randomly selected particles by 1 step,
            1 for moving them randomly.
    
    Return:
        - x1: numpy array
            the new configuration.
    """

    assert val <= 0.5, 'error: val must be < 0.5'

    moves = rng.uniform(size=len(x0))
    
    x1 = +x0

    if which_strategy == 0:
        x1[moves < val] -= 1
        x1[moves > 1 - val] += 1
    else:
        new_pos = rng.integers(0, n_sites, len(moves[moves < 2*val]))
        x1[moves < 2*val] = new_pos

    x1 = np.mod(x1, n_sites)  # periodic boundary conditions

    return x1

def proposal_exclud_vol(x0, n_sites, val, which_strategy):
    """
    Propose a move (to Metropolis algorithm) for a 1d lattice with PBCs as it is a hard-sphere gas
    (where each particle occupy a single site).
    It works similarly to `proposal`, except that a new move is generated until the new configuration
    does not contain more than one particle on each lattice site.
    """
    
    b = 0
    
    while b == 0:
        x1 = proposal(x0, n_sites, val, which_strategy)
        diff = np.ediff1d(np.sort(x1))
        if len(np.where(diff == 0)[0]) == 0:
            b = 1
    
    return x1

def compute_NN_distances(xs, n_sites = None):
    """
    It computes the distances between the closest particles along the 1d lattice.
    
    Parameters:
        - xs: numpy array
            Configuration of particles along the 1d lattice.
        - n_sites: integer
            Number of lattice sites (used for PBCs, None if there are not PBCs). 
    """

    sorted_xs = np.sort(xs)
    vec = np.ediff1d(sorted_xs)

    if n_sites is not None:
        vec = np.append(vec, sorted_xs[0] + n_sites - sorted_xs[-1])

    return vec

def LJlike_energy(xs, ene0 = +2, ene1 = +1, n_sites = 100):
    """ wrong, because `compute_NN_distances` does not take into account second closest particles."""

    dist = compute_NN_distances(xs, n_sites)

    n_0 = len(np.where(dist == 1)[0])
    n_1 = len(np.where(dist == 2)[0])
    # print(n_0, n_1)

    energy = n_0*ene0 + n_1*ene1
    
    return energy

def energy_fun(x, energies_pen, n_sites, if_PBC = True):
    """ 
    Compute the energy of a 1d lattice configuration `x` as specified by `energies_pen`.
    
    Parameters:
        - x: numpy array
            This is the 1d lattice configuration.
        - energies_pen: dict
            How the energy is defined; for example `energies_pen = {'1': 3, '2': -5, '3': -1}` means
        for each couple of particles at relative distance 1 you have energy cost +3, and so on;
        - n_sites: int
            Number of lattice sites, required if you have PBCs `if_PBC = True`.
        - if_PBC: boolean
            True if you have Periodic Boundary Conditions.
    """

    en = 0

    my_dist = x[:, None] - x[None, :]

    if if_PBC:
        dist1 = np.abs(my_dist)
        dist2 = np.abs(my_dist + n_sites)
        dist3 = np.abs(my_dist - n_sites)

        my_dist = np.min(np.stack([dist1, dist2, dist3]), axis=0)

    for i in energies_pen.keys():
        how_many = (np.shape(my_dist)[0]**2 - np.count_nonzero(my_dist - int(i)))/2
        en += how_many*energies_pen[i]

    return en

def run_Metropolis(x0, proposal, energy_function, quantity_function = lambda x: None, *, kT = 1, n_steps = 100, seed = 1, i_print = 100):
    """
    Run a Metropolis sampling with initial configuration `x0`, proposal move `proposal` and
    `energy_function` energy, for a n. of steps `n_steps` and with temperature `kT`.
    If `energy_function` has more than one output, `quantity_function` is ignored and the quantities of interest are the 2nd output of `energy_function`
    (in this way, they are computed together with the energy).
    The first argument of `energy_function` is the sampling variable.
    """

    if energy_function is None: energy_function = {'fun': lambda x : 0, 'args': ()}

    rng = np.random.default_rng(seed)

    x0_ = +x0  # TO AVOID OVERWRITING!
    
    traj = []
    ene = []
    quantities = []
    av_alpha = 0

    traj.append([])
    traj[-1] = +x0_

    # u0 = energy_function(x0_)

    out = energy_function['fun'](x0_, *energy_function['args'])

    u0 = out[0]

    if len(out) == 2:
        print('Warning: the quantities of interest are given by energy_function and not by quantity_function')
        q0 = out[1]  # if `energy_function` has more than one output, the second one is the quantity of interest
    else: q0 = quantity_function(x0_)
    
    ene.append([])
    ene[-1] = +u0

    quantities.append([])
    quantities[-1] = q0

    for i_step in range(n_steps):

        # x_try = +proposal(x0_)
        # u_try = +energy_function(x_try)
        x_try = +proposal['fun'](x0_, *proposal['args'])

        out = energy_function['fun'](x_try, *energy_function['args'])
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

        if np.mod(i_step, i_print) == 0: print(i_step)

    av_alpha = av_alpha/n_steps
    
    if quantities[0] is None:
        return np.array(traj), np.array(ene), av_alpha
    else:
        return np.array(traj), np.array(ene), np.array(quantities), av_alpha

def compute_sqrt_det(variab, weights, if_cholesky = True):
    """
    Parameters:
    
        - variab is `data.mol[name_mol].g` for the observables and `(fun_forces, pars, f)` for the force-field corrections 
            where `ff_correction = data.mol[name_mol].ff_correction`, `f = data.mol[name_mol].f` and `fun_forces = jax.jacfwd(ff_correction, argnums=0)`
            the Jax function for the forces, compute it just once at the beginning of the MC sampling
        - weights
        - if_cholesky: if True, compute the determinant of the covariance matrix (which is real-value, symmetric and semi-positive definite)
            by means of the Cholesky decomposition (faster for big matrices)

    Return:

        - measure
        - cov
    """

    if type(variab) is tuple: values = variab[0](variab[1], variab[2])
    else: values = np.hstack([variab[s] for s in variab.keys()])
    
    av_forces = np.einsum('ti,t->i', values, weights)
    cov = np.einsum('ti,tj,t->ij', values, values, weights) - np.outer(av_forces, av_forces)

    if not if_cholesky:
        measure = np.sqrt(np.linalg.det(cov))
    else:  # alternatively, exploit the Cholesky decomposition
        triang = np.linalg.cholesky(cov)
        measure = np.prod(np.diag(triang))

    return measure, cov

def block_analysis(x, size_blocks: list = None, delta = 1, n_conv = 50):

    size = len(x)
    mean = np.mean(x)
    std = np.std(x)/np.sqrt(size)

    if size_blocks is None:
        size_blocks = np.arange(1, np.int64(size/2) + delta, delta)

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
        for i in range(n_block):
            block_averages.append(np.mean(x[(size_block*i):(size_block*(i+1))]))
        block_averages = np.array(block_averages)

        n_blocks.append(n_block)
        epsilon.append(np.sqrt((np.mean(block_averages**2)-np.mean(block_averages)**2)/n_block))

    # find the optimal epsilon: smooth the epsilon function and find the first time it decreases
    kernel = np.ones(n_conv)/n_conv
    smooth = np.convolve(epsilon, kernel, mode='same')
    diff = np.ediff1d(smooth)
    wh = np.where(diff < 0)
    opt_epsilon = smooth[wh[0][0]]
    
    return mean, std, opt_epsilon, epsilon, smooth, n_blocks, size_blocks