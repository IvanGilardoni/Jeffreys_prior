import numpy as np
import jax
import jax.numpy as jnp

from scipy.optimize import minimize
from scipy.special import erf, erfinv

from coretools import Result

def my_erf(x):
    return erf(x/np.sqrt(2))

def my_inv_erf(x):
    return np.sqrt(2)*erfinv(x)

from basic_functions_bayesian import compute, compute_single

#%%

def flatten(x, s, m = None):
    """
    Given a list or a dictionary `x`, flatten the `m`-th element (if not `None`) of its key/attribute `s`.
    """
    if m is None:
        if type(x) is dict:
            try: out = [vars(x[n])[s] for n in x.keys()]
            except: out = [x[n][s] for n in x.keys()]
        elif type(x) is list: out =[vars(x[n])[s] for n in range(len(x))]
    else:
        if type(x) is dict:
            try: out = [vars(x[n])[s][m] for n in x.keys()]
            except: out = [x[n][s][m] for n in x.keys()]
        elif type(x) is list: out = [vars(x[n])[s][m] for n in range(len(x))]
    return out

def loss_fun(lambdas, p0, g, gexp, sigma_exp, alpha, if_return_all = False, if_cov = False):
    
    if len(g.shape) == 1: g = np.array([g])

    # new_weights = p0*jnp.exp(-jnp.dot(lambdas, g))
    # new_z = jnp.sum(new_weights)

    correction = jnp.dot(lambdas, g)
    """ shift is such that the physical Z is = Z/np.exp(shift) """
    shift = jnp.min(correction)
    correction -= shift

    new_weights = jnp.exp(-correction)*p0

    # assert not np.isnan(new_weights).any(), 'Error: new_weights contains None'

    logZ = jnp.log(jnp.sum(new_weights)) - shift
    new_weights = new_weights/jnp.sum(new_weights)

    av_g = jnp.dot(g, new_weights)
    chi2 = jnp.sum(((av_g - gexp)/sigma_exp)**2)
    # dkl = jnp.sum(new_weights*jnp.log(new_weights/p0))  # numerically unstable!!
    dkl = -logZ - jnp.dot(lambdas, av_g)
    loss_val = 1/2*chi2 + alpha*dkl
    
    if if_cov:
        cov = jnp.einsum('it,jt,t->ij', g, g, new_weights) - jnp.outer(av_g, av_g)
        return loss_val, chi2, dkl, new_weights, cov

    if if_return_all: return loss_val, chi2, dkl, new_weights
    else: return loss_val

loss_grad_fun = jax.grad(loss_fun, argnums=0)

def loss_and_grad(lambdas, p0, g, gexp, sigma_exp, alpha, loss_grad_fun):
    loss_value = loss_fun(lambdas, p0, g, gexp, sigma_exp, alpha)
    loss_grad_value = loss_grad_fun(lambdas, p0, g, gexp, sigma_exp, alpha)
    return loss_value, loss_grad_value

def build_perimeter_old(x0, dx, n_x = 100):

    x_min = x0 - dx
    x_max = x0 + dx

    scan = [np.linspace(x_min[0], x_max[0], n_x), np.linspace(x_min[1], x_max[1], n_x)]

    perimeter = [np.vstack((scan[0], x_min[1]*np.ones(n_x))), np.vstack((x_max[0]*np.ones(n_x), scan[1])),
        np.vstack((scan[0], x_max[1]*np.ones(n_x))), np.vstack((x_min[0]*np.ones(n_x), scan[1]))]

    perimeter = np.hstack(perimeter)

    return perimeter

def build_perimeter_old_long(x0, dx, n_x = 100):
    """
    Given a center `x0`, an array with half length of each side `dx` and the n. of steps for each side
    `n_x`, build the perimeter.

    Parameters
    ----------
    x0 : array_like
        Numpy 1d. array with the central point.
    dx : array_like
        Numpy 1d. array with half length of each side.
    n_x : int
        Integer for the n. of steps along each side (equal for all the sides).
    
    Returns
    -------
    peri : array_like
        Numpy 2d. array (M, N) with the perimeter (with M the n. of coordinates, namely, the dimensions).
    """

    x_min = x0 - dx
    x_max = x0 + dx

    """ 1. make a grid """
    """ this is limited to 3 dimensions """
    # scan = []
    # for i in range(len(x0)):
    #     scan.append(np.linspace(x_min[i], x_max[i], n_x))

    # grid = []

    # for x1 in scan[0]:
    #     for x2 in scan[1]:
    #         for x3 in scan[2]:
    #             grid.append(np.array([x1, x2, x3]))

    # grid = np.array(grid)

    """ this is the same as the commented lines above but for generic dimensionality """
    scan = [np.linspace(x_min[i], x_max[i], n_x) for i in range(len(x_min))]

    grid = np.array(np.meshgrid(*scan))
    grid = grid.reshape(len(x0), n_x**len(x0))

    """ 2. remove inner elements to get the perimeter (boundary blocks) """ 
    set_list = []

    for i in range(len(x0)):
        set_list.extend([np.where(grid[:, i] != scan[i][0])[0], np.where(grid[:, i] != scan[i][-1])[0]])

    set_list = [set(list(s)) for s in set_list]
    set_whc = set.intersection(*set_list)

    list_wh = list(set(list(np.arange(len(grid)))) - set_whc)

    peri = grid[list_wh]

    return peri

def build_perimeter(x0, dx, n_x = 100):
    """
    Given a center `x0`, an array with half length of each side `dx` and the n. of steps for each side
    `n_x`, build the perimeter.

    Parameters
    ----------
    x0 : array_like
        Numpy 1d. array with the central point.
    dx : array_like
        Numpy 1d. array with half length of each side.
    n_x : int
        Integer for the n. of steps along each side (equal for all the sides).
    
    Returns
    -------
    peri : array_like
        Numpy 2d. array (M, N) with the perimeter (with M the n. of coordinates, namely, the dimensions).
    """

    x_min = x0 - dx
    x_max = x0 + dx

    scans = [np.linspace(x_min[i], x_max[i], n_x) for i in range(len(x_min))]

    edges = []

    for i in range(len(scans)):
        arr = scans[:]
        del arr[i]
        edges.append(np.array(np.meshgrid(*arr)))
        edges[-1] = edges[-1].reshape(len(x0) - 1, n_x**(len(x0) - 1))

    peri = []

    for i in range(len(scans)):
        peri.append(np.insert(edges[i], i, scans[i][0]*np.ones(edges[i].shape[1])[None, :], axis=0))
        peri.append(np.insert(edges[i], i, scans[i][-1]*np.ones(edges[i].shape[1])[None, :], axis=0))

    peri = np.hstack(np.array(peri))

    return peri

class numerical_props(Result):  # old name: compute_depth
    def __init__(self, n_frames, sigma : float or np.ndarray, gexp, sigma_exp, alpha, delta_lambda = 500,
        n_perim = 100, if_scan = False):
        """
        Compute main properties of the posterior distribution, proportional to `np.exp(-loss_fun)`.
        The reference probability distribution is assumed to be uniform.
        
        Parameters
        ----------
        n_frames : int
            Integer variable for the total n. of frames.
        
        sigma : float or array_like
            The standard deviation of the observable (or the array of them), used to generate the array of values
            for the observables `g` as a (multivariate) normal distribution centered in `np.zeros(n_frames)`.
            Alternatively, if `len(sigma) == n_frames`, then `sigma` is rather the `g` array of observables
            itself, without random generation of it.
        
        gexp : float or array_like
            The experimental values of the observables.

        sigma_exp : float or array_like
            The estimated experimental errors for `gexp`.
        
        alpha : float
            The hyperparameter value.
        
        delta_lambda : float

        n_perim : int

        if_scan : bool
            Boolean variable, if True then compute the loss in the whole grid.
        """
        
        super().__init__()

        if (type(sigma) is float) or (type(sigma) is np.ndarray and len(sigma.shape) == 1 and sigma.shape[0] == 1):
            g = np.random.normal(0, sigma, n_frames)
            g = np.array([g])  # to have the same shape for one or more observables
        elif sigma.shape[0] == n_frames: g = sigma.T
        elif (len(sigma.shape) == 2) and (sigma.shape[1] == n_frames): g = sigma
        else:
            if len(sigma.shape) == 1: cov = np.diag(sigma**2)
            else: cov = sigma
            g = np.random.multivariate_normal(mean=np.zeros(cov.shape[0]), cov=cov, size=n_frames).T

        self.g = g
        """ The 2d. `numpy.ndarray` with the values of the observables, whose shape is `(M, N)` where `M` is
        the n. of observables and `N = n_frames`. """

        p0 = np.ones(n_frames)/n_frames
        # The 1d. `numpy.ndarray` with the reference distribution, assumed to be uniform.

        self.mini = minimize(loss_and_grad, np.zeros(g.shape[0]), args=(p0, g, gexp, sigma_exp, alpha, loss_grad_fun), jac=True, method='BFGS')
        """ The result of the minimization `scipy.optimize.minimize` of the `loss_fun` (through `loss_and_grad`)
        with input parameters previously defined (`p0, g, gexp, sigma_exp, alpha`). """

        self.min_lambda = self.mini.x
        """ The 1d. `numpy.ndarray` with the point of minimum of the loss function, determined by the minimization. """
        
        self.min_loss = self.mini.fun
        """ The min. value of the loss function, determined by the minimization. """

        out = loss_fun(self.min_lambda, p0, g, gexp, sigma_exp, alpha, if_return_all=True)

        self.min_avg = np.einsum('ij,j', g, out[3])
        """ The 1d. `numpy.ndarray` with the average values of the observables in the point of min. of `loss_fun`. """

        if g.shape[0] == 1:

            g = g[0, :]

            x_min = np.argwhere(g == np.min(g))[0]
            x_max = np.argwhere(g == np.max(g))[0]

            self.gbar = np.max(g)
            
            self.lim_loss = [-alpha*np.log(p0[x_max]) + 1/2*((np.max(g) - gexp)/sigma_exp)**2,
                -alpha*np.log(p0[x_min]) + 1/2*((np.min(g) - gexp)/sigma_exp)**2]

            lambda_max = self.min_lambda + delta_lambda
            lambda_min = self.min_lambda - delta_lambda

            val1 = loss_fun(lambda_min, p0, g, gexp, sigma_exp, alpha, True)
            val2 = loss_fun(lambda_max, p0, g, gexp, sigma_exp, alpha, True)

            self.lim_loss_num = [val1[0], val2[0]]
            self.lim_chi2 = [val1[1], val2[1]]
            self.lim_dkl = [val1[2], val2[2]]

            diff = np.abs((self.lim_loss_num[0] - self.lim_loss[0])/self.lim_loss[0])
            # if diff > 0.1: print('warning: mismatch of lim values %f != %f' % (self.lim_loss_num[0], self.lim_loss[0]))
            # assert diff < 0.1, '%f, mismatch of lim values' % diff

            # this WARNING works when only 1 frame is the dominant one!!

            # self.dV = np.min(self.lim_loss) - self.min_loss
            self.dV = np.min(self.lim_loss_num) - self.min_loss

            if if_scan:
                n_lambdas = 100
                self.scan_lambdas = np.linspace(lambda_min, lambda_max, n_lambdas)

                results = compute(self.scan_lambdas, p0, g, gexp, sigma_exp, alpha)
                self.scan_results = results
                self.scan_loss_min = np.min(results['lossf'])
                self.scan_lambda_min = self.scan_lambdas[np.argmin(results['lossf'])]
                self.scan_avg_min = results['av_g'][np.argmin(results['lossf'])]

        # elif g.shape[0] == 2:
        else:

            perimeter = build_perimeter(self.min_lambda, delta_lambda, n_perim)
            
            losses = []

            for i in range(perimeter.shape[1]):
                losses.append(loss_fun(perimeter[:, i], p0, g, gexp, sigma_exp, alpha))

            self.perim_loss = losses
            """ Values of the loss function along the discretized perimeter. """

            wh = np.argmin(losses)            
            self.lim_loss_num = np.min(losses)
            """ Min. value of the loss function along the discretized perimeter. """

            res = loss_fun(perimeter[:, wh], p0, g, gexp, sigma_exp, alpha, True)
            self.lim_chi2 = res[1]
            """ Value of the chi2 in correspondence of the min. loss along the perimeter. """
            
            self.lim_dkl = res[2]
            """ Value of the Kullback-Leibler divergence in crorrespondence of the min. loss
            along the perimeter. """
            
            self.lim_p = res[3]
            """ Reweighted distribution in correspondence of the min. loss along the perimeter. """
            
            self.lim_g = g[:, np.where(self.lim_p == np.max(self.lim_p))[0][0]]
            """ Values of the observables """

            self.dV = self.lim_loss_num - self.min_loss
            """ Difference between min. loss value along the perimeter and global min. of the loss. """

            if if_scan:  # TO BE FIXED!!
                n_lambda = 50
                lambdas = [np.linspace(self.min_lambda[0] - delta_lambda, self.min_lambda[0] + delta_lambda, n_lambda),
                    np.linspace(self.min_lambda[1] - delta_lambda, self.min_lambda[1] + delta_lambda, n_lambda)]

                # my_grid = np.meshgrid(lambdas, lambdas)

                out = {}

                for l1 in lambdas[0]:
                    out[l1] = {}
                    for l2 in lambdas[1]:
                        out[l1][l2] = vars(compute_single(np.array([l1, l2]), p0, g, gexp, sigma_exp, alpha))['lossf']

                self.results = out

class analytical_props(Result):  # old name: compute_depth_analytical
    def __init__(self, n_frames, sigma, gexp, sigma_exp, alpha):
        """
        Compute main analytical properties of the posterior distribution, proportional to `np.exp(-loss_fun)`.
        The reference probability distribution P0 is assumed to be Gaussian centered in zero and with
        variance-covariance matrix assumed to be diagonal with `C_ii = sigma_i**2`.

        Parameters
        ----------
        n_frames : int
            The total n. of frames.
        
        sigma : array_like
            The standard deviations of the Gaussian distribution for the values of the observables;
            the variance-covariance matrix is diagonal (namely, zero covariance).

        gexp : float or array_like
            The experimental values for the observables.

        sigma_exp : float or array_like
            The estimated experimental uncertainty (`float` variable for a single observable,
            1d. `numpy.ndarray` otherwise).
        
        alpha : float
            The value for the hyperparameter.

        Return
        ------
        min_lambda, min_avg, min_dkl, min_loss : array_like
            The `numpy.ndarray` variables for the properties of the loss function in its point of min:
            the point of min. itself `min_lambda`, the average values of the observables `min_avg`,
            the Kullback-Leibler divergence `min_dkl`, the value of the loss f. `min_loss`.
        
        gbar : float
            The estimated value for the furthest point from the center of `n_frames` points sampled from
            the Gaussian distrib. defined by `sigma` as above (zero mean, variance-covariance matrix diagonal).
        
        gbar_1d : float
            Defined if there is just a single observable, it is an estimate (given by the inverse erf)
            of the furthest point from the center of `n_frames` points sampled from the 1d Gaussian distrib.
            with zero mean and standard deviation `sigma`.

        lim_chi2 : float
            The estimated value for the asymptotic limit of the chi2, computed from `gbar`
            (or from `gbar_1d` if present).

        lim_dkl : float
            The estimated value for the asymptotic limit of the Kullback-Leibler divergence,
            given by `np.log(n_frames)`.
        
        lim_loss : float
            The estimated value for the asymptotic limit of the loss function,
            given by `1/2*lim_chi2 + alpha*lim_dkl`.
        
        dV : float
            The estimated value for the height of the barrier (potential energy difference),
            computed as `lim_loss - min_loss`.
        """
        super().__init__()

        self.min_lambda = -gexp/(alpha*sigma_exp**2 + sigma**2)
        self.min_avg = -self.min_lambda*sigma**2
        self.min_dkl = 1/2*np.sum((sigma*self.min_lambda)**2)
        self.min_loss = 1/2*np.sum(((self.min_avg - gexp)/sigma_exp)**2) + alpha*self.min_dkl

        self.gbar = np.sqrt(2*np.log(n_frames)*np.sum(sigma**2))
        self.lim_chi2 = ((self.gbar - gexp)/sigma_exp)**2

        if (type(sigma) is float) or (type(sigma) is np.ndarray and len(sigma.shape) == 1 and sigma.shape[0] == 1):
            self.gbar_1d = sigma*my_inv_erf(1 - 2/n_frames)
            self.lim_chi2 = ((self.gbar_1d - gexp)/sigma_exp)**2

        self.lim_dkl = np.log(n_frames)
        self.lim_loss = 1/2*self.lim_chi2 + alpha*self.lim_dkl
        self.dV = self.lim_loss - self.min_loss

class distances_nd():

    """compute basic quantities for 2d Gaussian distribution"""

    def __init__(self, p0, g, gexp, sigma_exp, alpha):

        self.ds0 = np.sqrt(np.max(np.sum(g**2, axis=0)))
        """the distance from the origin of the farthest point"""

        self.ds = np.linalg.norm(g - gexp[:, None], axis=0)
        """the distances between each point and the experimental one"""

        self.min_ds = np.min(self.ds)
        """the min. distance point - experimental"""

        wh1 = np.where(self.ds == np.min(self.ds))[0][0]
        self.bar_g = g[:, wh1]
        """the value of the observables at the closest point from the experimental one"""

        if gexp.shape == 2:
            self.ratio = self.bar_g[0]/self.bar_g[1]
            """the ratio g_y/g_x of `bar_g`"""

        self.lambdas = -self.bar_g*1e3
        """the lambdas opposite to `bar_g` and far from the origin"""

        res = loss_fun(self.lambdas, p0, g, gexp, sigma_exp, alpha, True)

        self.chi2 = res[1]
        """the chi2 for the distribution at `lambdas`"""

        wh2 = np.where(res[-1] == np.max(res[-1]))[0][0]

        self.ds2 = np.linalg.norm(g[:, wh2] - gexp)
        """the distance between the selected frame and gexp"""

def my_group_fun(x, tolerance, if_diff = True, threshold = None, value = None):
    """
    Given a numpy 1d array `x` and two float variables `value` and `tolerance`, find where `x` gets stuck
    into `value +/- tolerance` (if not `if_diff`). Namely, return `my_group`, containing the position of
    the first frame where `x` is stuck there and `how_many` with the n. of consecutive frames `x` stays there.
    
    If `if_diff`, then consider `np.ediff1d(x)` rather than `x` and find where it is lower than a threshold
    `tolerance` (neglect `value`).

    Parameters
    ----------
    x : numpy.ndarray
        The numpy 1d. array whereof we compute the blocks of consecutive close values.
    
    tolerance : float
        The value for the tolerance of being inside a region (either for `x` or `np.ediff1d(x)`,
        depending on `if_diff`).

    if_diff : bool
        Boolean, if True than consider the difference, otherwise `x` itself to be in the desired region.
    
    threshold : optional, int
        If not None, it is the integer min. n. of consecutive frames for the blocks.
    
    value : optional, float
        If not None, it is the central value of the region we are interested in,
        defined by `value +- tolerance` (if not `if_diff`).

    Return
    ------
    whs_first : numpy.ndarray
        Numpy 1d. array with the first element of each block of consecutive frames.
    
    whs_len : numpy.ndarray
        Numpy 1d. array with the length of each block of consecutive frames.
    
    whs : list
        List of blocks of consecutive frames.
    
    dif : optional, numpy.ndarray
        If `if_diff`, return the Numpy 1d. array with the absolute values of `np.ediff1d(x)`.
    """

    assert (value is None) or (not if_diff), 'error: choose among value or if_diff'

    if not if_diff:
        wh = np.where(np.abs(x - value) < tolerance)[0]
        # equivalent to:
        # wh = np.where(x > value - tolerance)
        # wh2 = np.where(x < value + tolerance)
        # wh = np.intersect1d(wh, wh2)
    else:
        dif = np.abs(np.ediff1d(x))
        wh = np.where(dif < tolerance)[0]

    whs = []  # wh grouped by consecutive values

    whs.append([wh[0]])

    for i in range(len(wh) - 1):
        if (wh[i + 1] != wh[i] + 1): whs.append([wh[i + 1]])
        else: whs[-1].append(wh[i + 1])
    
    whs_first = [it[0] for it in whs]  # first element of each item in whs (first frame in each consecutive block)
    whs_len = [len(it) for it in whs]  # how many elements in each consecutive block

    if threshold is not None:
        wh2 = np.argwhere(np.array(whs_len) > threshold)[:, 0]
        whs = [whs[i] for i in wh2]
        whs_first = np.array(whs_first)[wh2]
        whs_len = np.array(whs_len)[wh2]

    if not if_diff:
        return whs_first, whs_len, whs
    else:
        return whs_first, whs_len, whs, dif

class Group_points(Result):
    def __init__(self, x, tolerance, if_diff = True, threshold = None, value = None):
        super().__init__()

        out = my_group_fun(x, tolerance, if_diff=if_diff, threshold=threshold, value=value)
        
        self.whs_first = out[0]
        self.whs_len = out[1]
        self.whs = out[2]
        if if_diff: self.dif = out[3]

        self.whs_flat = [s2 for s in self.whs for s2 in s]

    # def flatten_whs(whs):
    #     return [s2 for s in whs for s2 in s]

