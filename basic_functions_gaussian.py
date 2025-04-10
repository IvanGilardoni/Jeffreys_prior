import numpy as np
import jax
import jax.numpy as jnp

from scipy.optimize import minimize
from scipy.special import erf, erfinv

def my_erf(x):
    return erf(x/np.sqrt(2))

def my_inv_erf(x):
    return np.sqrt(2)*erfinv(x)

from basic_functions_bayesian import compute, compute_single

#%%

def flatten(x, s, m = None):
    if m is None:
        if type(x) is dict: return [vars(x[n])[s] for n in x.keys()]
        elif type(x) is list:return [vars(x[n])[s] for n in range(len(x))]
    else:
        if type(x) is dict: return [vars(x[n])[s][m] for n in x.keys()]
        elif type(x) is list:return [vars(x[n])[s][m] for n in range(len(x))]

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

def build_perimeter(x0, dx, n_x = 100):

    x_min = x0 - dx
    x_max = x0 + dx

    scan = [np.linspace(x_min[0], x_max[0], n_x), np.linspace(x_min[1], x_max[1], n_x)]

    perimeter = [np.vstack((scan[0], x_min[1]*np.ones(n_x))), np.vstack((x_max[0]*np.ones(n_x), scan[1])),
        np.vstack((scan[0], x_max[1]*np.ones(n_x))), np.vstack((x_min[0]*np.ones(n_x), scan[1]))]

    perimeter = np.hstack(perimeter)

    return perimeter

class compute_depth():

    def __init__(self, n, sigma : float or np.ndarray, gexp, sigma_exp, alpha, delta_lambda = 10, n_perim = 100, if_scan = False):
        """sigma can also be g, in this way it is fixed (given as input), without randomness"""
    
        if (type(sigma) is float) or (type(sigma) is np.ndarray and len(sigma.shape) == 1 and sigma.shape[0] == 1):
            g = np.random.normal(0, sigma, n)
            g = np.array([g])  # to make the same for one or more observables
        
        elif sigma.shape[0] == 2:
            if len(sigma.shape) == 1: cov = np.diag(sigma**2)
            else: cov = sigma
            g = np.random.multivariate_normal(mean=(0, 0), cov=cov, size=n).T

        else:
            assert sigma.shape[1] == n
            g = sigma

        self.g = g

        p0 = np.ones(n)/n

        self.mini = minimize(loss_and_grad, jnp.zeros(g.shape[0]), args=(p0, g, gexp, sigma_exp, alpha, loss_grad_fun), jac=True, method='BFGS')

        self.min_lambda = self.mini.x
        self.min_loss = self.mini.fun

        out = loss_fun(self.min_lambda, p0, g, gexp, sigma_exp, alpha, if_return_all=True)

        self.min_avg = np.einsum('ij,j', g, out[3])

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

        elif g.shape[0] == 2:

            perimeter = build_perimeter(self.min_lambda, delta_lambda, n_perim)
            
            out = []

            for i in range(perimeter.shape[1]):
                out.append(loss_fun(perimeter[:, i], p0, g, gexp, sigma_exp, alpha, True))

            losses = np.array([out[i][0] for i in range(len(out))])
            wh = np.argwhere(losses == np.min(losses))[0][0]

            self.perim_losses = losses
            
            self.lim_loss_num = np.min(losses)
            self.lim_chi2 = out[wh][1]
            self.lim_dkl = out[wh][2]
            self.lim_p = out[wh][3]
            self.lim_g = g[:, np.where(self.lim_p == np.max(self.lim_p))[0][0]]

            self.dV = self.lim_loss_num - self.min_loss

            if if_scan:
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

class compute_depth_analytical():

    def __init__(self, n, sigma, gexp, sigma_exp, alpha):

        self.lambda_min = -gexp/(alpha*sigma_exp**2 + sigma**2)
        self.avg_min = -self.lambda_min*sigma**2
        # chi2 = 1/2*((avg_min - gexp)/sigma_exp)**2
        # self.loss_min = 1/2*(alpha*gexp**2)*(alpha + sigma**2)/(alpha*sigma_exp**2 + sigma**2)**2
        self.loss_min = 1/2*((self.avg_min - gexp)/sigma_exp)**2 + 1/2*alpha*(self.lambda_min*sigma)**2

        self.gbar = sigma*my_inv_erf(1 - 2/n)
        self.lim_chi2 = ((self.gbar - gexp)/sigma_exp)**2
        self.lim_dkl = np.log(n)
        self.lim_value = 1/2*self.lim_chi2 + alpha*self.lim_dkl
        self.dV = self.lim_value - self.loss_min

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

def my_group_fun(x, value, tolerance):
    """Given a numpy 1d array `x` and two float variables `value` and `tolerance`,
    find where `x` gets stuck into `value +/- tolerance`. Namely, return `my_group`, containing
    the position of the first frame where `x` is stuck there and `how_many` with the n. of consecutive
    frames `x` stays there."""

    wh = np.where(x > value - tolerance)
    wh2 = np.where(x < value + tolerance)
    wh = np.intersect1d(wh, wh2)
    
    my_group = []
    how_many = []

    my_group.append(wh[0])
    how_many.append(1)

    for i in range(len(wh) - 1):
        if (wh[i + 1] != wh[i] + 1):
            my_group.append(wh[i + 1])
            how_many.append(1)
        else:
            how_many[-1] += 1

    return my_group, how_many
