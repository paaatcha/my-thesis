# -*- coding: utf-8 -*-
"""

Autor: André Pacheco
Email: pacheco.comp@gmail.com

This files implements the functions to estimate the dirichlet distribution using the maximum likelihood estimation
algorithm  according to following paper:
Estimating a Dirichlet distribution. Thomas P. Minka
Link: https://tminka.github.io/papers/dirichlet/minka-dirichlet.pdf

Most of this code was inspired by the one developed by Eric Suh:
Link: https://github.com/ericsuh/dirichlet

If you find any bug, please e-mail me.
"""

import numpy as np
import scipy.special as sp
from time import time

def _log_likelihood(data, alpha):
    """
    This function computes the Dirichlet log likelihood according to equation 4 in the paper.
    :param data: it is a 2D array with one sample per row. For example, if p is 1 x K vector and you have N samples,
                 data must be N x K
    :param alpha: an array representing the dirichlet parameters. It must be K-dimensional to match with data
    :return: a float representing the dirichlet log likelihood - log(data|alpha)
    """

    # Checking if data and alpha are in the same dimension
    num_samples, dim_data = data.shape
    dim_alpha = len(alpha)
    if dim_data != dim_alpha:
        print ("data and alpha must have the same dimension. Right now data is {}-d and alpha is {}-d".format(
               dim_data, dim_alpha))

    avg_log_data = np.log(data).mean(axis=0)
    L1 = sp.gammaln(alpha.sum()) - sp.gammaln(alpha).sum() + ((alpha-1)*avg_log_data).sum()
    return num_samples * L1


def _digamma_inv(alpha, tol=1.48e-9, max_iter=10):
    """
    This function computes the inverse of digamma function according to the Newton's method, described in appendice C
    of the paper.
    :param alpha: an array representing the dirichlet parameters
    :param tol: the tolerance to stop the iteration
    :param maxiter: the max iterations
    :return: the digamma inverse with the same shape of the inputs
    """

    alpha = np.asarray(alpha, dtype=np.float)
    x0 = np.piecewise(alpha, [alpha >= -2.22, alpha < -2.22],
                      [(lambda x: np.exp(x) + 0.5), (lambda x: -1/(x+np.euler_gamma))])

    for i in range(max_iter):
        x1 = x0 - (sp.psi(x0) - alpha)/sp.polygamma(1, x0)
        if np.linalg.norm(x1 - x0) < tol:
            break
        x0 = x1

    return x1


def _init_guess_alpha(data, ep=1e-6):
    """
    This function computes a initial guess for alpha based on data
    :param data: it is a 2D array with one sample per row. For example, if p is 1 x K vector and you have N samples,
                 data must be N x K
    :return: the initial guess for alpha
    """
    E = data.mean(axis=0)
    E2 = (data**2).mean(axis=0)

    return ((E[0] - E2[0])/(ep+E2[0]-E[0]**2)) * E


def estimate(data, tol=1e-8, max_iter=2000, ep=1e-8, report=False):
    """
    This function computes the Dirichlet maximum likelihood estimation according to the fixed point algorithm described
    in the paper.
    :param data: it is a 2D array with one sample per row. For example, if p is 1 x K vector and you have N samples,
                 data must be N x K
    :param tol: the tolerance to stop the iteration
    :param max_iter: the max iteration
    :param report: set it True to get the time a and iterations taken converge
    :return: a K-d array with the dirichlet parameters    """


    if report:
        init_time = time()

    # This is to numerical stability
    data = data + ep

    avg_log_data = np.log(data).mean(axis=0)
    alpha_0 = _init_guess_alpha(data)

    for i in range(max_iter):
        alpha_1 = _digamma_inv(sp.psi(alpha_0.sum()) + avg_log_data)
        if abs(_log_likelihood(data, alpha_1) - _log_likelihood(data, alpha_0)) < tol:
            break
        alpha_0 = alpha_1

    if report:
        end_time = time()
        total_time = end_time - init_time
        return alpha_1, total_time, i
    else:
        return alpha_1


def cov_matrix (alphas):
    """
    This function computes the Dirichlet covariance matrix according to the parameters alpha
    :param alphas: an array representing the Dirichlet parameters
    :return: the covariance matrix
    """
    n = len(alphas)
    cov = np.zeros((n,n))
    a0 = sum(alphas)
    for i in range(n):
        for j in range(n):
            if i == j:
                cov[i,j] = (alphas[i]*(a0 - alphas[i])) / ( (a0**2) * (a0 + 1) )
            else:
                cov[i,j] = -(alphas[i] * alphas[j]) / ((a0**2) * (a0 + 1))
    return cov


def test_convergence(dim_alpha=(3,100,1), n_samples=(100,10000,100), max_iter=1000, tol=1e-8):
    from tqdm import tqdm

    if not isinstance(dim_alpha, tuple) and not isinstance(dim_alpha, list):
        dim_alpha = (dim_alpha, dim_alpha, 1)

    if not isinstance(n_samples, tuple) and not isinstance(n_samples, list):
        n_samples = (n_samples, n_samples, 1)

    all_times, all_dim, all_samples, all_norms = list(), list(), list(), list()
    # with tqdm(total=int(1+(dim_alpha[1]/dim_alpha[2])-dim_alpha[0]), ascii=True, ncols=100) as t:
    with tqdm(total=5, ascii=True, ncols=100) as t:
        for _ in range(5):
            for dim in range(dim_alpha[0], dim_alpha[1]+1, dim_alpha[2]):
                all_dim.append(dim)
                for N in range(n_samples[0], n_samples[1]+1, n_samples[2]):
                    all_samples.append(N)
                    alphas = np.random.uniform(0, 100, dim)
                    data = np.random.dirichlet(alphas, N)
                    al, total_time, ite = estimate(data, tol=tol, max_iter=max_iter, report=True)
                    all_times.append(total_time)
                    all_norms.append(np.linalg.norm(al - alphas) / np.linalg.norm(al))
            t.update()

    return all_times, all_dim, all_samples, all_norms


