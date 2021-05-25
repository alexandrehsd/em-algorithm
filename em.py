"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from utils import GaussianMixture
import utils


def compute_soft_probabilities(X: np.ndarray, gaussian_mixture: GaussianMixture) -> np.ndarray:
    """
    Compute the soft probabilities for every data point belonging to each cluster j = 1, ..., K.

    :param X: (n, d) array holding the data
    :param gaussian_mixture: gaussian mixture model
    :return: np.ndarray (n, K) with the soft probabilities (posterior probabilities)
        of every cluster over each data point.
    """
    n, d = X.shape

    K = gaussian_mixture.proportion.shape[1]

    # (x - mu)
    center_distance = X[:, np.newaxis] - gaussian_mixture.mean

    if d > 1:
        # Sigma^(-1)
        inv_covariance = np.linalg.inv(gaussian_mixture.covariance)

        # det(Sigma^(-1))
        determinant = np.linalg.det(inv_covariance)
    else:
        inv_covariance = 1 / gaussian_mixture.covariance
        determinant = inv_covariance

    # (x - mu)^T * Sigma
    temp1 = np.matmul(center_distance[:, :, np.newaxis], inv_covariance)

    # ((x - mu)^T * Sigma) * (x - mu)
    temp2 = np.matmul(temp1, center_distance.reshape((n, K, d, 1)))

    # -1/2 * (x - mu)^T * Sigma * (x - mu)
    exponent = -0.5 * temp2.reshape((n, K))

    # 1 / ((2*pi)^(1/2) * det(Sigma^(-1)))
    norm_constant = 1.0 / ((np.sqrt(2 * np.pi) ** d) * np.sqrt(determinant) + 1e-16)
    norm_constant = norm_constant.reshape((1, K))

    # compute responsibilities
    soft_probabilities = gaussian_mixture.proportion * norm_constant * np.exp(exponent)

    return soft_probabilities


def compute_log_likelihood(soft_probabilities: np.ndarray, responsibilities: np.ndarray) -> np.ndarray:
    """
    Compute the expected log-likelihood over all the data points given the parameter set and the responsibilities

    :param soft_probabilities: the posterior probability of each point belonging to class j = 1, ... , k
    :param responsibilities: responsibilities matrix
    :return: the log-likelihood over all the data points given the parameter set and the posterior probabilities
    """
    return np.sum(responsibilities * np.log((soft_probabilities + 1e-16) / (responsibilities + 1e-16)))


def estep(X: np.ndarray, gaussian_mixture: GaussianMixture) -> Tuple[np.ndarray, np.ndarray]:
    """
    E-step: Softly assigns each datapoint to a gaussian component

    :param X: (n, d) array holding the data
    :param gaussian_mixture: the current gaussian mixture
    :return:
        responsibilities: (n, K) array holding the soft counts for all components for all examples
        expected_log_likelihood: expected log-likelihood of the assignment
    """

    soft_probabilities = compute_soft_probabilities(X, gaussian_mixture)
    normalize_constant = np.sum(soft_probabilities, axis=1)
    responsibilities = soft_probabilities / normalize_constant[:, np.newaxis]

    expected_log_likelihood = compute_log_likelihood(soft_probabilities, responsibilities)

    return responsibilities, expected_log_likelihood


def compute_means(X: np.ndarray, responsibilities: np.ndarray, total_resp: np.ndarray) -> np.ndarray:
    """
    Update the means in the M Step

    :param X: (n, d) array holding the data
    :param responsibilities: responsibilities matrix
    :param total_resp: Total responsibility of each cluster
    :return: updated means
    """
    mean = np.matmul(responsibilities.T, X) / total_resp[:, np.newaxis]
    return mean


def compute_mixture_weights(total_resp: np.ndarray, n: float, K: int) -> np.ndarray:
    """
    Update the mixture weights in the M Step

    :param total_resp: Total responsibility array
    :param n: number of data points
    :param K: number of mixture components
    :return: updated mixture weights
    """
    return (total_resp / n).reshape((1, K))


def compute_variances(X: np.ndarray, mean: np.ndarray, responsibilities: np.ndarray,
                      total_resp: np.ndarray) -> np.ndarray:
    """
    Update the covariance matrix in the M Step

    :param X: (n, d) array holding the data
    :param mean: updated means
    :param responsibilities: responsibilities matrix
    :param total_resp: Total responsibility of each cluster
    :return: updated covariance matrix, or updated variances for 1D data
    """
    n, d = X.shape
    K = mean.shape[0]

    center_distance = X[:, np.newaxis] - mean
    if d > 1:
        center_distance_prod = np.matmul(center_distance.reshape((n, K, d, 1)),
                                         center_distance.reshape((n, K, 1, d)))
    else:
        center_distance_prod = (center_distance ** 2).reshape((n, K, d, 1))

    responsibilities = responsibilities.reshape((n, K, 1, 1))
    covariance = np.add.reduce(responsibilities * center_distance_prod) / total_resp[:, np.newaxis, np.newaxis]
    return covariance


def mstep(X: np.ndarray, responsibilities: np.ndarray) -> GaussianMixture:
    """
    M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    :param X: (n, d) array holding the data
    :param responsibilities: (n, K) array holding the responsibilities for all components for all examples
    :return GaussianMixture: the new gaussian mixture
    """
    n, K = responsibilities.shape

    total_resp = np.sum(responsibilities, axis=0)

    mean = compute_means(X, responsibilities, total_resp)
    proportion = compute_mixture_weights(total_resp, n, K)

    covariance = compute_variances(X, mean, responsibilities, total_resp)

    return GaussianMixture(mean, covariance, proportion)


def run(X: np.ndarray, gaussian_mixture: GaussianMixture,
        plot_results: bool = False) -> Tuple[GaussianMixture, np.ndarray, float]:
    """
    Runs the EM algorithm
    
    :param X: (n, d) array holding the data
    :param gaussian_mixture: GaussianMixture initialized object
    :param plot_results: boolean, plot results or not
    :return: 
        gaussian_mixture: the new gaussian mixture
        responsibilities: (n, K) array holding the soft counts for all components for all examples
        LL_new: log-likelihood of the current assignment
    """
    _, d = X.shape

    LL_old = None
    LL_new = None

    plot_results =  d == 1 and plot_results
    if plot_results:
        iter = 0
        artifacts = []
        lls = []

    while LL_old is None or LL_new - LL_old > 1E-6 * np.abs(LL_new):
        LL_old = LL_new

        # E-Step
        responsibilities, LL_new = estep(X, gaussian_mixture)

        # M-Step
        gaussian_mixture = mstep(X, responsibilities)

        if d > 1:
            # custom regularization applied to avoid a mixture component domination
            utils.covariance_clip(gaussian_mixture.covariance, clip=3, rate=0.8)
        else:
            utils.covariance_clip_1D(gaussian_mixture.covariance, clip=2, limit=2)

        # Plot results
        if plot_results:
            iter += 1
            lls.append(LL_new)
            artifacts.append([responsibilities, gaussian_mixture])

    if plot_results:
        xaxis = list(range(iter))
        for iteration in range(iter):
            resp = artifacts[iteration][0]
            gmm = artifacts[iteration][1]

            if d > 1:
                # Plot curves - used for animations
                utils.plot(X, gmm, resp, lls[0:iteration + 1], title="title")
            else:
                utils.plot_1D(X, gmm, resp, lls[0:iteration + 1],
                              filename='./images/onedim0' + str(iteration), iteration=iteration,
                              xaxis_ll=xaxis[0:iteration + 1])

    return gaussian_mixture, responsibilities, LL_new
