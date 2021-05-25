"""Mixture model for collaborative filtering"""
from typing import NamedTuple, Tuple
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
import numpy as np
import scipy


class GaussianMixture(NamedTuple):
    """
    Tuple holding a gaussian mixture model
    """
    # (K, d) array - each row corresponds to a gaussian component mean
    mean: np.ndarray

    # (K, d, d) array - each element corresponds to the covariance matrix of a component
    covariance: np.ndarray

    # (1, K) array - each element corresponds to the weight of a component
    proportion: np.ndarray


def generate_positive_semidefinite(size: Tuple[int, int, int], mean: float = 0.0, std: float = 0.9):
    """
    Generate a positive semi-definite matrix to be used as a covariance matrix

    :param size: dimensions of the output positive semidefinite matrix
    :param mean: mean of the cluster
    :param range: range of the values in the matrix
    :return: a matrix of size=size
    """
    K = size[0]

    M = np.random.normal(mean, std, size=size)

    # After matrix generation, guarantee that each of them is positive semidefinite
    for j in range(K):
        M[j] = np.matmul(M[j], M[j].T)

    return M


def init(X: np.ndarray, K: int, seed: int = 0) -> Tuple[GaussianMixture, np.ndarray]:
    """Initializes the mixture model with random points as initial
    means and uniform assignments

    :param X: (n, d) array holding the data
    :param K: number of components
    :param seed: random seed
    :return:
        mixture: the initialized gaussian mixture
        init_post: (n, K) array holding the initial posterior probabilities for all components for all examples
    """

    np.random.seed(seed)

    n, d = X.shape
    # initialize all pi's with the same value
    proportion = np.full((1, K), 1.0 / K)

    # means array with Kxd elements
    mean = np.random.uniform(-10, 10, size=(K, d))

    # select K random points as initial means
    # mu = X[np.random.choice(n, K, replace=False)]

    # covariance matrices array with K dxd positive semidefinite matrices
    covariance = generate_positive_semidefinite(size=(K, d, d))

    # Initialize a GaussianMixture object
    gaussian_mixture = GaussianMixture(mean, covariance, proportion)

    # initialize the posterior matrix
    init_post = np.ones((n, K)) / K

    return gaussian_mixture, init_post


def plot_1D(X: np.ndarray, gaussian_mixture: GaussianMixture, responsibilities: np.ndarray, log_likelihood: float,
            filename: str, iteration: int, xaxis_ll: list):
    """
    Plot the 1D Gaussian Mixture Model along with the data points and the expected log-likelihood

    :param X: (n, d) array holding the data
    :param gaussian_mixture: GaussianMixture object containing the gaussian mixture model
    :param responsibilities: responsibilities matrix for each data point for all clusters
    :param log_likelihood: array of expected log-likelihood
    :param filename: filename with the filepath to save the output image file
    :param iteration: number of current iteration
    :param xaxis_ll: list of x-axis point for plotting
    :return: Save the output image of the GMM and the log-likelihood. Returns None
    """

    x_values = np.arange(-12, 12, 0.1).reshape(1, 240)
    gmm = np.zeros((240, ))

    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    legend = ["Component 1", "Component 2", "Component 3", "GMM"]

    # GMM Plot
    for j, col in enumerate(colors):
        y_values = scipy.stats.norm(gaussian_mixture.mean[j][0],
                                    gaussian_mixture.covariance[j][0])
        if j == 0:
            y = y_values.pdf(x_values) * gaussian_mixture.proportion[0][j]
        else:
            y = y + y_values.pdf(x_values) * gaussian_mixture.proportion[0][j]

        axs[0].plot(x_values.reshape(240, ), y_values.pdf(x_values).reshape(240, ) * gaussian_mixture.proportion[0][j],
                 "--", c=colors[j], linewidth=1)

        axs[0].set_ylim([-0.005, 0.22])
        axs[0].set_xlim([-12, 12])
        axs[0].set_xlabel("$x$")
        axs[0].set_ylabel("$p(x)$")

        gmm += y_values.pdf(x_values).reshape(240, ) * gaussian_mixture.proportion[0][j]

    # plot the gmm
    axs[0].plot(x_values.reshape(240, ), gmm, "-", color="#000000", linewidth=0.7)
    axs[0].plot(x_values, y, c="#000000")
    axs[0].set_title("GMM with 3 components\niteration=" + str(iteration), fontsize=10)

    for i in range(len(X)):
        axs[0].scatter(X[i], 0, color=responsibilities[i])
    axs[0].legend(legend)

    # log-likelihood plot
    if iteration == 0:
        axs[1].scatter(xaxis_ll, log_likelihood, s=16)
    else:
        axs[1].plot(xaxis_ll, log_likelihood)
        axs[1].scatter(xaxis_ll, log_likelihood, s=16)

    axs[1].set_ylim([-6000, -1000])
    axs[1].set_xlim([-0.05, 42.05])
    axs[1].set_title("Expected log-likelihood\niteration=" + str(iteration), fontsize=10)
    axs[1].set_xlabel("# Iteration")
    axs[1].set_ylabel("Expected log-likelihood")

    fig.tight_layout()
    fig.savefig(filename)
    fig.clf()  # clear image buffer


def plot(X: np.ndarray, gaussian_mixture: GaussianMixture, responsibilities: np.ndarray, log_likelihood: float,
         title: str):
    """
    Plots the mixture model for 2D data

    :param X: (n, d) array holding the data
    :param gaussian_mixture: GaussianMixture object containing the gaussian mixture model
    :param responsibilities: responsibilities matrix for each data point for all clusters
    :param log_likelihood: array of expected log-likelihood
    :param title: title of the plot
    :return: None
    """
    _, K = responsibilities.shape

    fig, ax = plt.subplots()
    ax.title.set_text(title)
    ax.set_xlim((-10, 10))
    ax.set_ylim((-10, 10))

    for i, point in enumerate(X):
        ax.scatter(X[i][0], X[i][1], color=responsibilities[i], alpha=0.5, linewidths=0)

    x, y = np.meshgrid(np.linspace(-10, 10, 100), np.linspace(-10, 10, 100))
    pos = np.dstack((x, y))

    for j in range(K):
        mean = gaussian_mixture.mean[j]
        covariance = gaussian_mixture.covariance[j]
        normal = multivariate_normal(mean, covariance)
        # circle = Circle(mean, covariance, color=color[j], fill=False)
        ax.contour(x, y, normal.pdf(pos), alpha=1.0, zorder=10)
        # legend = "mu = ({:0.2f}, {:0.2f})\n stdv = {:0.2f}".format(
        # mean[0], mean[1], covariance)
        # ax.text(mean[0], mean[1], legend)

    plt.axis('equal')
    plt.show()


def gen_sample(mean: list, std: list, seed: float, N: int = 600, K: int = 3, dims: int = 2):
    """
    Generate random samples of data points

    :param mean: mean array of the data points
    :param std: standard deviation of each cluster
    :param seed: random seed
    :param N: number of data points to be generated
    :param K: number of centers of the data points
    :param dims: data points dimensions
    :return: (n, dims) sample array
    """

    samples = np.empty((N, dims))

    np.random.seed(seed)
    for k in range(K):
        n = int(N / K)
        samples[n * k:n * (k + 1)] = np.random.normal(loc=mean[k], scale=std[k] ** 2, size=(n, dims))

    return samples


def covariance_clip_1D(covariance: np.ndarray, clip: float, limit: float):
    """
    Apply regularization to the variances of 1D data

    :param covariance: covariance matrix, in this case, a array of variances
    :param clip: clip threshold
    :param limit: limit assigned to the variances above the clip threshold
    :return: None
    """
    covariance[covariance > clip] = limit


def covariance_clip(covariance: np.ndarray, clip: float, rate: float):
    """
    Apply regularization to the variances of D-dimensional data

    :param covariance: covariance matrix
    :param clip: clip threshold
    :param rate: 1 - rate of penalization applied to the covariance matrix
    :return: None
    """
    if np.sum(covariance > clip) >= 1:
        covariance *= rate


def rmse(X: np.ndarray, Y: np.ndarray):
    """
    Compute the root-mean squared error of the predictions against the data

    :param X: (n, d) array holding the data
    :param Y: (n, d) array holding the predictions
    :return: Root-Mean Squared Error
    """
    return np.sqrt(np.mean((X - Y) ** 2))


def bic(X: np.ndarray, gaussian_mixture: GaussianMixture,
        log_likelihood: float) -> float:
    """
    Computes the Bayesian Information Criterion for a gaussian mixture model

    :param X: (n, d) array holding the data
    :param mixture: a mixture of gaussians
    :param log_likelihood: the log-likelihood of the data

    :return: the BIC for this mixture
    """

    n, d = X.shape
    # A formula for calculation of free parameters can be found at
    # http://www.ijetch.org/papers/144-L080.pdf
    # Accordingly to this reference the original formula is
    # p = (len(mixture.p) - 1) + len(mixture.mu) * d + len(mixture.var) * d * (d - 1) / 2
    # But since we constrain the covariance matrix to be and identity matrix, what we got is
    p = (len(gaussian_mixture.p) - 1) + len(gaussian_mixture.mean) * d + len(gaussian_mixture.var)

    return log_likelihood - 1 / 2 * p * np.log(n)
