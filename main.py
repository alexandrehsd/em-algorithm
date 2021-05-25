import utils
import em


if __name__ == '__main__':
    data_seed = 1
    X = utils.gen_sample(mean=[-5, 1, 8], std=[1.1, 1.1, 1.2], seed=data_seed, N=600, K=3, dims=1)

    model_seed = 2
    k = 3  # Number of mixture components (clusters)
    gaussian_mixture, responsibilities = utils.init(X, K=k, seed=model_seed)
    gaussian_mixture, responsibilities, LL = em.run(X, gaussian_mixture, plot_results=True)