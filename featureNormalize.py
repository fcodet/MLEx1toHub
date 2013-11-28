import numpy as np

def featureNormalize(X):
    dimX = np.shape(X)
    m = dimX[0]
    n = dimX[1]-1
    mu = np.zeros((n, 1))
    sigma = np.zeros((n, 1))
    for i in range(0, n):
        mu[i,0] = X[:, i+1].mean()
        sigma[i,0] = X[:, i+1].std()
        X[:, i+1] = (X[:, i+1] - mu[i, 0] * (np.mat(np.ones((m, 1)))))
        X[:, i+1] = X[:, i+1] / sigma[i, 0]
    return (X, mu, sigma)



