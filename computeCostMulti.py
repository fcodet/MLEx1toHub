import numpy as np
def computeCostMulti(X, y, theta):
    m = len(y)
    J = 0
    A = X*theta - y
    J = J + np.float(A.transpose() * A)
    J = J/(2*m)
    return J
