
from computeCostMulti import *
from AlgebraFunctions import *
from PlottingFunctions import *
import numpy as np
def gradientdescentMulti(X,y,theta,alpha, num_iters):
    dimy = np.shape(y)
    m = dimy[0]
    J_history = zerovector(num_iters)

    for iters in range(0, num_iters):
        A = X*theta - y
        for i in range(0,len(theta)):
            theta[i,0] = theta[i,0] - alpha / m * (X[:,i].transpose()*A)
        J_history[iters] = computeCostMulti(X, y, theta)
        #print J_history[iters]
    Plot(range(0, num_iters), J_history)
    return [theta ,J_history]



