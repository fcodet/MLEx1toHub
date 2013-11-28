
from computeCost import *
from AlgebraFunctions import *
from PlottingFunctions import *
import numpy as np
def gradientdescent(X,y,theta,alpha, num_iters):
	m = len(y)
	J_history = zerovector(num_iters)

	for iter in range(0, num_iters):
		##temp0 = 0
		##temp1 = 0
		##for i in range(0,m):
			##temp0 = temp0 + (np.float(( X[i,]*theta) - y[i])) * X[i,0]
			##temp1 = temp1 + (np.float(( X[i,]*theta) - y[i])) * X[i,1]
		##theta[0,0] = theta[0,0] - alpha / m * temp0
		##theta[1,0] = theta[1,0] - alpha / m * temp1
		A = X*theta - y
		theta[0,0] = theta[0,0] - alpha / m * (X[:,0].transpose()*A)
		theta[1,0] = theta[1,0] - alpha / m * (X[:,1].transpose()*A)
		J_history[iter] = computeCost(X, y, theta)
		print J_history[iter]
	Plot(range(0,num_iters),J_history)
	return theta



