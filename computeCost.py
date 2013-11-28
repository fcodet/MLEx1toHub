import numpy as np
def computeCost(X, y, theta):
	m = len(y)
	J = 0
	#for i in range(0,m):
		#J = J + (np.float(( X[i,]*theta) - y[i]))*(np.float(( X[i,]*theta) - y[i]))
		#A = theta.transpose()*X.transpose() - y.transpose()
	A = X*theta - y
	J = J + np.float(A.transpose() * A)
	J = J/(2*m)
	return J



