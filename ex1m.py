from warmUPExercise import *
from FileOperations import *
from PlottingFunctions import *
from AlgebraFunctions import *
from computeCost import *
from gradientDescent import *
from featureNormalize import *
from computeCostMulti import *
from gradientdescentMulti import *

print('loading data...')
data = loadcsv('ex1data2.txt')
print('data loaded.')
data = np.mat(data)
vX = np.concatenate((data[:, 0],data[:, 1]), 1)
vy = data[:, 2]
m = len(vy)
X = np.concatenate((np.ones((m, 1)), vX), 1)
#X = np.mat(np.array([np.ones((m , 1)), np.mat(vX )] ))
y = np.mat(vy)

print ('Normalising Features')

fN_result = featureNormalize(X)
mu = fN_result[1]
sigma = fN_result[2]


alpha = 0.01
num_iters = 400
theta = np.mat(np.ones((3, 1)))
gd_result = gradientdescentMulti(X, y, theta, alpha, num_iters)
theta = gd_result[0]
J1 = gd_result[1]
print('theta with alpha=0.01')
print(theta)
x1 = (1650.0 - mu[0]) / sigma[0]
x2 = (3.0 - mu[1]) / sigma[1]
prediction = theta[0,0] + theta[1,0]*x1 + theta[2,0]*x2
print('prediction  with alpha=0.01:')
print prediction

alpha = 0.1
num_iters = 400
theta = np.mat(np.ones((3, 1)))
gd_result = gradientdescentMulti(X, y, theta, alpha, num_iters)
theta = gd_result[0]
J2 = gd_result[1]
print('theta with alpha=0.1:')
print(theta)
x1 = (1650.0 - mu[0]) / sigma[0]
x2 = (3.0 - mu[1]) / sigma[1]
prediction = theta[0,0] + theta[1,0]*x1 + theta[2,0]*x2
print('prediction  with alpha=0.1:')
print prediction

alpha = 1.00
num_iters = 400
theta = np.mat(np.ones((3, 1)))
gd_result = gradientdescentMulti(X, y, theta, alpha, num_iters)
theta = gd_result[0]
J3 = gd_result[1]
print('theta with alpha=0.1:')
print(theta)
x1 = (1650.0 - mu[0]) / sigma[0]
x2 = (3.0 - mu[1]) / sigma[1]
prediction = theta[0,0] + theta[1,0]*x1 + theta[2,0]*x2
print('prediction  with alpha=1.0:')
print prediction

MultiPlot(range(0,num_iters), [J1,J2,J3])





