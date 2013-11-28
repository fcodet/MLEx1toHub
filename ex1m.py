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

alpha = 0.01
num_iters = 400

theta = np.mat(np.ones((3, 1)))

theta = gradientdescentMulti(X, y, theta, alpha, num_iters)












