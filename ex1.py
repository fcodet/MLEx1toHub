print("Running warmup exercise...")
print("5x5 Identity matrix")
from warmUPExercise import *
from FileOperations import *
from PlottingFunctions import *
from AlgebraFunctions import *
from computeCost import *
from gradientDescent import *
warmUPExercise()
data = loadcsv('ex1data1.txt')
vX = data[:, 0]
vy = data[:, 1]
y = np.mat(vy).transpose()
m = len(y)
Plot(vX,vy)
#X = np.mat(np.array([onevector(m), vX]))
X = np.mat([onevector(m), vX])
X = X.transpose()
#X is a the form:
# [ 1 x(1) ]
# [ 1 x(2) ]
# [ 1 x(3) ]
theta = np.mat(np.array([zerovector(2)]))
theta = theta.transpose()
#Theta is of the form:
#[ theta0 ]
#[ theta1 ]

iterations = 800
alpha = 0.01

J = computeCost(X, y , theta)
print("Initial cost J(0):")
print J

theta = gradientdescent(X, y, theta, alpha, iterations)
print("Calculated Theta:")
print theta

predict1 = [[1 , 3.5]] * theta
print('For population = 35,000, we predict a profit of:')
print predict1*10000
predict2 = [[1 , 7]] * theta
print('For population = 70,000, we predict a profit of:')
print predict2*10000

theta1_vals = np.linspace(-10,10,25)
theta2_vals = np.linspace(-1,4,25)

J_Vals = np.zeros((len(theta1_vals),len(theta2_vals)))

for i in range(0, len(theta1_vals)):
	for j in range(0, len(theta2_vals)):
		t = np.array([[theta1_vals[i]], [theta2_vals[j]]])
		J_Vals[i,j]=computeCost(X,y,t)


SurfacePlot(theta1_vals,theta2_vals,J_Vals)



