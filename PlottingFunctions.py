from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import matplotlib.pyplot as plt

def Plot(x,y):
	ax = plt.axes()
	ax.scatter(x,y)
	plt.show()

def SurfacePlot(x,y,z):
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	x, y = np.meshgrid(x, y)
	ax.plot_surface(x,y,z,rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	plt.show()
