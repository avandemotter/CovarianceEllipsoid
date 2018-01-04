# CovarianceEllipsoid
'''python
import numpy as np
from numpy import cos, sin, pi
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

phi = np.linspace(0, 2 * pi)
theta = np.linspace(-pi / 2, pi / 2)
phi, theta = np.meshgrid(phi, theta)

# cov: covariance matrix
# mu: mean data
cov = [[1.0, 0.5, 0.3],
       [0.5, 2.0, 0.0],
       [0.3, 0.0, 3.0]]

mu = [1, 2, 3]

# w: eigenvalue diagonal matrix
# v: eigenvector matrix, each row is an eigenvector
w, v = LA.eig(cov)

# For M standard deviations spread of data, the radii of the ellipsoid while be given by M*SQRT(eigenvalues)
M = 1
w = M*np.sqrt(w)

# generate data for "unrotated" ellipsoid
x = cos(theta) * sin(phi) * w[0]
y = cos(theta) * cos(phi) * w[1]
z = sin(theta) * w[2]

# rotate data with orientation matrix v
a = np.kron(np.reshape(v[0], (3, 1)), x)
b = np.kron(np.reshape(v[1], (3, 1)), y)
c = np.kron(np.reshape(v[2], (3, 1)), z)

data = a + b + c
n = np.size(data, 1)

# translate with center mu
x = data[0:n, :] + mu[0]
y = data[n:2*n, :] + mu[1]
z = data[2*n:3*n, :] + mu[2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(x, y, z)

plt.show()
'''
