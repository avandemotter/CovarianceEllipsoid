# CovarianceEllipsoid
```python
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
```
```matlab
Cov = [1 0.5 0.3
       0.5 2 0
       0.3 0 3];
mu = [1 2 3]';

[U,L] = eig(Cov);
% L: eigenvalue diagonal matrix
% U: eigen vector matrix, each column is an eigenvector

% For N standard deviations spread of data, the radii of the eliipsoid will
% be given by N*SQRT(eigenvalues).

N = 1; % choose your own N
radii = N*sqrt(diag(L));

% generate data for "unrotated" ellipsoid
[xc,yc,zc] = ellipsoid(0,0,0,radii(1),radii(2),radii(3));

% rotate data with orientation matrix U and center mu
a = kron(U(:,1),xc); 
b = kron(U(:,2),yc); 
c = kron(U(:,3),zc);

data = a+b+c; n = size(data,2);

x = data(1:n,:);
y = data(n+1:2*n,:); 
z = data(2*n+1:end,:);

% now plot the rotated ellipse
% sc = surf(x,y,z); shading interp; colormap copper
h = surfl(x, y, z); colormap copper
title('actual ellipsoid represented by mu and Cov')
axis equal
alpha(0.7)
```
