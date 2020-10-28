import matplotlib.pyplot as plt
from scipy import *
from scipy import integrate
from scipy.integrate import ode
import numpy as np

## Vector field function
def vf(x):
  dx = x[0]**2 - x[0]
  dx = np.array([dx, 0])
  return dx

#Vector field
x = np.linspace(-1, 2, 50)
y = np.linspace(-2, 4, 10)
X, Y = np.meshgrid(x, y)
U, V  = vf([X, Y])

#Normalize arrows
N = np.sqrt(U**2+V**2)  
U2, V2 = U/N, V/N

fig = plt.figure(num=1)
ax=fig.add_subplot(111)
ax.quiver(X, Y, U, V)
ax.plot(x, vf([x,None])[0])
ax.grid()

plt.xlim([-1,2])
plt.ylim([-2,4])
plt.xlabel(r"$x$")
plt.ylabel(r"$\frac{dx}{dt}$")
plt.show()