import matplotlib.pyplot as plt
from scipy import *
from scipy import integrate
from scipy.integrate import ode
import numpy as np

fig = plt.figure(num=1)
ax=fig.add_subplot(111)

## Vector field function
chp_air = 1000
def vf(t,x):
  theta = [100, 100, 10, 10, 10, 1]
  u = [50, 10]
  Q_HE = lambda x, u, theta: theta[5]*chp_air*(1-np.exp(-theta[4]/(theta[5]*chp_air)))*(x[1]-u[1])
  dTm_dt = (u[0]-theta[2]*(x[0]-x[1])-theta[3]*(x[0]-u[1]))/theta[0]
  dTc_dt = (theta[2]*(x[0]-x[1])-Q_HE(x,u,theta))/theta[1]

  dx=[]
  dx.append(-x[1])
  dx.append(x[0])
  return np.array([dTm_dt, dTc_dt])

#Solution curves
t0=0; tEnd=10; dt=0.01;
r = ode(vf).set_integrator('vode', method='bdf',max_step=dt)
ic=[[10,10], [20,20], [30,40]]
color=['r','b','g']
for k in range(len(ic)):
    Y=[];T=[];S=[];
    r.set_initial_value(ic[k], t0).set_f_params()
    while r.successful() and r.t +dt < tEnd:
        r.integrate(r.t+dt)
        Y.append(r.y)

    S=np.array(np.real(Y))
    ax.plot(S[:,0],S[:,1], color = color[k], lw = 1.25)

#Vector field
X,Y = np.meshgrid( np.linspace(0,50,20),np.linspace(0,50,20) )
U, V = vf(0, [X,Y])
#U = -Y
#V = X
#Normalize arrows
N = np.sqrt(U**2+V**2)  
U2, V2 = U/N, V/N
ax.quiver( X,Y,U2, V2)


plt.xlim([0, 50])
plt.ylim([0, 50])
plt.xlabel(r"$x$")
plt.ylabel(r"$y$")
plt.show()