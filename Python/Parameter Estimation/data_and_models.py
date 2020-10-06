import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler

def RK4_solver(f, x0, N, dt, u, theta):
    x_solved = np.zeros((N+1, len(x0))) #Premakes the solution arrays for faster execution
    dx_solved = np.zeros((N, len(x0))) #Premakes the solution arrays for faster execution
    x_solved[0] = np.array(x0)
    for i in range(N): 
        K1 = f(x_solved[i], u[i], theta) #Make sure f returns an array of shape (2,)
        K2 = f(x_solved[i]+K1/2, u[i], theta)
        K3 = f(x_solved[i]+K2/2, u[i], theta)
        K4 = f(x_solved[i]+K3, u[i], theta)
        K = K1/6 + K2/3 + K3/3 + K4/6
        x_solved[i+1] = x_solved[i] + K*dt
        dx_solved[i] = K
    return x_solved[:-1], dx_solved

class dynamical_model: 
    def __init__(self, dxdt, f_x, n_params):
        #Both dxdt and f_x is lambda functions. 
        self.theta = np.zeros(n_params)
        self.f_x = lambda x, u, th: np.array([f for f in f_x])
        self.dxdt = lambda x, u, th: np.array([dx for dx in dxdt])
    
    def setup(self, t0, t1, N_steps, skip_el): 
        self.t0 = t0
        self.t1 = t1 
        self.N_steps = N_steps
        self.dt = (t1-t0)/N_steps
        self.skip_el = skip_el
    
    def find_input_vals(self, u_range, change=0.01):
        #u_range = [(u1_min, u1_max), (u2_min, u2_max)]
        u = np.array([np.zeros(self.N_steps) for _ in u_range])
        u_init = np.array([np.random.uniform(u_r[0], u_r[1]) for u_r in u_range])
        for i in range(0, N): 
            ch = np.random.uniform(0, 1)
            u1[i] = u1_rand
            u2[i] = u2_rand
            if ch < change: 
                u1_rand = np.random.uniform(u_range[0][0], u_range[0][1])
                #u2_rand = np.random.uniform(u_range[1][0], u_range[1][1])
            
    return np.array([u1, u2]).T
        
    def solve(self, x0, u, theta, skip_sol=20): 
        x_sol, dxdt_sol = RK4_solver(self.dxdt, x0, self.N_steps, self.dt, u, theta)
        return x_sol[::self.skip_el], dxdt_sol[::self.skip_el]

    def solve_N(self, N_data_points, x0_range, u_range, theta_range): 
        N_sim = int(N_data_points/(self.N_steps/self.skip_el))
        X_NN = np.zeros((N_sim*int(self.N_steps/self.skip_el), len(self.dxdt))) #NN 8 inputs: [x1, x1, dx1, dx2, u1, u2, Cm, Cc]
        y_NN = np.zeros((N_sim*int(self.N_steps/self.skip_el), len(self.theta))) #NN 6 outputs (all parameters also considering Cm and Cc)
        for i in range(N_sim): 
            x0 = np.array([np.random.uniform(x_init[0], x_init[1]) for x_init in x0_range])
            u = 
            theta = np.array([np.random.uniform(th[0], th[1]) for th in theta_range])


            #Storing values in arrays
            x_sol = x_sol[::jump_el]
            dx_sol = dx_sol[::jump_el]
            u_in = u_in[::jump_el]
            j = 0
            for x, dx, u in zip(x_sol, dx_sol, u_in): 
                X_NN[i*len(x_sol)-1+j][0] = x[0]
                X_NN[i*len(x_sol)-1+j][1] = x[1]
                X_NN[i*len(x_sol)-1+j][2] = dx[0]
                X_NN[i*len(x_sol)-1+j][3] = dx[1]
                X_NN[i*len(x_sol)-1+j][4] = u[0]
                X_NN[i*len(x_sol)-1+j][5] = u[1]
                X_NN[i*len(x_sol)-1+j][6] = theta[0]
                X_NN[i*len(x_sol)-1+j][7] = theta[1]
                
                y_NN[i*len(x_sol)-1+j][0] = theta[0]
                y_NN[i*len(x_sol)-1+j][1] = theta[1]
                y_NN[i*len(x_sol)-1+j][2] = theta[2]
                y_NN[i*len(x_sol)-1+j][3] = theta[3]
                y_NN[i*len(x_sol)-1+j][4] = theta[4]
                y_NN[i*len(x_sol)-1+j][5] = theta[5]
                j += 1

