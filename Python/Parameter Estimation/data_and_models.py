import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow import keras 

class dynamical_model: 
    def __init__(self, dxdt, f_x, n_params):
        #Both dxdt and f_x is lambda functions. 
        self.theta = np.zeros(n_params)
        self.f_x = lambda x, u, th: np.array([f for f in f_x])
        self.dxdt = lambda x, u, th: np.array([dx for dx in dxdt])
    
    def setup(self, t0, t1, N_steps, skip_el, x0_range, u_range, theta_range): 
        self.t0 = t0
        self.t1 = t1 
        self.N_steps = N_steps
        self.dt = (t1-t0)/N_steps
        self.skip_el = skip_el
        self.x0_range = x0_range 
        self.u_range = u_range
        self.theta_range = theta_range 

    def RK4_solver(self, x0, u, theta):
        x_solved = np.zeros((self.N_steps+1, len(x0))) #Premakes the solution arrays for faster execution
        dx_solved = np.zeros((self.N_steps, len(x0))) #Premakes the solution arrays for faster execution
        x_solved[0] = np.array(x0)
        for i in range(self.N_steps): 
            K1 = self.dxdt(x_solved[i], u[i], theta) #Make sure f returns an array of shape (2,)
            K2 = self.dxdt(x_solved[i]+K1/2, u[i], theta)
            K3 = self.dxdt(x_solved[i]+K2/2, u[i], theta)
            K4 = self.dxdt(x_solved[i]+K3, u[i], theta)
            K = K1/6 + K2/3 + K3/3 + K4/6
            x_solved[i+1] = x_solved[i] + K*self.dt
            dx_solved[i] = K
        return x_solved[:-1], dx_solved
    
    def find_input_vals(self, change=0.01):
        #u_range = [(u1_min, u1_max), (u2_min, u2_max)]
        u = np.array([np.zeros(self.N_steps) for _ in self.u_range])
        u_init = np.array([np.random.uniform(u_r[0], u_r[1]) for u_r in self.u_range])
        u = [[u] for u in u_init]
        for _ in range(0, self.N_steps): 
            for u_el, u_r in zip(u, self.u_range): 
                ch = np.random.uniform(0, 1)/len(u)
                if ch < change: 
                    u_el.append(np.random.uniform(u_r[0], u_r[1]))
                    #u2_rand = np.random.uniform(u_range[1][0], u_range[1][1])
        return np.array(u).T             
        
    def solve(self, x0, u, theta): 
        x_sol, dxdt_sol = self.RK4_solver(x0, u, theta)
        return x_sol[::self.skip_el], dxdt_sol[::self.skip_el]


class generate_data: 
    def __init__(self, model): 
        self.model = model

    def solve_N(self, N_data_points, filename_X, filename_y): #todo: Make a plotting function and multiprocessing
        N_sim = int(N_data_points/(self.model.N_steps/self.model.skip_el))
        X_NN = np.zeros((N_sim*int(self.model.N_steps/self.model.skip_el), len(self.model.dxdt))) #NN 8 inputs: [x1, x1, dx1, dx2, u1, u2, Cm, Cc]
        y_NN = np.zeros((N_sim*int(self.model.N_steps/self.model.skip_el), len(self.model.theta))) #NN 6 outputs (all parameters also considering Cm and Cc)
        for i in range(N_sim): 
            x0 = np.array([np.random.uniform(x_init[0], x_init[1]) for x_init in self.model.x0_range])
            u_in = self.model.find_input_vals(self.model.u_range)
            theta = np.array([np.random.uniform(th[0], th[1]) for th in self.model.theta_range])

            x_sol, dxdt_sol = self.model.solve(x0, u, theta)

            #Storing values in arrays
            x_sol = x_sol[::self.model.skip_el]
            dxdt_sol = dxdt_sol[::self.model.skip_el]
            u_in = u_in[::self.model.skip_el]
            j = 0
            for x, dx, u in zip(x_sol, dxdt_sol, u_in): 
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
        np.save(filename_X, X_NN)
        np.save(filename_y, y_NN)
        return X_NN, y_NN

    def load_data(self, filename_X, filename_y): 
        X_NN = np.load(filename_X)
        y_NN = np.load(filename_y)
        return X_NN, y_NN

    def prepare_data(self, X_NN, y_NN): 
        X_scaler = MinMaxScaler() 
        y_scaler = MinMaxScaler() 
        self.X_scaler = X_scaler.fit(X_NN)
        self.y_scaler = y_scaler.fit(y_NN) 

        X_NN_scaled = self.X_scaler.transform(X_NN)
        y_NN_scaled = self.y_scaler.transform(y_NN)

        X_train, X_test, y_train, y_test = train_test_split(X_NN_scaled, y_NN_scaled, test_size=0.1)
        return X_train, X_test, y_train, y_test, X_NN_scaled, y_NN_scaled
        

class create_NN: 
    def __init__(self, node_limit=(10, 400), layer_limit=(1,4), N_models=4*20, epochs=5): 
        self.nodes = np.arange(node_limit[0], node_limit[1], 1)
        self.layer_limit = np.arange(layer_limit[0], layer_limit[1], 1)

    def make_NN(self, N_layers, N_nodes): 
        model = keras.Sequential()
        model.add(keras.Input(shape=(8,) )) #Stopped here
        for i in range(N_layers): 
        model.add(keras.layers.Dense(N_nodes, activation='relu') )
