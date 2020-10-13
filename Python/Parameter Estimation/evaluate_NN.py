import numpy as np
import pandas as pd 

class TestNN: 
    def __init__(self, NN_model, dynamical_model, data_model): 
        self.NN_model = NN_model 
        self.dyn_model = dynamical_model 
        self.data_model = data_model 

    def find_theta_one(self): 
        u = self.dyn_model.find_input_vals() 
        theta_rand = np.array([np.random.uniform(th[0], th[1]) for th in self.dyn_model.theta_range])
        x0 = np.array([np.random.uniform(x0[0], x0[1]) for x0 in self.dyn_model.x0_range])
        x_sol, dxdt_sol = self.dyn_model.solve(x0, u, theta_rand) 
        X_NN = np.append(x_sol, dxdt_sol, axis=1)
        X_NN = np.append(X_NN, u, axis=1)
        y_NN = np.array([np.ones(len(X_NN))*th for th in theta_rand]).T 