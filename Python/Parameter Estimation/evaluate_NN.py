import numpy as np
import pandas as pd 

class TestNN: 
    def __init__(self, NN_model, dynamical_model, data_model, n_y=6, y_include=[0,1,2,3,4,5]): 
        self.NN_model = NN_model 
        self.dyn_model = dynamical_model 
        self.data_model = data_model 
        self.n_y = n_y
        self.y_include = np.array(y_include)

    def find_theta_one(self): 
        u = self.dyn_model.find_input_vals() 
        theta_rand = np.array([np.random.uniform(th[0], th[1]) for th in self.dyn_model.theta_range])
        x0 = np.array([np.random.uniform(x0[0], x0[1]) for x0 in self.dyn_model.x0_range])
        x_sol, dxdt_sol, u = self.dyn_model.solve(x0, u, theta_rand) 
        X_NN = np.append(x_sol, dxdt_sol, axis=1)
        X_NN = np.append(X_NN, u, axis=1)
        y_NN = np.array([np.ones(len(X_NN))*th for th in theta_rand]).T 

        if len(self.y_include) < self.n_y:
            n = np.linspace(0, self.n_y-1, self.n_y, dtype=np.int8)
            n_x = np.delete(n, self.y_include)
            X_NN = np.append(X_NN.T, y_NN.T[n_x], axis=0).T
            y_NN = y_NN[:,self.y_include]


        X_NN = self.data_model.X_scaler.transform(X_NN)
        y_pred = self.NN_model.predict(X_NN)
        y_pred = self.data_model.y_scaler.inverse_transform(y_pred)
        return y_NN, y_pred 

    def find_theta_N(self, N_sim): 
        y_conf, _ = self.find_theta_one()
        y_full = np.zeros(shape=(N_sim,)+y_conf.shape)
        for i in range(len(y_full)):
            y_real, y_pred = self.find_theta_one() 
            y_f_scaled = y_pred/y_real
            y_full[i] = y_f_scaled

        theta_data = []
        for el in y_full.T: #t1 is (n_sim, n_steps, n_theta)
            theta_avg_temp = []
            theta_std_temp = []
            for i, el2 in enumerate(el): 
                theta_avg_temp.append(np.average(el2))
                theta_std_temp.append(np.std(el2))
            theta_data.append([theta_avg_temp, theta_std_temp])
        theta_data = np.array(theta_data)
        return y_full, theta_data


