import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow import keras 
from math import ceil
from kerastuner.tuners import Hyperband
import time
import os

class dynamical_model: 
    def __init__(self, dxdt, f_x):
        #Both dxdt and f_x is lambda functions. 
        self.f_x = lambda x, u, th: np.array(f_x(x,u,th))
        self.dxdt = lambda x, u, th: np.array(dxdt(x,u,th))
    
    def setup(self, t0, t1, N_steps, skip_el, x0_range, u_range, theta_range): 
        self.t0 = t0
        self.t1 = t1 
        self.N_steps = N_steps
        self.dt = (t1-t0)/N_steps
        self.skip_el = skip_el
        self.x0_range = x0_range 
        self.u_range = u_range
        self.theta_range = theta_range
        self.N_x = len(self.x0_range)
        self.N_u = len(self.u_range)
        self.N_theta = len(self.theta_range)
    
    def __str__(self): 
        try:
            print("t0:{}, t1:{}, N:{}, dt:{}, skip_el:{}, N_x:{}, N_u:{}, N_th:{}".format(self.t0, self.t1, 
                self.N_steps, self.dt, self.skip_el, len(self.x0_range), len(self.u_range), len(self.theta_range)))
        except: 
            pass 

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
        u = np.zeros((len(self.u_range), self.N_steps))
        for i in range(len(u)): #Goes through one variable at a time. 
            for j in range(len(u[i])): 
                if j == 0: 
                    u[i,j] = np.random.uniform(self.u_range[i][0], self.u_range[i][1])
                else: 
                    ch = np.random.random() 
                    if change > ch: 
                        u[i,j] = np.random.uniform(self.u_range[i][0], self.u_range[i][1])
                    else: 
                        u[i,j] = u[i,j-1]
        return np.array(u).T             
        
    def solve(self, x0, u, theta): 
        x_sol, dxdt_sol = self.RK4_solver(x0, u, theta)
        return x_sol[::self.skip_el], dxdt_sol[::self.skip_el]

    def generator(self, batch_size=32): #I have decided to let x0 have values close to each other, with diff of +-2
        self.N_steps = 2
        self.t0 = 0 
        self.t1 = 0.02
        self.dt = (self.t1 - self.t0)/self.N_steps 
        while True:
            X_NN = np.zeros((batch_size, 6))
            y_NN = np.zeros((batch_size, 6))
            for j in range(batch_size):
                x0 = np.random.uniform(self.x0_range[0][0], self.x0_range[0][1]) 
                x0 = np.array([x0, x0+np.random.uniform(-2, 2)])
                u = self.find_input_vals() 
                theta = [] 
                for th in self.theta_range: 
                    theta.append(np.random.uniform(th[0], th[1]))
                x_sol, dxdt_sol = self.solve(x0, u, theta)
                X_NN[j][0:2] = x_sol[-1] 
                X_NN[j][2:4] = dxdt_sol[-1]
                X_NN[j][4:6] = u[-1] 
                y_NN[j] = np.array(theta)
                #X_NN[j][6:] = np.array(theta[0:2])
                #y_NN[j] =np.array(theta[2:])
            yield X_NN, y_NN
                

    def plot_solve(self, x0, u, theta, names_x, names_u): 
        x_sol, dx_sol = self.solve(x0, u, theta)
        t = np.linspace(self.t0, self.t1, self.N_steps)
        fig, ax = plt.subplots(3,1)
        ax[0].set_title("States")
        [ax[0].plot(t, x, label=names_x[i]) for i, x in enumerate(x_sol.T)]
        ax[1].set_title("Derivatives of states")
        [ax[1].plot(t, dx, label=names_x[i]+"'") for i, dx in enumerate(dx_sol.T)]
        ax[2].set_title("Inputs")
        [ax[2].plot(t, u, label=names_u[i]) for i, u in enumerate(u.T)]
        for axx in ax: 
            axx.legend()
            axx.grid()
        
        fig.tight_layout(h_pad=-0.2)
        plt.show()

class generate_data: 
    def __init__(self, model): 
        self.model = model

    def solve_N(self, N_data_points, filename_X, filename_y, change=0.005): #todo: Make a plotting function and multiprocessing
        N_sim = ceil(N_data_points/(self.model.N_steps/self.model.skip_el))
        X_NN = np.zeros((N_sim*ceil(self.model.N_steps/self.model.skip_el), 2*self.model.N_x+self.model.N_u)) #NN 8 inputs: [x1, x1, dx1, dx2, u1, u2, Cm, Cc]
        y_NN = np.zeros((N_sim*ceil(self.model.N_steps/self.model.skip_el), self.model.N_theta)) #NN 6 outputs (all parameters also considering Cm and Cc)
        for i in range(N_sim): 
            x0 = np.array([np.random.uniform(x_init[0], x_init[1]) for x_init in self.model.x0_range])
            u_in = self.model.find_input_vals(change=change)
            theta = np.array([np.random.uniform(th[0], th[1]) for th in self.model.theta_range])

            x_sol, dxdt_sol = self.model.solve(x0, u_in, theta)

            #Storing values in arrays
            # x_sol = x_sol[::self.model.skip_el]
            # dxdt_sol = dxdt_sol[::self.model.skip_el]
            u_in = u_in[::self.model.skip_el]
            j = 0
            #print(x_sol)
            #print("len(x) = {}, len(dxdt) = {}, len(u) = {}".format(len(x_sol), len(dxdt_sol), len(u_in)))
            for x, dx, u in zip(x_sol, dxdt_sol, u_in): 
                #print("i = {}, j = {}, while len(x) = {}. product = {}.".format(i, j, len(x_sol), len(x_sol)*i))
                #First include x's
                k = 0
                for x_el in x: 
                    X_NN[(i*len(x_sol)+j),k] = x_el
                    k += 1
                #Then derivatives:
                for dx_el in dx: 
                    X_NN[i*len(x_sol)+j][k] = dx_el
                    k += 1
                
                #Finally inputs
                for u_el in u: 
                    X_NN[i*len(x_sol)+j][k] = u_el
                    k += 1

                #Then adding NN solutions 
                for k, th in enumerate(theta): 
                    y_NN[i*len(x_sol)+j][k] = th

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

    def generator(self, X_scaler_data, y_scaler_data, batch_size=32): 
        while True:
            X_scaler = MinMaxScaler() 
            y_scaler = MinMaxScaler() 
            X_scaler = X_scaler.fit(X_scaler_data)
            y_scaler = y_scaler.fit(y_scaler_data) 
            gen_data = self.model.generator(batch_size=batch_size) 
            for i in range(batch_size):
                X_NN, y_NN = next(gen_data)
                X_NN = X_scaler.transform(X_NN)
                y_NN = y_scaler.transform(X_NN)
                yield X_NN, y_NN
        
class handle_NNs: 
    def __init__(self, N_inputs, N_outputs, node_limit=(10, 400), layer_limit=(1,4), epochs=5): 
        self.N_inputs = N_inputs 
        self.N_outputs = N_outputs 

    def plot_opt_NNs(self, N_plots):
        plots = np.linspace(0, len(self.res)-1, N_plots, dtype=np.int16)
        res_opt = self.sort_results(print_res=False)
        fig, ax = plt.subplots(1,1)
        for i, res in enumerate(res_opt): 
            if i in plots: 
                ax.plot(res_opt[res], label=res)

        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.legend() 
        plt.show()

    def build_model(self, hp): #Builds cone-shaped sequential models
        N1 = hp.Int('n_starting_nodes', min_value=32, max_value=2048, step=128)
        N_layers = hp.Int('n_layers', min_value=1, max_value=10, step=3)
        #Find number of nodes in all layers: 
        a = (self.N_outputs-N1)/N_layers
        N_nodes = [int(N1 + a*i) for i in range(N_layers)]    

        model = keras.Sequential()
        model.add(keras.Input(shape=(self.N_inputs,), name="Input_Layer")) 
        for i, node in enumerate(N_nodes): 
            model.add(keras.layers.Dense(node, activation='relu', name=f"Layer_{i+1}"))
            model.add(keras.layers.Dropout(0.1, name=f"Dropout_{i+1}"))

        model.add(keras.layers.Dense(self.N_outputs, name="Output_Layer"))
        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate',values=[1e-3, 1e-4])),
                      loss='mse', metrics=['accuracy'])
        return model

    def find_opt_hyperparams(self, train_gen, test_gen, max_epochs=5, project_name="Param_est_Case_1"):
        tuner = Hyperband(self.build_model,
                          objective='val_loss',
                          max_epochs=max_epochs,
                          #executions_per_trial=3,
                          project_name=project_name, 
                          directory=os.path.normpath('C:/'))

        tuner.search_space_summary()
        
        t1 = time.time() 
        tuner.search(X_train, y_train, validation_data=(X_test, y_test), callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)], verbose=2)
        print("Tiden for denne kommandoen var: {} min".format(round((time.time()-t1)/60,2)))
        best_model = tuner.get_best_models(1)[0]
        best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
        best_model.summary()
        print(best_hyperparameters)
        best_model.save(os.path.abspath("Trained Models\model_1"))
        return best_model 

    def train_model(self, model, X_train, X_test, y_train, y_test, max_epochs=5, model_name="fully_trained_model1"):
        pass 


if __name__ == "__main__": 
    chp_air = 1000 
    Q_HE = lambda x, u, theta: theta[5]*chp_air*(1-np.exp(-theta[4]/(theta[5]*chp_air)))*(x[1]-u[1])
    dTm_dt = lambda x, u, theta: (u[0]-theta[2]*(x[0]-x[1])-theta[3]*(x[0]-u[1]))/theta[0]
    dTc_dt = lambda x, u, theta: (theta[2]*(x[0]-x[1])-Q_HE(x,u,theta))/theta[1]

    dxdt = lambda x,u,theta: np.array([dTm_dt(x,u,theta), dTc_dt(x,u,theta)])
    f_x = lambda x,u,theta: np.array([Q_HE(x,u,theta)])

    x0_range = [(20, 60), (20, 60)]
    u_range = [(0, 30), (20, 30)]
    theta_range = [(100, 800), (40, 400), (5,20), (0.1,1), (1,5), (0.3, 1.2)]

    model = dynamical_model(dxdt, f_x)
    model.setup(0, 60, 600, 20, x0_range, u_range, theta_range)

    data_model = generate_data(model) 
    X_NN, y_NN = data_model.load_data(os.path.abspath("Training_data\X_data.npy"), os.path.abspath("Training_data\y_data.npy")) #could also generate data 
    #X_NN, y_NN = data_model.solve_N(2e6, "X_data.npy", "y_data.npy")

    #Move Cm and Cc from y to X: 
    X_NN = np.append(X_NN.T, y_NN.T[:2], axis=0).T
    y_NN = y_NN[:,2:]
    X_train, X_test, y_train, y_test, X_NN_scaled, y_NN_scaled = data_model.prepare_data(X_NN[0:500], y_NN[0:500])

    handler = handle_NNs(len(X_train[0,:]), len(y_train[0,:]))
    handler.find_opt_hyperparams2(X_train, X_test, y_train, y_test, project_name="Param_est_case1_test1")

    

