import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow import keras 
from math import ceil

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
        
class handle_NNs: 
    def __init__(self, N_inputs, N_outputs, node_limit=(10, 400), layer_limit=(1,4), N_models=4*20, epochs=5): 
        self.nodes = np.arange(node_limit[0], node_limit[1], 1)
        self.layer_limit = np.arange(layer_limit[0], layer_limit[1], 1)
        self.N_inputs = N_inputs 
        self.N_outputs = N_outputs 

    def make_NN(self, N_layers, N_nodes, dropout=0.1): 
        model = keras.Sequential()
        model.add(keras.Input(shape=(self.N_inputs,), name="Input_Layer")) 
        for i in range(N_layers): 
            model.add(keras.layers.Dense(N_nodes, activation='relu', name=f"Layer_{i}"))
            model.add(keras.layers.Dropout(dropout, name=f"Dropout_{i}"))
        model.add(keras.layers.Dense(self.N_outputs, name="Output_Layer"))
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model

    def make_range_NN(self, N_nodes=(10, 100), N_layers=(1, 4), N_net=40):
        #Rounds N_tries to fit even tries in each layers 
        dN_layers = N_layers[1]-N_layers[0] + 1 #The number of layers that is tried 
        dN_nodes = round(N_net/dN_layers) #Number of node attempts 
        NN_layers = np.linspace(N_layers[0], N_layers[1], dN_layers, dtype=np.int16) 
        NN_nodes = np.linspace(N_nodes[0], N_nodes[1], dN_nodes, dtype=np.int16) 

        NNs = []
        NN_lay_nod = []
        for layer in NN_layers: 
            for node in NN_nodes: 
                NNs.append(self.make_NN(layer, node)) 
                NN_lay_nod.append([layer, node])
        return NNs, NN_lay_nod

    def find_opt_hyperparams(self, X_data, y_data, N_nodes=(10, 100), N_layers=(1, 4), N_net=40, epochs=5): 
        res = {} #Will contain {1: [loss, acc]}
        NNs, NN_lay_nod = self.make_range_NN(N_nodes=N_nodes, N_layers=N_layers,N_net=N_net)
        for i, NN in enumerate(NNs): 
            print(f"\rWorking on NN {i+1} of {len(NNs)}")
            history = NN.fit(X_data, y_data, epochs=epochs, verbose=0)
            history = history.history 
            res[i] = [history['loss'], history['accuracy'], NN_lay_nod[i]]
        self.res = res
        return res 

    def sort_results(self, print_res=True):  
        res_sorted = {} 
        nodes_sorted = []
        layers_sorted = []
        least_loss = 0
        for i in range(len(self.res)): #Sorts the NNs
            least_loss_temp = 1e9
            for j in range(len(self.res)): #Finds the best/worst NNs 
                loss_NN = self.res[j][0][-1]
                if loss_NN < least_loss_temp and loss_NN > least_loss: 
                    j_least = j 
                    least_loss_temp = loss_NN 
            res_sorted[f"NN_{j_least}"] = self.res[j_least][0]
            layers_sorted.append(self.res[j_least][2][0])
            nodes_sorted.append(self.res[j_least][2][1])
            least_loss = least_loss_temp
        if print_res:
            print("The rankings of the Neural Networks are :")
            for i, res in enumerate(res_sorted): 
                print(f"{res} with a loss of {res_sorted[res][-1]}. Nodes = {nodes_sorted[i]}, layers = {layers_sorted[i]}")
        return res_sorted


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
        N_layers = hp.Int('n_layers', min_value=1, max_value=10, step=2)
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


    

