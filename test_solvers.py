import matplotlib.pyplot as plt
import torch
import nn_solver_PINN_approach as nspa
import numpy as np


class CompareFEM():
    def __init__(self, pts,q_FEM, q_NN):
        """
        pts - the points on which we compare the solutions of the neural network solver and the FEM solver
        q_FEM - the value of the committor at each point in pts as computed by the FEM
        q_NN - the value of the committor at each point in pts as computed by the neural network solver
        """
        self.pts = pts
        self.q_FEM = q_FEM
        self.q_NN = q_NN
    
    def plot_difference(self):
        fig, ax = plt.subplots()
        c = self.q_FEM - self.q_NN
        s = ax.scatter(self.pts[:,0], self.pts[:,1], c = c, s= 1)
        fig.colorbar(s)
        return fig, ax

class NNCommittorApprox():
    def __init__(self, model, pts) -> None:
        self.model = model
        self.pts = pts
        q = self.get_committor()
        self.q = q.detach().numpy().squeeze()
    
    def get_committor(self):
        return self.model(self.pts)

    def plot_q(self):
        fig, ax = plt.subplots()
        c = self.q
        s = ax.scatter(self.pts[:,0], self.pts[:,1], c= c, s= 1)
        fig.colorbar(s)
        return fig, ax
    
class PINNModel():
    def __init__(self, model_weights_filepath, layer_array) -> None:
        # layer array should be compatible with the model weights
        self.model = nspa.NeuralNetwork(layer_array=layer_array)
        self.model.load_state_dict(torch.load(model_weights_filepath))

class HybridModel():
    def __init__(self, model_weights_filepath, layer_array) -> None:
        # layer array should be compatible with the model weights
        self.model = nspa.NeuralNetwork(layer_array=layer_array)
        self.model.load_state_dict(torch.load(model_weights_filepath))

if __name__=="__main__":
    # hybrid = HybridModel("filepath", [2,20,1])
    pinn = PINNModel("./model_weights/maier-stein-model-weights-dateime-2022-7-26-11-45-alpha-10.0-layer_array-[2, 20, 1]-epochs-100", [2,20,1])
    
    # get points where we will compare the FEM and NN approach
    pts = np.loadtxt('MaierStein_pts.csv', delimiter=',', dtype=float)

    # work around the error of inputting numpy data into torch nn
    pts_torch = torch.tensor(pts).float()
    
    # get the pinn approximation at the pts
    pinn_approx = NNCommittorApprox(pinn.model, pts_torch)
    q_pinn = pinn_approx.q

    # plot pinn approx: 
    pinn_approx.plot_q()

    # FEM approximation:
    q_FEM = np.loadtxt('MaierStein_q_FEM.csv', delimiter=',', dtype=float)

    # comparison for pinn
    compare = CompareFEM(pts, q_FEM, q_pinn)
    compare.plot_difference()

    # comparison for FEM

    plt.show()


