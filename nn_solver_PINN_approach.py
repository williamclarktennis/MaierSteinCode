import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import math

import matplotlib.pyplot as plt

import numpy as np

from datetime import datetime

import os

from tqdm import tqdm

in_size = 2
out_size = 1

class NeuralNetwork(nn.Module):
    """
    nn.Linear documentation: 
    (https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)

    Input: (*, H_{in}) where * means any number of 
    dimensions including none and H_{in} =in_features.

    Output: (*, H_{out}) where all but the last dimension 
    are the same shape as the input and H_{out} =out_features.
    """
    
    def __init__(self, layer_array):
        super().__init__()

        assert type(layer_array) == type([])

        self.linears = nn.ModuleList([nn.Linear(layer_array[i],layer_array[i+1]) for i in range(len(layer_array)-1)])

    def forward(self, X):

        assert X.shape[-1] == in_size

        # define tanh activation func
        tanh = nn.Tanh()
        # define sigmoid activation function
        sig = nn.Sigmoid()

        for i in range(len(self.linears)-1):
            X = self.linears[i](X)
            X = tanh(X)

        X = self.linears[-1](X)
        X = sig(X)

        return X

def get_Laplacian_partialx_partialy(q_NN: NeuralNetwork, X: torch.tensor) -> torch.tensor:
    """
    
    """
    assert X.shape[-1] == in_size

    # get outputs 
    q_NN_outputs = q_NN(X)

    # specify the vector in the vector Jacobian product: 
    vector_1 = torch.ones_like(q_NN_outputs)

    # compute vector Jacobian product
    jacobian_x_y = torch.autograd.grad(outputs= q_NN_outputs,\
        inputs= X,grad_outputs=vector_1,allow_unused=True,\
            retain_graph=True,create_graph=True)

    # get first column of vector Jacobian product, i.e: partial 
    # deriv of q_NN with respect to x
    partial_x = jacobian_x_y[0][:,0]

    # ditto:
    partial_y = jacobian_x_y[0][:,1]

    # specify next vector: 
    vector2 = torch.ones_like(partial_x)
    
    # compute jacobian of partial_x
    jacobian_xx_xy = torch.autograd.grad(outputs = partial_x, inputs = X,\
        grad_outputs= vector2, allow_unused=True, retain_graph=True)

    # compute jacobian of partial_y
    jacobian_yx_yy = torch.autograd.grad(outputs = partial_y, inputs = X,\
        grad_outputs= vector2, allow_unused=True, retain_graph=True)
    
    laplacian = jacobian_xx_xy[0][:,0] + jacobian_yx_yy[0][:,1]

    laplacian = laplacian[:,None]
    partial_x = partial_x[:,None]
    partial_y = partial_y[:,None]

    assert laplacian.shape[-1] == out_size
    assert partial_x.shape[-1] == out_size
    assert partial_y.shape[-1] == out_size

    return laplacian, partial_x, partial_y

def L_q(q_NN: NeuralNetwork, X:torch.tensor) -> torch.tensor:
    """
    Apply the Kolomogorov backwards operator to q_NN at X
    """

    assert X.shape[-1] == in_size

    x = X[:,0][:,None]
    y = X[:,1][:,None]

    assert x.shape[-1] == out_size
    assert y.shape[-1] == out_size

    laplacian_q, partial_x, partial_y = get_Laplacian_partialx_partialy(q_NN,X)

    lq = (x-x**3-10*x*y**2)* partial_x - (1+x**2)*y * partial_y + 0.1/2 * laplacian_q

    assert lq.shape[-1] == out_size

    return lq


"""
Acquire the training data:
"""
def get_mesh_of_sample_space(xl, xr, n, yl, yr, m):
    x = torch.linspace(xl,xr,n)
    y = torch.linspace(yl,yr,m)
    xx, yy = torch.meshgrid(x,y)
    xx_f = torch.flatten(xx)
    yy_f = torch.flatten(yy)
    return x,y,xx,yy,xx_f,yy_f

def get_points_on_A_B():
    """
    Recall that A is the circle with center (-1,0) and
    radius 0.3. 
    Thus, we can obtain a uniform sample of points from the 
    perimeter of this circle using trigonometry. 
    """
    pi = math.pi
    x = torch.linspace(0, 2*pi - (2*pi / 100),100)
    bA, bB = torch.zeros((100,2)), torch.zeros((100,2))
    bA[:,0], bA[:,1] = -1, 0
    bA[:,0] += 0.3* torch.cos(x)
    bA[:,1] += 0.3* torch.sin(x)

    assert bA.shape[-1] == in_size

    bB[:,0], bB[:,1] = 1, 0
    bB[:,0] += 0.3* torch.cos(x)
    bB[:,1] += 0.3* torch.sin(x)

    assert bB.shape[-1] == in_size

    ######## Visualize the points on A and B:
    # fig, (ax1, ax2) = plt.subplots(1,2)
    # ax1.scatter(bB[:,0],bB[:,1], s=1)
    # ax1.set_title("bB")
    # ax2.scatter(bA[:,0],bA[:,1], s=2)
    # ax2.set_title("bA")
    # plt.show()

    bA.requires_grad_(True)
    bB.requires_grad_(True)

    return bA, bB

def get_training_pts_not_from_A_B_but_uniform():
    """
    Make uniform grid of [-2,2]x[-0.5,0.5] with A and B removed
    """
    x = torch.linspace(-2,2,100)
    y = torch.linspace(-0.5,0.5,100)
    xx, yy = torch.meshgrid(x,y)
    xx = torch.flatten(xx)
    yy = torch.flatten(yy)
    temp = torch.zeros((100*100,2))
    temp[:,0] = xx
    temp[:,1] = yy

    output = torch.zeros((0,2))
    # remove points in A or B:
    for point in temp:
        assert point.shape == torch.Size([2])
        if not in_A(point) and not in_B(point):
            point = point.reshape([1,2])
            output = torch.cat((output,point))

    ###### VISUALIZE: #########
    # fig, ax = plt.subplots()
    # ax.scatter(output[:,0],output[:,1], s=1)
    # plt.show()

    assert output.shape[-1] == in_size

    output.requires_grad_(True)

    return output

def in_A(point) -> True or False:
    assert point.shape == torch.Size([2])
    x, y = point[0], point[1]
    return (x+1)**2 + y**2 <= 0.3**2

def in_B(point) -> True or False:
    assert point.shape == torch.Size([2])
    x,y = point[0],point[1]
    return (x-1)**2 + y**2 <= 0.3**2

class Training():
    """
    Training implements the PINN loss function model where
    the model to be trained will learn the boundary conditions. 
    """
    def __init__(self,NN,optimizer,loss_fn, epochs, alpha):
        self.NN = NN
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.alpha = alpha

        self.training_points_on_bdy_of_A, self.training_points_on_bdy_of_B = get_points_on_A_B()
        self.training_points_not_in_A_or_B = get_training_pts_not_from_A_B_but_uniform()

        labels_for_training_pts_not_in_A_or_B = torch.zeros((len(self.training_points_not_in_A_or_B),out_size))
        train_data = TensorDataset(self.training_points_not_in_A_or_B, labels_for_training_pts_not_in_A_or_B)
        self.train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    def train(self):
        epochs = self.epochs
        loss_plot = torch.zeros(0)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            loss_plot = self.single_training_loop(loss_plot)
        print("Done!")

        return loss_plot

    def single_training_loop(self, loss_plot):
        size = len(self.train_dataloader.dataset)
        self.NN.train()
        
        for batch, (X,y) in enumerate(self.train_dataloader):
            
            # compute prediction error
            pred_pts_not_in_A_or_B = L_q(self.NN,X)
            pred_pts_on_bdy_A = math.sqrt(self.alpha) * self.NN(self.training_points_on_bdy_of_A)
            pred_pts_on_bdy_B = math.sqrt(self.alpha) * self.NN(self.training_points_on_bdy_of_B) - math.sqrt(self.alpha)
            
            
            pred_vector = torch.cat((pred_pts_not_in_A_or_B,pred_pts_on_bdy_A,pred_pts_on_bdy_B))

            """For debugging only. This is for training neural network entirely on boundary A and B"""
            # pred_vector = torch.cat((pred_pts_on_bdy_A,pred_pts_on_bdy_B))
            """Joke"""

            assert pred_vector.shape[-1] == out_size
            truth = torch.zeros_like(pred_vector)
            loss = self.loss_fn(pred_vector,truth)

            # backpropogation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                loss_plot = torch.cat((loss_plot, torch.tensor([loss])))
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        return loss_plot

class PINNTrainingVarTrainData():
    """
    Training implements the PINN loss function model where
    the model to be trained will learn the boundary conditions. 
    """
    def __init__(self,NN,optimizer,loss_fn, epochs, alpha, training_data):
        self.NN = NN
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs
        self.alpha = alpha

        self.training_points_on_bdy_of_A, self.training_points_on_bdy_of_B = get_points_on_A_B()

        self.training_points_not_in_A_or_B = training_data

        labels_for_training_pts_not_in_A_or_B = torch.zeros((len(self.training_points_not_in_A_or_B),out_size))
        train_data = TensorDataset(self.training_points_not_in_A_or_B, labels_for_training_pts_not_in_A_or_B)
        self.train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
    
    def train(self):
        epochs = self.epochs
        loss_plot = torch.zeros(0)
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            loss_plot = self.single_training_loop(loss_plot)
        print("Done!")

        return loss_plot

    def single_training_loop(self, loss_plot):
        size = len(self.train_dataloader.dataset)
        self.NN.train()
        
        for batch, (X,y) in enumerate(self.train_dataloader):
            
            # compute prediction error
            pred_pts_not_in_A_or_B = L_q(self.NN,X)
            pred_pts_on_bdy_A = math.sqrt(self.alpha) * self.NN(self.training_points_on_bdy_of_A)
            pred_pts_on_bdy_B = math.sqrt(self.alpha) * self.NN(self.training_points_on_bdy_of_B) - math.sqrt(self.alpha)
            
            # put errors together into one vector
            pred_vector = torch.cat((pred_pts_not_in_A_or_B,pred_pts_on_bdy_A,pred_pts_on_bdy_B))

            assert pred_vector.shape[-1] == out_size
            truth = torch.zeros_like(pred_vector)
            loss = self.loss_fn(pred_vector,truth)

            # backpropogation
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                loss_plot = torch.cat((loss_plot, torch.tensor([loss])))
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        return loss_plot

def save_model(model=None, layer_array=None, alpha=None, epochs=None, filename= None):
    # save the model into your directory
    if filename != None:
        torch.save(model.state_dict(), filename)
        return filename
    else:
        cwd = os.getcwd()
        if cwd != "/Users/williamclark/Documents/1mathematics/UMD_reu/MaierSteinCode":
            os.chdir("/Users/williamclark/Documents/1mathematics/UMD_reu/MaierSteinCode")
        
        x = datetime.today()
        year, month, day, hour, minute = x.year, x.month, x.day, x.hour, x.minute

        filename = f"maier-stein-model-weights-dateime-{year}-{month}-{day}-{hour}-{minute}-alpha-{alpha}-layer_array-{layer_array}-epochs-{epochs}"
        filepath = "./model_weights/" + filename
        torch.save(model.state_dict(), filepath)
        return filename


# def main():

#     # get_points_on_A_B()

#     train = True

#     layer_array = [in_size,20,out_size]

#     if train:
#         # initialize the model:
#         # model = NeuralNetwork(hidden_size=20, hidden_size2=20)
#         model = NeuralNetwork(layer_array=layer_array)

#         # for_debugging = model.parameters()

#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#         loss_fn = nn.MSELoss()

#         # train the model:
#         my_training_object = Training(model, optimizer, loss_fn, epochs = 100, alpha= 10.0)
#         loss_plot = my_training_object.train()

#         # save the model:
#         filename = save_model(model,layer_array,alpha = my_training_object.alpha, epochs=my_training_object.epochs)

#         # visualize training: 
#         fig, ax = plt.subplots()
    #     ax.plot(loss_plot)
    #     plt.savefig("./training_loss_images/"+ filename+".png")
    #     plt.clf()

    # visualize = True

    # if visualize:

    #     # load model weights
    #     # my_model = NeuralNetwork(hidden_size=20, hidden_size2=20)
    #     my_model = NeuralNetwork(layer_array= layer_array)
    #     if not train:
    #         # change the working directory to the MaierSteinCode folder
    #         cwd = os.getcwd()
    #         if cwd != "/Users/williamclark/Documents/1mathematics/UMD_reu/MaierSteinCode":
    #             os.chdir("/Users/williamclark/Documents/1mathematics/UMD_reu/MaierSteinCode")
    #         filepath = f"./model_weights/maier-stein-model-weights-dateime-2022-7-25-11-52-alpha-100.0-layer_array-[2, 20, 20, 1]-epochs-100"
    #     else: 
    #         filepath = "./model_weights/" + filename
    #     my_model.load_state_dict(torch.load(filepath))

    #     # load the points we want to visualize the model solution on:
    #     x, y, xx, yy, xx_f, yy_f = get_mesh_of_sample_space(-2.0, 2.0, 100, -0.5, 0.5, 100)
    #     input = torch.zeros((100*100,2))
    #     input[:,0] = xx_f
    #     input[:,1] = yy_f
        
    #     z = my_model(input)
    #     z = torch.reshape(z, (len(x), len(y)))

    #     max_z = torch.max(z).item()
    #     min_z = torch.min(z).item()

        # # matplotlib stuff: 
        # fig, ax = plt.subplots()
        # levels = torch.linspace(min_z, max_z, 20)
        # cs = ax.contourf(xx.detach().numpy(), yy.detach().numpy(), z.detach().numpy(), levels = levels)
        # fig.colorbar(cs)
        # ax.scatter(np.array([-1]), np.array([0]))
        # ax.scatter(np.array([1]), np.array([0]))
        # # plt.savefig("./approx_images/"+ filename + ".png")

        # fig, ax = plt.subplots()

        # pts = get_training_pts_not_from_A_B_but_uniform()
        # pts_a, pts_b = get_points_on_A_B()
        # pts = torch.cat((pts,pts_a,pts_b))
        # assert pts.shape[-1] == 2
        # z_new = my_model(pts)
        # cs1 = ax.scatter(pts[:,0].detach().numpy(), pts[:,1].detach().numpy(), c= z_new.detach().numpy())
        # fig.colorbar(cs1)

        # bA, bB = get_points_on_A_B()
        # ax.scatter(bA[:,0].detach().numpy(), bA[:,1].detach().numpy(), s=0.5)
        # ax.scatter(bB[:,0].detach().numpy(), bB[:,1].detach().numpy(), s=0.5)

        # ax.set_title("Neural Network Approximation to Committor Function")
        # plt.savefig("./approx_images/"+ filename + ".png")

        # plt.show()



# if __name__ =="__main__":
#     layer_array = [2,20,1]
#     my_model = NeuralNetwork(layer_array= layer_array)
#     optimizer = torch.optim.Adam(my_model.parameters(), lr=1e-3)
#     loss_fn = nn.MSELoss()
#     my_training_object = TrainingPINNwTrajectoryTrainingData(my_model, optimizer, loss_fn, epochs = 30, alpha= 10.0)
#     my_training_object.train()
#     fig, ax = plt.subplots()

#     pts = my_training_object.training_points_not_in_A_or_B
#     ind = [(pts[i,0]-1)**2+pts[i,1]**2 >= 0.3**2 and (pts[i,0]+1)**2 + pts[i,1]**2 >= 0.3**2 for i in tqdm(range(len(pts)))]
#     pts = pts[ind]
#     pts_a, pts_b = get_points_on_A_B()
#     pts = torch.cat((pts,pts_a,pts_b))
#     assert pts.shape[-1] == 2
#     z_new = my_model(pts)
#     cs1 = ax.scatter(pts[:,0].detach().numpy(), pts[:,1].detach().numpy(), c= z_new.detach().numpy())
#     fig.colorbar(cs1)

#     plt.show()

#     #main()
