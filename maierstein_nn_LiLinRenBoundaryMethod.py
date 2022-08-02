from torch import nn
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import maierstein_nn as ms

from tqdm import tqdm

import os 

os.chdir("/Users/williamclark/Documents/1mathematics/UMD_reu/MaierSteinCode")

in_size = 2
out_size = 1

def visualize_chiAchiB():
    """
    chiA should change smoothly from 0 to 1 when inputs transition from boundary of 
    A^iota to boundary of A. chiA should be equal to 0 everywhere besides A^iota.
    """
    # define tanh function 
    tanh = nn.Tanh()

    # get omega (the sample space) ([-2,2]x[-0.5,0.5])
    x = torch.linspace(-2,2,100)
    y = torch.linspace(-0.5,0.5,100)
    xx, yy = torch.meshgrid(x,y)

    # define smooth function chiA
    tanargA = 10* ((xx+1)**2+yy**2 - (0.3 + 0.02)**2 )
    chiA = 1/2 - 1/2 * tanh(tanargA)

    # define smooth function chiB
    tanargB = 10* ((xx-1)**2 + yy**2 - (0.3+0.02)**2 )
    chiB = 1/2 - 1/2 * tanh(tanargB)

    fig, ax = plt.subplots()
    sc = ax.scatter(xx,yy,c=chiA)
    # ax.set_xlim(-1.5,-0.5)
    fig.colorbar(sc)

    fig, ax = plt.subplots()
    sc = ax.scatter(xx,yy,c=chiB)
    # ax.set_xlim(0.5,1.5)
    fig.colorbar(sc)

def chiAchiB(X):

    assert X.shape == torch.Size([len(X),in_size])

    # define tanh function 
    tanh = nn.Tanh()

    # define smooth function chiA
    tanargA = 10* ((X[:,0]+1)**2+X[:,1]**2 - (0.3 + 0.02)**2 )
    chiA = 1/2 - 1/2 * tanh(tanargA)

    # define smooth function chiB
    tanargB = 10* ((X[:,0]-1)**2 + X[:,1]**2 - (0.3+0.02)**2 )
    chiB = 1/2 - 1/2 * tanh(tanargB)

    chiA = chiA[:,None]
    chiB = chiB[:,None]

    return chiA, chiB

def q_t(X, NN):

    chiA, chiB = chiAchiB(X)
    out = (1-chiA)*((1-chiB)* NN(X)+chiB)
    return out

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

def get_Laplacian_partialx_partialy(NN,trial_solution, X: torch.tensor) -> torch.tensor:
    """
    
    """
    assert X.shape[-1] == in_size

    # get outputs 
    q_outputs = trial_solution(X, NN)

    # specify the vector in the vector Jacobian product: 
    vector_1 = torch.ones_like(q_outputs)

    # compute vector Jacobian product
    jacobian_x_y = torch.autograd.grad(outputs= q_outputs,\
        inputs= X,grad_outputs=vector_1,allow_unused=True,\
            retain_graph=True,create_graph=True)

    # get first column of vector Jacobian product, i.e: partial 
    # deriv of q with respect to x
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
    #laplacian = jacobian_xx_xy[0][:,1] + jacobian_yx_yy[0][:,0]

    laplacian = laplacian[:,None]
    partial_x = partial_x[:,None]
    partial_y = partial_y[:,None]

    assert laplacian.shape[-1] == out_size
    assert partial_x.shape[-1] == out_size
    assert partial_y.shape[-1] == out_size

    return laplacian, partial_x, partial_y

def L_q(NN, trial_sol, X:torch.tensor) -> torch.tensor:
    """
    Apply the Kolomogorov backwards operator to q at X
    """

    assert X.shape[-1] == in_size

    x = X[:,0][:,None]
    y = X[:,1][:,None]

    assert x.shape[-1] == out_size
    assert y.shape[-1] == out_size

    laplacian_q, partial_x, partial_y = get_Laplacian_partialx_partialy(NN , trial_sol , X)

    lq = (x-x**3-10*x*y**2)* partial_x - (1+x**2)*y * partial_y + 0.1/2 * laplacian_q
    # lq = (y-y**3-10*y*x**2)* partial_y - (1+y**2)*x * partial_x + 0.1/2 * laplacian_q

    assert lq.shape[-1] == out_size

    return lq


class TrainingLiLin():
    """
    Training implements the PINN loss function model where
    the model to be trained will learn the boundary conditions. 
    """
    def __init__(self,NN,optimizer,loss_fn, epochs, training_points):
        self.NN = NN
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.epochs = epochs




        # toss out the training points in A and B that aren't boundary points:
        # ind = [(x[i,0]-1)**2+x[i,1]**2 >= 0.3**2 and (x[i,0]+1)**2 + x[i,1]**2 >= 0.3**2 for i in tqdm(range(len(x)))]
        # self.training_points = torch.squeeze(x[ind])

        # self.training_points = x

        # make the labels
        self.training_points = training_points


        labels = torch.zeros((len(self.training_points),out_size))

        train_data = TensorDataset(self.training_points, labels)
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
            # X = torch.cat((X,self.bdyA,self.bdyB))
            pred = L_q(NN = self.NN,trial_sol = q_t,X = X)
            assert pred.shape[-1] == out_size
            # y = torch.zeros_like(pred)
            loss = self.loss_fn(pred,y)

            # backpropogation
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                loss_plot = torch.cat((loss_plot, torch.tensor([loss])))
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        return loss_plot

def visualize_sample_path(sample_path):
    n = len(sample_path)
    import maierstein_euler_maruyama as msem
    my_color = msem.get_color(n)
    fig, ax = plt.subplots()
    ax.scatter(sample_path[:,0].detach().numpy(),sample_path[:,1].detach().numpy(),s=1,c=my_color)
    import maier_stein_vector_field as msvf
    msvf.make_direction_field(ax)

if __name__=="__main__":
    
    visualize_chiAchiB()
    plt.show()

    train = True
    if train:
        # initialize the model:
        layer_array = [2,20,1]
        model = NeuralNetwork(layer_array=layer_array)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        # train the model:
        x = torch.load("./training_data/delta_rarified-num_pts-63173-delta-0.005-bumps-1000-tau-1000-h_param-0.2-w_param-0.025-deltat-0.001-datetime-2022-7-29-11-25.pt")
        
        # work around the saved data problem:
        training_points = x.clone()


        # training_points = ms.get_training_pts_not_from_A_B_but_uniform()
        # bdya, bdyb = ms.get_points_on_A_B()
        # training_points = torch.cat((training_points, bdya, bdyb))

        my_training_object = TrainingLiLin(model, optimizer, loss_fn, epochs = 20, training_points=training_points)
        loss_plot = my_training_object.train()

        pts = my_training_object.training_points
        visualize_sample_path(pts)

        assert pts.shape[-1] == 2

        # remove 
        ind = [(pts[i,0]-1)**2+pts[i,1]**2 >= 0.3**2 and (pts[i,0]+1)**2 + pts[i,1]**2 >= 0.3**2 for i in tqdm(range(len(pts)))]
        pts = pts[ind]

        # z_new = q_t(pts, model)
        chiA, chiB = chiAchiB(pts)
        z_new = (1-chiA)* ((1-chiB)* model(pts) + chiB)

        

        fig, ax = plt.subplots()
        cs1 = ax.scatter(pts[:,0].detach().numpy(), pts[:,1].detach().numpy(), c= z_new.detach().numpy())
        fig.colorbar(cs1)

        plt.show()

        # save the model:
        # filename = save_model(model,layer_array,alpha = my_training_object.alpha, epochs=my_training_object.epochs)

        # visualize training: 
        # fig, ax = plt.subplots()
        # ax.plot(loss_plot)
        # plt.savefig("./training_loss_images/"+ filename+".png")
        # plt.clf()

    # visualize_chiAchiB()
    # plt.show()