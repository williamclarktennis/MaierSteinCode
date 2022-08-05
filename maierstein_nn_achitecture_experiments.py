import nn_solver_PINN_approach as ms
import nn_solver_hybrid_approach as ha
import torch
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
import numpy as np
from datetime import datetime

def PINN(training_set, alpha,epochs,layer_array,lr):
    print(f"-----------Working on alpha={alpha} and hidden_layer = {layer_array} and training set of length {len(training_set)}------------")
    
    # initialize model with layer_array
    model = ms.NeuralNetwork(layer_array=layer_array)

    # set up the training:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    my_training_object = ms.PINNTrainingVarTrainData(NN=model, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs, alpha=alpha, training_data=training_set)
    
    # train the model and get the loss plot:
    loss_plot = my_training_object.train()

    # make the model weights file name and save it 
    x = datetime.today()
    year, month, day, hour, minute = x.year, x.month, x.day, x.hour, x.minute
    filename = f"./model_weights/pinn_NumOfTrPts-{len(training_set)}_LayerArray-{layer_array}_alpha-{alpha}_epochs-{epochs}_DATE-{year}-{month}-{day}-{hour}-{minute}"
    torch.save(model.state_dict(), filename)

    # plot the loss function and save it to correct folder
    ax = plt.gca()
    ax.set_ylim([0.0, 10])
    plt.plot(loss_plot)
    plt.savefig("./training_loss_images/"+ f"pinn_NumOfTrPts-{len(training_set)}_LayerArray-{layer_array}_alpha-{alpha}_epochs-{epochs}_DATE-{year}-{month}-{day}-{hour}-{minute}"+".png")
    plt.clf()

def hybrid(training_set, layer_array, lr, epochs): 
    # initialize model with layer_array
    model = ms.NeuralNetwork(layer_array=layer_array)

    # set up the training:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    my_training_object = ms.TrainingLiLin(NN=model, optimizer=optimizer, loss_fn=loss_fn, epochs=epochs, training_data=training_set)

    # train the model and get the loss plot:
    loss_plot = my_training_object.train()

    # make the model weights file name and save it 
    x = datetime.today()
    year, month, day, hour, minute = x.year, x.month, x.day, x.hour, x.minute
    filename = f"./model_weights/hyrbid_NumOfTrPts-{len(training_set)}_LayerArray-{layer_array}_epochs-{epochs}_DATE-{year}-{month}-{day}-{hour}-{minute}"
    torch.save(model.state_dict(), filename)

    # plot the loss function and save it to correct folder
    ax = plt.gca()
    ax.set_ylim([0.0, 10])
    plt.plot(loss_plot)
    plt.savefig("./training_loss_images/"+ f"hyrbid_NumOfTrPts-{len(training_set)}_LayerArray-{layer_array}_epochs-{epochs}_DATE-{year}-{month}-{day}-{hour}-{minute}"+".png")
    plt.clf()


if __name__=="__main__":
    os.chdir("/Users/williamclark/Documents/1mathematics/UMD_reu/MaierSteinCode")

    # T1b = torch.load("./training_data/delta_rarified-AB_REMOVEDnum_pts-956-delta-0.05-bumps-1000-tau-1000-h_param-0.2-w_param-0.025-deltat-0.001-datetime-2022-8-3-16-12.pt")
    # T2b = torch.load("./training_data/delta_rarified-AB_REMOVEDnum_pts-17255-delta-0.01-bumps-1000-tau-1000-h_param-0.2-w_param-0.025-deltat-0.001-datetime-2022-8-3-16-13.pt")
    # T3b = torch.load("./training_data/delta_rarified-AB_REMOVEDnum_pts-31547-delta-0.007-bumps-1000-tau-1000-h_param-0.2-w_param-0.025-deltat-0.001-datetime-2022-8-3-16-19.pt")
    # T4b = torch.load("./training_data/delta_rarified-AB_REMOVEDnum_pts-54636-delta-0.005-bumps-1000-tau-1000-h_param-0.2-w_param-0.025-deltat-0.001-datetime-2022-7-29-11-25.pt")
    T5b = torch.load("./training_data/fem_mesh_points_A_B_removed-True_4224totalpts.pt")
    T6b = torch.load("./training_data/uniformAB_REMOVEDsample9878pts.pt")

    T5b.requires_grad_(True)
    T6b.requires_grad_(True)

    # training_sets = [T1b, T2b, T3b, T4b, T5b, T6b]
    training_sets = [T5b, T6b]
    alphas = [10.0, 15.0, 20.0]
    epochss = [100]
    layer_arrays = [[2,20,1],[2,30,1],[2,20,20,1],[2,30,20,1],[2,30,30,1]]
    lrs = [1e-3]

    combinations = [[T,a,b,c,d] for T in training_sets for a in alphas for b in epochss for c in layer_arrays for d in lrs]

    with Pool() as pool:
        pool.starmap(PINN, combinations)

    # T1 = torch.load("./training_data/delta_rarified-num_pts-1110-delta-0.05-bumps-1000-tau-1000-h_param-0.2-w_param-0.025-deltat-0.001-datetime-2022-8-3-16-12.pt")
    # T2 = torch.load("./training_data/delta_rarified-num_pts-20038-delta-0.01-bumps-1000-tau-1000-h_param-0.2-w_param-0.025-deltat-0.001-datetime-2022-8-3-16-13.pt")
    # T3 = torch.load("./training_data/delta_rarified-num_pts-36597-delta-0.007-bumps-1000-tau-1000-h_param-0.2-w_param-0.025-deltat-0.001-datetime-2022-8-3-16-19.pt")
    # T4 = torch.load("./training_data/delta_rarified-num_pts-63173-delta-0.005-bumps-1000-tau-1000-h_param-0.2-w_param-0.025-deltat-0.001-datetime-2022-7-29-11-25.pt")
    # T5 = torch.load("./training_data/fem_ellipse_with_AB_4888_pts.pt")
    # T6 = torch.load("./training_data/uniformsample11250pts.pt")

    # hybrid_training_sets = [T1, T2, T3, T4, T5, T6]
    # hyrbid_layer_arrays = ? 

    # combinations_h = [[T, layer_array, lr, epochs] for T in training_sets for layer_array in layer_arrays for lr in lrs for epochs in epochss]

    # with Pool() as pool:
    #     pool.starmap(hybrid,combinations)

                        
                    
    

