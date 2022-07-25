import maierstein_nn as ms
import torch
import matplotlib.pyplot as plt
import tqdm as tqdm
import os
from multiprocessing import Pool
from multiprocessing import Process, Lock
import numpy as np

def f(alpha,epochs,hidden_layer,lr):
    print(f"-----------Working on alpha={alpha} and hidden_layer = {hidden_layer}------------")
    model = ms.NeuralNetwork(layer_array=hidden_layer)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    my_training_object = ms.Training(model, optimizer, loss_fn, epochs = epochs, alpha= alpha)
    loss_plot = my_training_object.train()
    filepath = ms.save_model(model,layer_array=hidden_layer,alpha = my_training_object.alpha, epochs=my_training_object.epochs)
    ax = plt.gca()
    ax.set_ylim([0.0, 10])
    plt.plot(loss_plot)
    plt.savefig("./training_loss_images/"+ filepath+".png")
    plt.clf()


if __name__=="__main__":
    os.chdir("/Users/williamclark/Documents/1mathematics/UMD_reu/MaierSteinCode")

    alphas = [3.0, 9.0, 27.0, 81.0]
    epochss = [100]
    hidden_layers = [[2,3,1],[2,9,1],[2,27,1],[2,3,3,1],[2,9,9,1],[2,27,27,1]]
    lrs = [1e-3]

    combinations = [[a,b,c,d] for a in alphas for b in epochss for c in hidden_layers for d in lrs]

    with Pool() as pool:
        pool.starmap(f, combinations)

                        
                    
    

