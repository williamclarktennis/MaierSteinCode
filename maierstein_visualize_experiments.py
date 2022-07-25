import matplotlib.pyplot as plt
import os
import json
import maierstein_nn as ms
import torch

import numpy as np

os.chdir("/Users/williamclark/Documents/1mathematics/UMD_reu/MaierSteinCode/model_weights")
filepaths = os.listdir()

pts = ms.get_training_pts_not_from_A_B_but_uniform()
pts_a, pts_b = ms.get_points_on_A_B()
pts = torch.cat((pts,pts_a,pts_b))

count = 0

f = np.empty((0,2),dtype=object)

for filepath in filepaths:

    if "[" not in filepath:
        continue
    if "layer_array" not in filepath:
        continue
    if ".png" in filepath:
        continue
    left_brac_pos = filepath.find("[")
    right_brac_pos = filepath.find("]")
    string_list = filepath[left_brac_pos:right_brac_pos+1]
    layer_array = json.loads(string_list)

    j = np.array([[filepath,layer_array]],dtype=object)
    f = np.concatenate((f,j),axis=0)

    my_model = ms.NeuralNetwork(layer_array= layer_array)
    my_model.load_state_dict(torch.load(filepath))


    # plt can only handle 20 figures in memory at once
    if count % 20 == 0:
        plt.show()

    fig, ax = plt.subplots()

    assert pts.shape[-1] == 2
    z_new = my_model(pts)
    ax.scatter(pts[:,0].detach().numpy(), pts[:,1].detach().numpy(), c= z_new.detach().numpy())
    ax.scatter(pts_a[:,0].detach().numpy(), pts_a[:,1].detach().numpy(), s=0.5)
    ax.scatter(pts_b[:,0].detach().numpy(), pts_b[:,1].detach().numpy(), s=0.5)

    ax.set_title(filepath)
    plt.savefig("../approx_images/" + filepath + ".jpg")

    count += 1

# np.save("filepaths_and_layer_arrays.npy",f)


plt.show()