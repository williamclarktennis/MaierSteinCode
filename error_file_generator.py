import os
import nn_solver_PINN_approach as ms
import numpy as np
import json
import torch
import math
import pandas as pd

q_FEM = np.loadtxt('MaierStein_q_FEM.csv', delimiter=',', dtype=float)
pts = np.loadtxt('MaierStein_pts.csv', delimiter=',', dtype=float)
# work around the error of inputting numpy data into torch nn
pts_torch = torch.tensor(pts).float()

class Filepaths():
    def __init__(self, keyword, dir_path):
        self.dir_path = dir_path
        self.keyword = keyword
        self.filepaths = self.get_filepaths()
    
    def get_filepaths(self):
        filepaths = np.array(os.listdir(self.dir_path))
        ind = [self.keyword in filepaths[i] for i in range(len(filepaths))]
        filepaths = filepaths[ind]
        return filepaths

class ModelWeightsFeatures():
    def __init__(self, filepath):
        self.filepath = filepath
        self.layer_array = self.get_layer_array()
        self.num_pts = self.get_num_pts()
        self.alpha = self.get_alpha()
        self.model = self.get_model()
        self.q = self.model(pts_torch).detach().numpy().squeeze()
        self.error_tuple = self.get_error_tuple()

    def get_error_tuple(self):
        rmse = math.sqrt(np.sum((self.q-q_FEM)**2) / len(pts))
        mae = np.sum(np.absolute(self.q - q_FEM)) / len(pts)
        ma = np.amax(np.absolute(self.q - q_FEM))
        return (rmse, mae, ma)

    def get_model(self):
        model = ms.NeuralNetwork(layer_array=self.layer_array)
        model.load_state_dict(torch.load("./model_weights1/" + self.filepath))
        return model
    
    def get_alpha(self):
        astr = "alpha-"
        l_astr = len("alpha-")
        i = self.filepath.find(astr) + l_astr
        # find the first - after i:
        j = self.filepath.find("_", i)
        alpha = float(self.filepath[i:j])
        return alpha

    def get_num_pts(self):
        str = "NumOfTrPts-"
        l_str = len(str)
        i = self.filepath.find(str) + l_str
        j = self.filepath.find("_", i)
        num_pts = int(self.filepath[i:j])
        return num_pts

    def get_layer_array(self):
        left_brac_pos = self.filepath.find("[")
        right_brac_pos = self.filepath.find("]")
        string_list = self.filepath[left_brac_pos:right_brac_pos+1]
        layer_array = json.loads(string_list)   
        return layer_array

class HybridModelWeightsFeatures():
    "hyrbid_NumOfTrPts-956_LayerArray-[2, 20, 20, 1]_epochs-100_DATE-2022-8-11-14-45"
    def __init__(self, filepath):
        self.filepath = filepath
        self.layer_array = self.get_layer_array()
        self.num_pts = self.get_num_pts()
        self.model = self.get_model()
        self.q = self.model(pts_torch).detach().numpy().squeeze()
        self.error_tuple = self.get_error_tuple()

    def get_error_tuple(self):
        rmse = math.sqrt(np.sum((self.q-q_FEM)**2) / len(pts))
        mae = np.sum(np.absolute(self.q - q_FEM)) / len(pts)
        ma = np.amax(np.absolute(self.q - q_FEM))
        return (rmse, mae, ma)

    def get_model(self):
        model = ms.NeuralNetwork(layer_array=self.layer_array)
        model.load_state_dict(torch.load("./model_weights1/" + self.filepath))
        return model

    def get_num_pts(self):
        str = "NumOfTrPts-"
        l_str = len(str)
        i = self.filepath.find(str) + l_str
        j = self.filepath.find("_", i)
        num_pts = int(self.filepath[i:j])
        return num_pts

    def get_layer_array(self):
        left_brac_pos = self.filepath.find("[")
        right_brac_pos = self.filepath.find("]")
        string_list = self.filepath[left_brac_pos:right_brac_pos+1]
        layer_array = json.loads(string_list)   
        return layer_array

if __name__=="__main__":
    # my_f = Filepaths(keyword="pinn", dir_path="./model_weights1/")
    # model_features_list = np.array([ModelWeightsFeatures(filepath) for filepath in my_f.filepaths])
    # lines = np.array([[f"{obj.num_pts}",f"{obj.alpha}",f"{obj.layer_array}",f"{obj.error_tuple[0]:.3f}",f"{obj.error_tuple[1]:.3f}",f"{obj.error_tuple[2]:.3f}"] for obj in model_features_list])
    # columns = ["# of training points", "alpha", "neural net architecture", "RMSE", "mean absolute error", "max absolute error"]
    # df = pd.DataFrame(lines,columns=columns)
    # df.to_csv("./error_data1/pinn_error_data.csv")

    my_f = Filepaths(keyword="hyrbid", dir_path="./model_weights1/")
    model_features_list = np.array([HybridModelWeightsFeatures(filepath) for filepath in my_f.filepaths])
    lines = np.array([[f"{obj.num_pts}",f"{obj.layer_array}",f"{obj.error_tuple[0]:.3f}",f"{obj.error_tuple[1]:.3f}",f"{obj.error_tuple[2]:.3f}"] for obj in model_features_list])
    columns = ["# of training points", "neural net architecture", "RMSE", "mean absolute error", "max absolute error"]
    df = pd.DataFrame(lines,columns=columns)
    df.to_csv("./error_data1/hybrid_error_data.csv")

