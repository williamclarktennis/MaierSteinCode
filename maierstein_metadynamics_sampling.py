
# sampling training data for the maier stein neural network. 
# what is the dimensionality of this training data? 
# answer: torch.Size([*,in_size]) where in_size = 2. 

# Steps: 
# 1) Run euler maruyama with metadynamics and return the
# location of the guassian deposits. 
# 2) Run euler maruyama with the b field 
# as b + \nabla v_bias(all guassian deposits)
# 3) Take the simulation from prior step and
# apply delta rarification to make it trimmer

# Step 1: 
# return with guassian locations. 
# how many guassian locations do we want to retrieve? 
# answer: 1000. 
# What is the dimension of the guassian locations storage array?
# answer: (1000,2)
# Will we have to find derivative of some function of guassian locations? 
# answer: YES. Hence, use torch._require_grad(True)
# what is the timestep length delta_t? 
# answer: delta_t = 1e-5
# what is the total amount of time that we will run the simulation? 
# answer: T = iterations * delta_t
# How often will we deposit guassians? 
# answer: every tau = 1000 iterations. 
# what value will we use for the width parameter? 
# answer: we will first try width_parameter = 5.0
# what value will we use for the height parameter? 
# answer: we will first try height_parameter = 5.0
# So it sounds like the iteration indeces will be 
# 0, 1, ..., iterations-1. Given that we want to retrieve 1000
# guassian locations, what should be the value of iterations? 
# answer: We need exactly (1/tau)*1000 iterations, where
# tau is the number of iterations per deposit. 
# How do we vectorize this implementation as much as possible? 

from random import sample
import torch
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

import maierstein_euler_maruyama as msem
import maier_stein_vector_field as msvf

delta_t = 1e-3
epsilon = 0.1
beta = 10.0
tau = 1000
height_param = 0.2
width_param = 0.025
bumps = 1000
delta = 0.007

x = datetime.today()
year, month, day, hour, minute = x.year, x.month, x.day, x.hour, x.minute

def V_bias(sample_point, g_l):

    assert sample_point.shape == torch.Size([2])

    # make current location available #g_l times for later computation
    s = sample_point.repeat(g_l.shape[0],1)

    arg = ((g_l[:,0]-s[:,0])/width_param)**2 + ((g_l[:,1]-s[:,1])/width_param)**2

    v_bias = torch.sum(height_param * torch.exp(-1/2 * arg))

    return v_bias


def B(sample_point, g_l):
    """
    B = b_field(sample_path[i]) + \nabla V_bias(sample_path[i])
    """
    # get output
    V_bias_output = V_bias(sample_point, g_l)

    # specify the vector in the vector Jacobian product: 
    vector_1 = torch.ones_like(V_bias_output)

    # compute vector Jacobian product
    jacobian_x_y = torch.autograd.grad(outputs= V_bias_output,\
        inputs= (sample_point,g_l),grad_outputs=vector_1)[0]

    assert jacobian_x_y.shape == torch.Size([2])

    x = sample_point[0]
    y = sample_point[1]
    b_field = torch.tensor([x-x**3-beta* x* y**2, -(1+x**2)*y])

    out = b_field - jacobian_x_y

    assert out.shape == torch.Size([2])

    return out

def get_guassian_locations(num_g_l=int(1e3), iterations=int(1e6)):
    # 2 = in_size of neural network.
    sample_path = torch.zeros((iterations,2)).requires_grad_(True)

    sqrt_eps = math.sqrt(epsilon)

    for i in tqdm(range(iterations-1)):
        # get guassian sample with mean 0 and variance delta_t.
        # since scale is stand. dev., we have stan-dev = math.sqrt(delta_t)
        stan_dev = math.sqrt(delta_t) * torch.ones(2)
        mean = torch.zeros(2)
        dW = torch.normal(mean = mean, std = stan_dev) 

        # find out how many guassian deposits have been dropped so far:
        curr_num_g_l = i//tau + 1

        # pick out the guassian locations from the sample_path: 
        ind = torch.linspace(0,tau * (curr_num_g_l-1),curr_num_g_l, dtype = int)
        g_l = sample_path[ind]

        b= B(sample_path[i], g_l)

        # work around the error: "a view of a leaf Variable that requires grad is being used in an in-place operation."
        with torch.no_grad():
            sample_path[i+1] = sample_path[i] +  b * delta_t + sqrt_eps * dW

    # pick out the guassian locations from the sample_path: 
    ind = torch.linspace(0,tau * (num_g_l-1),num_g_l, dtype=int)
    return sample_path[ind], sample_path

def run_biased_sample_path(num_pts, g_l):
    sample_path = torch.zeros((num_pts,2)).requires_grad_(True)
    sqrt_eps = math.sqrt(epsilon)
    for i in tqdm(range(num_pts-1)):
        # get guassian sample with mean 0 and variance delta_t.
        # since scale is stand. dev., we have stan-dev = math.sqrt(delta_t)
        stan_dev = math.sqrt(delta_t) * torch.ones(2)
        mean = torch.zeros(2)
        dW = torch.normal(mean = mean, std = stan_dev)

        # get biased b field vector
        b= B(sample_path[i], g_l)

        # work around the error: "a view of a leaf Variable that requires grad is being used in an in-place operation."
        with torch.no_grad():
            sample_path[i+1] = sample_path[i] +  b * delta_t + sqrt_eps * dW

    return sample_path

def get_delta_rarified_sample_path(sample_path):
    N = len(sample_path)
    mask = torch.ones((N,))
    delta2 = delta**2
    for k in tqdm(range(N)):
        if mask[k]==1:
            dist2 = (sample_path[:,0]-sample_path[k,0])**2 + (sample_path[:,1]-sample_path[k,1])**2
            ind = torch.argwhere(dist2 < delta2)
            mask[ind] = 0
            mask[k]=1

    ind = torch.argwhere(mask==1)
    out = torch.squeeze(sample_path[ind])
    return out

def visualize_sample_path(sample_path):
    n = len(sample_path)

    my_color = msem.get_color(n)
    fig, ax = plt.subplots()
    ax.scatter(sample_path[:,0].detach().numpy(),sample_path[:,1].detach().numpy(),s=1,c=my_color)

    # msvf.make_direction_field(ax)
    # 
    #ax.set_xlim(left=-2,right=2)
    #ax.set_ylim(bottom=-0.5, top=0.5)
    return fig, ax

def visualize_guassian_locations(g_l):
    fig, ax = plt.subplots()
    ax.scatter(g_l[:,0].detach().numpy(),g_l[:,1].detach().numpy(),s=2)
    # 
    #ax.set_xlim(left=-2,right=2)
    #ax.set_ylim(bottom=-0.5, top=0.5)
    return fig, ax

def debug():
    def y(x):
        return x
    def y_1(x):
        return torch.exp(x)
    x = torch.tensor(0.0)
    x.requires_grad_(True)
    y_output = y(x) + y_1(x)
    vector_1 = torch.ones_like(y_output)
    jac = torch.autograd.grad(y_output,x,grad_outputs=vector_1)[0]
    print(jac)

if __name__=="__main__":
    import os
    os.chdir("/Users/williamclark/Documents/1mathematics/UMD_reu/MaierSteinCode")
    
    # STEP 1: 
    # g_l, s_p = get_guassian_locations(bumps,bumps*tau)
    # fig, ax = visualize_sample_path(sample_path=s_p)
    # fig.savefig(f"./guassian_deposits_metadynamics/s_p_image-bumps-{bumps}-tau-{tau}-h_param-{height_param}-w_param-{width_param}-deltat-{delta_t}-datetime-{year}-{month}-{day}-{hour}-{minute}.png")
    # g_l_fig, g_l_ax = visualize_guassian_locations(g_l = g_l)
    # g_l_fig.savefig(f"./guassian_deposits_metadynamics/g_l_image-bumps-{bumps}-tau-{tau}-h_param-{height_param}-w_param-{width_param}-deltat-{delta_t}-datetime-{year}-{month}-{day}-{hour}-{minute}.png")
    # torch.save(g_l,f"./guassian_deposits_metadynamics/g_l-bumps-{bumps}-tau-{tau}-h_param-{height_param}-w_param-{width_param}-deltat-{delta_t}-datetime-{year}-{month}-{day}-{hour}-{minute}.pt")
    # torch.save(s_p,f"./guassian_deposits_metadynamics/sample_path-bumps-{bumps}-tau-{tau}-h_param-{height_param}-w_param-{width_param}-deltat-{delta_t}-datetime-{year}-{month}-{day}-{hour}-{minute}.pt")
    
    # STEP 2:
    # g_l = torch.load("./guassian_deposits_metadynamics/g_l-bumps-1000-tau-1000-h_param-0.2-w_param-0.025-deltat-0.001-datetime-2022-7-28-12-48.pt")
    # num_pts = int(1e6)
    # s_p = run_biased_sample_path(num_pts,g_l)
    # s_p_fig, s_p_ax = visualize_sample_path(s_p)
    # s_p_fig.savefig(f"./training_data/sample_path-num_pts-{num_pts}-bumps-{bumps}-tau-{tau}-h_param-{height_param}-w_param-{width_param}-deltat-{delta_t}-datetime-{year}-{month}-{day}-{hour}-{minute}.png")
    # torch.save(s_p,f"./training_data/sample_path-num_pts-{num_pts}-bumps-{bumps}-tau-{tau}-h_param-{height_param}-w_param-{width_param}-deltat-{delta_t}-datetime-{year}-{month}-{day}-{hour}-{minute}.pt")


    # STEP 3: 
    # delta rarification
    pts = torch.load("./training_data/sample_path-num_pts-1000000-bumps-1000-tau-1000-h_param-0.2-w_param-0.025-deltat-0.001-datetime-2022-7-28-14-20.pt")
    delta_pts = get_delta_rarified_sample_path(pts)
    num_pts = len(delta_pts)
    torch.save(delta_pts,f"./training_data/delta_rarified-num_pts-{num_pts}-delta-{delta}-bumps-{bumps}-tau-{tau}-h_param-{height_param}-w_param-{width_param}-deltat-{delta_t}-datetime-{year}-{month}-{day}-{hour}-{minute}.pt")
    delta_fig, delta_ax = visualize_sample_path(delta_pts)
    delta_fig.savefig(f"./training_data/delta_rarified-num_pts-{num_pts}-delta-{delta}-bumps-{bumps}-tau-{tau}-h_param-{height_param}-w_param-{width_param}-deltat-{delta_t}-datetime-{year}-{month}-{day}-{hour}-{minute}.png")

    plt.show()

