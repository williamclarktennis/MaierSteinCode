from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
from maier_stein_vector_field import make_direction_field

# author: William Clark

# all column numpy arrays of size 2 will be (2,1) instead of (2,)

# we have beta = 10:
beta = 10.0
epsilon = 0.1
delta_t = 1e-4

def get_b_of_x_y(x_y):
    # x_y should be array of shape (2,1)
    # where x = x_y[0] and 
    # y = x_y[1]
    b_of_x_y = np.zeros((2,1))
    b_of_x_y[0] = x_y[0] - x_y[0]**3 - beta * x_y[0] * x_y[1] ** 2
    b_of_x_y[1] = -(1+x_y[0]**2)* x_y[1]
    return b_of_x_y

def get_sample_path(n_iterations, init_point):
    # init_point must be shape (2,1)

    # we will have sample_path[0] = [0,0]
    sample_path = np.zeros((n_iterations,2,1))
    sample_path[0] = init_point

    # run simulation
    for i in tqdm(range(n_iterations-1)):

        # get value of b at current sample point
        b_of_x_y = get_b_of_x_y(sample_path[i])

        # get guassian sample with mean 0 and variance delta_t.
        # since scale is stand. dev., we have stan-dev = math.sqrt(delta_t)
        stan_dev = math.sqrt(delta_t)
        weiner = np.random.normal(loc = 0.0, scale = stan_dev, size = (2,1)) 

        # compute next sample point: 
        sample_path[i+1] = sample_path[i] + b_of_x_y * delta_t + math.sqrt(epsilon) * weiner

    return sample_path

def get_color(n_iterations):
    color = []
    delta = 3/n_iterations
    channel = 0
    for i in range(n_iterations):
        if channel < 1:
            rgb = (0,0,channel)
        elif channel < 2:
            rgb = (0,channel-1.0,1.0)
        elif channel <= 3:
            rgb = (0.5*(channel - 2.0), 1.0, 1.0)
        color.append(rgb)
        channel += delta

    assert len(color) == n_iterations
    return color

def plot_equilibrium_points(ax):
    ax.scatter(-1,0, s=20, c = ['red'])
    ax.scatter(1,0, s=20, c = ['red'])

    ax.annotate(r"$O_-$", (-1,0), xytext = (-0.8,0.2), arrowprops = {"width":1, "headwidth": 5, "headlength":5}, bbox = dict(boxstyle="round", fc="0.8", pad=0.3))
    ax.annotate(r"$O_+$", (1,0), xytext=(1.2,0.2), arrowprops = {"width":1, "headwidth": 5, "headlength":5}, bbox = dict(boxstyle="round", fc="0.8", pad=0.3))
    

def main():
    init_1 = np.array([[-1],[0]])
    init_2 = np.array([[0],[0]])
    init_3 = np.array([[1],[0]])
    n_iterations = int(1e5)

    init_points = [init_1, init_2, init_3]

    fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()

    my_axes = [ax,ax1,ax2]

    for i in range(len(my_axes)):
        circleA = plt.Circle((-1, 0), 0.3, color='pink')
        circleB = plt.Circle((1, 0), 0.3, color='green')
        my_axes[i].add_patch(circleA)
        my_axes[i].add_patch(circleB)

    assert len(my_axes) == len(init_points)

    # plot direction fields
    for i in my_axes:
        make_direction_field(i)

    color = get_color(n_iterations=n_iterations)

    # plot SDE solutions: 
    for i, i_point in enumerate(init_points):
        sample_path = get_sample_path(n_iterations, i_point)
        my_axes[i].scatter(sample_path[:,0], sample_path[:,1], s=1, c=color)
        #plot regions A and B: 

    
    # plot equilibirum points: 
    for i in my_axes:
        plot_equilibrium_points(i)

    # make axis and title labels
    T = int(n_iterations * delta_t)

    for i, my_ax in enumerate(my_axes): 
        my_ax.set_xlabel("x axis")
        my_ax.set_ylabel("y axis")
        my_ax.set_title(f"Maier Stein SDE Solution on time interval [0,{T}] with initial point ({init_points[i][0,0]}, {init_points[i][1,0]})")
        my_ax.text(-1,-1, "Solution grows lighter as time increases", bbox = dict(boxstyle="round", fc="0.8", pad=0.3))
    plt.show()


if __name__=="__main__":
    main()