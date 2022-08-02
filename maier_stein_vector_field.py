import matplotlib.pyplot as plt
import numpy as np

# author: William Clark

beta = 10.0

def make_direction_field(ax):

    x,y = np.meshgrid(np.linspace(-2,2,20),np.linspace(-2,2,20))

    u = x - x**3 - beta * x * y ** 2
    v = -(1+x**2) * y

    u_shape = u.shape
    v_shape = v.shape
    u_1 = u.reshape((*u_shape,1))
    v_1 = v.reshape((*v_shape,1))
    vector = np.concatenate((u_1,v_1), axis=2)

    norm = np.linalg.norm(vector, axis=2)

    u = u/ (norm)
    v = v/ (norm)


    
    ax.quiver(x,y,u,v)

    return ax

def main():

    fig, ax = plt.subplots()
    make_direction_field(ax)
    plt.show()

if __name__=="__main__":
    main()