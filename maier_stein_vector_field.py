import matplotlib.pyplot as plt
import numpy as np

# author: William Clark

beta = 10.0

def make_direction_field(ax):

    x,y = np.meshgrid(np.linspace(-2,2,10),np.linspace(-2,2,10))

    u = x - x**3 - beta * x * y ** 2
    v = -(1+x**2) * y

    
    ax.quiver(x,y,u,v)

    return ax

def main():

    fig, ax = plt.subplots()
    make_direction_field(ax)
    plt.show()

if __name__=="__main__":
    main()