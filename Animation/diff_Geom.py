import numpy as np
import matplotlib.pyplot as plt 

num = 100

init_dots = np.random.randint(100, size=(num, 2))

# def LaplacianSmoothing(init_dots):
#     dots = init_dots
#     for i in range(100):
#         dots
#         plt.plot(init_dots[:,0],init_dots[:,1])
#     plt.show()
        
import math

def sin_pi_sq():
    x = np.linspace(0,10,1000)
    y = np.sin(x**2)
    plt.plot(x,y)
    plt.show()

if __name__ =='__main__':
    sin_pi_sq()