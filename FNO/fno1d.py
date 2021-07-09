import numpy as np
import torch
import matplotlib.pyplot as plt
import scipy.fft

x = np.mgrid[:3, :3, :3 ]

scipy.fft.fftn(x, axes=(1, 2))