import tensorflow as tf

import gempy as gp


from ipywidgets import interact, interactive
import os
import numpy as np
import sys
import timeit
import csv
import pandas as pd

import matplotlib.pyplot as plt


import tensorflow_probability as tfp
import pandas as pd
from gempy import create_data, map_series_to_surfaces
from gempy.assets.geophysics import GravityPreprocessing
tfd = tfp.distributions
sys.path.append('/Users/zhouji/Google Drive/RWTH/')
from regularModel import *

modelName = 'model2'


path = '/Users/zhouji/Google Drive/RWTH/GP_old/notebooks'
orientation_path = "/data/input_data/George_models/model2_1_orientations.csv"
surface_path = '/data/input_data/George_models/'+modelName+'_surface_points.csv'

grav_res_x = 4
grav_res_y = 4
# X = [250,300, 700,750]
# Y = [250,300, 700,750]

X = [200,400,600,800]
Y = [200,400,600,800]

# X = np.linspace(250, 750, grav_res_x)
# Y = np.linspace(250, 750, grav_res_y)
r = []
for x in X:
  for y in Y:
    r.append(np.array([x,y]))
receivers = np.array(r)


model1 = Model(path,surface_path,orientation_path,receivers = receivers,dtype='float64')
# model1.plot_model()

mu = model1.mu_true
model1.scalar_field(mu)

gp.plot_3d(model1,show_scalar=False,show_lith = False,show_surfaces=False, notebook=False,scalar_field = 'Strat_Series')