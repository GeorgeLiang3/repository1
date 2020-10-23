import tensorflow as tf


from ipywidgets import interact, interactive
import os
import numpy as np
import sys
import timeit
import csv
sys.path.append('/Users/zhouji/Google Drive/RWTH/GP_old')

import matplotlib.pyplot as plt

import gempy as gp
from gempy.core.tensor.tensorflow_graph_test import TFGraph
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
from gempy import create_data, map_series_to_surfaces
from gempy.assets.geophysics import GravityPreprocessing
tfd = tfp.distributions


geo_model = gp.create_model('Fold')

path = '/Users/zhouji/Google Drive/RWTH/GP_old/notebooks/'
# set regular grid to low resolution to save memory

gp.init_data(geo_model, [0, 1000., 0, 1000., 0, 1000.], [50, 50, 50],
                 path_o=path+ "/data/input_data/George_models/model2_1_orientations.csv",
                  path_i=path + '/data/input_data/George_models/'+'model2'+'_surface_points.csv',
             default_values=True)

gp.map_stack_to_surfaces(geo_model,
                        {"Strat_Series": (
    'rock2', 'rock1'), "Basement_Series": ('basement')},
                         remove_unused_series=True)

gp.set_interpolator(geo_model,
                    compile_theano=True,
                    theano_optimizer='fast_compile',
                    )

sol = gp.compute_model(geo_model)

gp.plot_3d(geo_model,show_results = False, show_lith=False,notebook=False)