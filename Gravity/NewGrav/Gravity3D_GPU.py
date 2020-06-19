import sys
import os
import numpy as np

import timeit
sys.path.append("./gempy")


import gempy as gp
from gempy.core.tensor.tensorflow_graph_test import TFGraph
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd
from gempy import create_data, map_series_to_surfaces
from gempy.assets.geophysics import GravityPreprocessing
tfd = tfp.distributions

def Plot_2D_scaler_field(grid, scaler_field):
    G = grid[np.where(grid[:, 1] == [grid[-1][1]])[0]]
    S = scaler_field.numpy()[np.where(grid[:, 1] == [grid[0][1]])[0]]
    XX = G[:, 0].reshape([50, 50])
    ZZ = G[:, 2].reshape([50, 50])
    S = S.reshape([50, 50])
    plt.contour(XX, ZZ, S)

path = './gempy/notebooks/'
geo_data = create_data([0, 1000, 0, 1000, 0, 1000], resolution=[50, 50, 50],
                       path_o=path+ "/data/input_data/jan_models/model2_orientations.csv",
                       path_i=path + "/data/input_data/jan_models/model2_surface_points.csv")
map_series_to_surfaces(geo_data, {"Strat_Series": (
    'rock2', 'rock1'), "Basement_Series": ('basement')})

geo_data.add_surface_values([2.61, 3.1, 2.92])

# Gravity test
# ---------
grav_res = 20
X = np.linspace(0, 1000, grav_res)
Y = np.linspace(0, 1000, grav_res)
Z = 300
xyz = np.meshgrid(X, Y, Z)
xy_ravel = np.vstack(list(map(np.ravel, xyz))).T


geo_data.set_centered_grid(xy_ravel, resolution=[10, 10, 15], radius=5000)
interpolator = geo_data.interpolator
dtype = interpolator.additional_data.options.df.loc['values', 'dtype']

dips_position, dip_angles, azimuth, polarity, surface_points_coord, fault_drift, grid, values_properties = interpolator.get_python_input_block()[
    0:-3]

centers = geo_data.rescaling.df.loc['values', 'centers'].astype('float32')


g = GravityPreprocessing(geo_data.grid.centered_grid)
tz = g.set_tz_kernel()

len_rest_form = interpolator.additional_data.structure_data.df.loc[
    'values', 'len surfaces surface_points'] - 1
Range = interpolator.additional_data.kriging_data.df.loc['values', 'range']
C_o = interpolator.additional_data.kriging_data.df.loc['values', '$C_o$']
rescale_factor = interpolator.additional_data.rescaling_data.df.loc[
    'values', 'rescaling factor']

rf = rescale_factor
nugget_effect_grad = np.cast[dtype](
    np.tile(interpolator.orientations.df['smooth'], 3))
nugget_effect_scalar = np.cast[interpolator.dtype](
    interpolator.surface_points.df['smooth'])

surface_points_coord = tf.Variable(surface_points_coord, dtype=tf.float32)


TFG = TFGraph(dips_position, dip_angles, azimuth,
              polarity, fault_drift,
              grid, values_properties, len_rest_form, Range,
              C_o, nugget_effect_scalar, nugget_effect_grad,
              rescale_factor)
surface_points_coord = tf.convert_to_tensor(surface_points_coord)

Z_x = TFG.scalar_field(surface_points_coord)
scalar_field_at_surface_points = TFG.get_scalar_field_at_surface_points(Z_x)
formations_block = TFG.export_formation_block(
    Z_x, scalar_field_at_surface_points, values_properties)

lg_0 = interpolator.grid.get_grid_args('centered')[0]
lg_1 = interpolator.grid.get_grid_args('centered')[1]
dips_position = tf.convert_to_tensor(dips_position)
dip_angles = tf.convert_to_tensor(dip_angles)
azimuth = tf.convert_to_tensor(azimuth)
polarity = tf.convert_to_tensor(polarity)

fault_drift = tf.convert_to_tensor(fault_drift)
grid = tf.convert_to_tensor(grid)
values_properties = tf.convert_to_tensor(values_properties)
len_rest_form = tf.convert_to_tensor(len_rest_form)
Range = tf.convert_to_tensor(Range, tf.float32)
C_o = tf.convert_to_tensor(C_o)
nugget_effect_grad = tf.convert_to_tensor(nugget_effect_grad)
nugget_effect_scalar = tf.convert_to_tensor(nugget_effect_scalar)
rescale_factor = tf.convert_to_tensor(rescale_factor, tf.float32)

densities = formations_block[1][lg_0:lg_1]

## Plot the ground truth
grav = TFG.compute_forward_gravity(tz, lg_0, lg_1, densities)
Data = grav


thickness = 200 # here just keep thickness constant
Number_para = int(surface_points_coord.shape[0]/2)
mu_prior = 600 * tf.ones(Number_para, dtype=tf.float32) 
sigma = 10
## covariance = sigma ^2
cov_prior = sigma ** 2 * tf.eye(Number_para, dtype=tf.float32)
cov = 0.08 * tf.eye(grav.shape[0],dtype = tf.float32)

# rescale the prior
mu_prior_r = (mu_prior-centers[2])/rf+0.5001
cov_prior_r = ((sigma) / rf)** 2 * tf.eye(Number_para, dtype=tf.float32)  #$$# double check here

@tf.function
def calculate_grav(surface_coord):
    TFG = TFGraph(dips_position, dip_angles, azimuth,
              polarity, fault_drift,
              grid, values_properties, len_rest_form, Range,
              C_o, nugget_effect_scalar, nugget_effect_grad,
              rescale_factor)
    Z_x = TFG.scalar_field(surface_coord)

    scalar_field_at_surface_points = TFG.get_scalar_field_at_surface_points(Z_x)
    formations_block = TFG.export_formation_block(
        Z_x, scalar_field_at_surface_points, values_properties)

    densities = formations_block[1][lg_0:lg_1]

    gravity = TFG.compute_forward_gravity(tz, lg_0, lg_1, densities)

    return gravity


@tf.function
def joint_log_post(Data, mu):
    """[summary]

    Arguments:
        Data {[Tensor]} -- [description]
        _control_position {[Tensor]} -- [description]cov_prior_r = ((sigma-centers[2])/rf+0.5001)**2 * tf.eye(Number_para, dtype=tf.float32)

    Returns:
        [type] -- [description]
    """
    
    Z_coord = tf.expand_dims(tf.concat([((mu-thickness-centers[2])/rf+0.5001),((mu-centers[2])/rf+0.5001)],axis=0),axis=1)
    surface_coord =tf.concat([surface_points_coord[:,:2],Z_coord],axis = 1)

    # define prior distribution
    mvn_prior = tfd.MultivariateNormalTriL(
        loc=mu_prior_r,
        scale_tril = tf.linalg.cholesky(cov_prior_r))
        
    # define likelihood
    Gm_ = calculate_grav(surface_coord)

    mvn_likelihood = tfd.MultivariateNormalTriL(
        loc=Gm_,
        scale_tril=tf.linalg.cholesky(cov))

    # return the posterior probability
    return (mvn_prior.log_prob(tf.squeeze(Z_coord[Number_para:]))
                + mvn_likelihood.log_prob(Data))


mu = tf.random.normal(mu_prior.shape, mean = 600, stddev = 100, dtype = tf.float32)


start = timeit.default_timer()
with tf.GradientTape() as t:
    t.watch(surface_points_coord)
    loss = TFG.scalar_field(surface_points_coord)
grad = t.gradient(loss,surface_points_coord)
print(grad)

end = timeit.default_timer()
print('time for gradient calculation: %.3f' % (end - start))