"""
1.1 -Basics of geological modeling with GemPy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

"""
# %%
# Importing GemPy
import gempy as gp

# Importing auxiliary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from gempy import create_data, map_series_to_surfaces

# %% 
geo_model = gp.create_model('model9_theano')

# %%
data_path = '/Users/zhouji/Google Drive/RWTH/GP_old/notebooks'
# Importing the data from CSV-files and setting extent and resolution
gp.init_data(geo_model, [0, 1000., 0, 1000., 0, 1000.], [50, 50, 50],
             path_o=data_path + "/data/input_data/George_models/irregu_orientations.csv",
             path_i=data_path + "/data/input_data/George_models/irregu_surface_points.csv",
             default_values=True)


# %%
map_series_to_surfaces(geo_model, {"Strat_Series": (
    'rock2', 'rock1'), "Basement_Series": ('basement')})


# %%
plot = gp.plot_2d(geo_model, show_lith=False, show_boundaries=False)
plt.show()

# %%
# gpv = gp.plot_3d(geo_model, image=False, plotter_type='basic')

# %% 
gp.set_interpolator(geo_model,
                    compile_theano=True,
                    theano_optimizer='fast_compile',
                    )


# %% 
sol = gp.compute_model(geo_model)

# %%
gp.plot_2d(geo_model, show_data=True,)
plt.show()

# %%
gp.plot_2d(geo_model, show_data=False, show_scalar=True, show_lith=False)
plt.show()

# %%
gp.plot_2d(geo_model, series_n=1, show_data=False, show_scalar=True, show_lith=False)
plt.show()

# %%
# This illustrates well the fold-related deformation of the stratigraphy,
# as well as the way the layers are influenced by the fault.
# 
# The fault network modeling solutions can be visualized in the same way:
# 

# %% 
geo_model.solutions.scalar_field_at_surface_points

# %%
gp.plot_2d(geo_model, show_block=True, show_lith=False)
plt.show()

# %%
gp.plot_2d(geo_model, series_n=1, show_block=True, show_lith=False)
plt.show()

# %%
# Marching cubes and vtk visualization
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# In addition to 2D sections we can extract surfaces to visualize in 3D
# renderers. Surfaces can be visualized as 3D triangle complexes in VTK
# (see function plot\_surfaces\_3D below). To create these triangles, we
# need to extract respective vertices and simplices from the potential
# fields of lithologies and faults. This process is automatized in GemPy
# with the function get\_surface
# 

# %% 
ver, sim = gp.get_surfaces(geo_model)
gpv = gp.plot_3d(geo_model, image=False, plotter_type='basic')

# %%
# Using the rescaled interpolation data, we can also run our 3D VTK
# visualization in an interactive mode which allows us to alter and update
# our model in real time. Similarly to the interactive 3D visualization of
# our input data, the changes are permanently saved (in the
# InterpolationInput dataframe object). Additionally, the resulting changes
# in the geological models are re-computed in real time.
# 


# %%
# Adding topography
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
geo_model.set_topography(d_z=(350, 750))

# %%
gp.compute_model(geo_model)
gp.plot_2d(geo_model, show_topography=True)
plt.show()


# sphinx_gallery_thumbnail_number = 9
gpv = gp.plot_3d(geo_model, plotter_type='basic', show_topography=True, show_surfaces=True,
                 show_lith=True,
                 image=False)

# %%
# Compute at a given location
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 
# This is done by modifying the grid to a custom grid and recomputing.
# Notice that the results are given as *grid + surfaces\_points\_ref +
# surface\_points\_rest locations*
# 

# %% 
x_i = np.array([[3, 5, 6]])
sol = gp.compute_model(geo_model, at=x_i)

# %%
# Therefore if we just want the value at **x\_i**:

# %%
sol.custom

# %%
# This return the id, and the scalar field values for each series

# %%
# Save the model
# ~~~~~~~~~~~~~~
# 

# %%
# GemPy uses Python [pickle] for fast storing temporary objects
# (https://docs.python.org/3/library/pickle.html). However, module version
# consistency is required. For loading a pickle into GemPy, you have to
# make sure that you are using the same version of pickle and dependent
# modules (e.g.: ``Pandas``, ``NumPy``) as were used when the data was
# originally stored.
# 
# For long term-safer storage we can export the ``pandas.DataFrames`` to
# csv by using:
# 

# %% 
gp.save_model(geo_model)
