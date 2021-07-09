#%%

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import timeit
import sys
sys.path.append('..')
from Rosenbrock_A import Rosenbrock_dist

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.animation as animation
tfd = tfp.distributions
tfdtype = tf.float32

def constant(x):
    return tf.constant(x, tfdtype)

#%%
# # Initialize a single 4-variate Gaussian.
# mu = [1., 2, 3 ,4]
# cov = [[ 0.36,  0.12,  0.06, 0.05],
#        [ 0.12,  0.29, -0.13, 0.10],
#        [ 0.06, -0.13,  0.26, 0.32],
#        [ 0.06, -0.13,  0.26, 0.32]]


# Initialize a single 2-variate Gaussian.
mu = [0., 0]
cov = [[ 0.36,  0.12,],
       [ 0.12,  0.29,]]
RB_dist = Rosenbrock_dist(mu = mu, cov = cov)

fig,ax = RB_dist.draw_post()
# @tf.function
# def log_prob(eva_point):
#     mvn = tfd.MultivariateNormalFullCovariance(
#     loc=mu,
#     covariance_matrix=cov)
#     return mvn.log_prob(eva_point)

def negative_log_posterior(mu):
    return tf.negative(RB_dist.joint_log_post(mu))

# %%
# mu0=tf.convert_to_tensor([-1.,0.2,3.2,1.])
mu0=tf.convert_to_tensor([-0.75,0.5])



#%%
# Function for printing
def loss(mu):
    lost =  negative_log_posterior(mu)[0]
    return lost

# Function for tensorflow optimizer
def loss_minimize():
    lost =  negative_log_posterior(mu)
    return lost

# Gradient descent
Adam = tf.keras.optimizers.Adam(
    learning_rate=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08
)

mu_init = mu0
cost_A = []
mu_list = [mu_init]
mu = tf.Variable(mu_init)
start = timeit.default_timer()

for step in range(200):

    Adam.minimize(loss_minimize, var_list=[mu])
    cost_A.append(loss(mu).numpy())

    mu_list.append(mu.numpy())
end = timeit.default_timer()
print('Time Adam: ',end - start)
Mu_array = np.array(mu_list)


## plotting the trace
fig,ax = RB_dist.draw_post()

ax.plot(Mu_array[:,0],Mu_array[:,1],c = 'k')
ax.scatter(Mu_array[-1,0],Mu_array[-1,1],c='k')
ax.scatter(Mu_array[0,0],Mu_array[0,1],c='r')
plt.show()

MAP = mu_list[-1]
# MAP = [-0.5,0.2]
# MAP = [[0.25,0.05]]
MAP = tf.convert_to_tensor(MAP,dtype=tf.float32)
# MAP = tf.expand_dims(MAP,axis=0)

#%%
def explicityHessian(MAP):
  Hess = tf.TensorArray(tf.float32, size=2)
  for i in range(2):

    tangents = np.zeros(MAP.shape)
    tangents[i]=1
    tangents = tf.convert_to_tensor(tangents,dtype=tf.float32)

    with tf.autodiff.ForwardAccumulator(MAP, tangents) as acc:
      with tf.GradientTape(watch_accessed_variables=False) as t:
        t.watch(MAP)
        joint_log =  loss(MAP)
      grad = t.gradient(joint_log,MAP)
    hess = acc.jvp(grad)
    Hess = Hess.write(i, hess)
  return(Hess.stack())

start = timeit.default_timer()
Hess = explicityHessian(MAP)
end = timeit.default_timer()
print('time for gradient calculation: %.3f' % (end - start))

# Hess = tf.squeeze(Hess)
# %%

def Laplace_appro(Hessian,C_prior):
    cov_post = tf.linalg.inv(
        (tf.add(Hessian, tf.linalg.inv(C_prior))))
    return cov_post

cov_post = Laplace_appro(Hess,tf.constant(cov, tf.float32))

#%%
from scipy.stats import multivariate_normal
loc  = MAP

# cov = cov_post
x, y = np.mgrid[MAP[0]-0.9:MAP[0]+0.9:.01, MAP[1]-0.5:MAP[1]+0.5:.01]

pos = np.empty(x.shape + (2,)) 
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = tfd.MultivariateNormalTriL(MAP,scale_tril=tf.linalg.cholesky(cov_post))
vmin = rv.log_prob(MAP+tf.constant([0.5,0.05]))
vmax = rv.log_prob(MAP)
lvls = np.linspace(vmin,vmax,3)

fig,ax = RB_dist.draw_post()
ax.plot(MAP[0],MAP[1],'r.')
ax.set_title('Laplacian approximation at MAP',fontsize=20)
ax.contour(x,y,rv.log_prob(pos),levels = lvls, cmap = 'Reds')

# %%
## BFGS
tfm = tfp.math
tfm.value_and_gradient(loss, MAP)

# %%
import contextlib
import functools
import time

def make_val_and_grad_fn(value_fn):
  @functools.wraps(value_fn)
  def val_and_grad(x):
    return tfp.math.value_and_gradient(value_fn, x)
  return val_and_grad


@contextlib.contextmanager
def timed_execution():
  t0 = time.time()
  yield
  dt = time.time() - t0
  print('Evaluation took: %f seconds' % dt)


def np_value(tensor):
  """Get numpy value out of possibly nested tuple of tensors."""
  if isinstance(tensor, tuple):
    return type(tensor)(*(np_value(t) for t in tensor))
  else:
    return tensor.numpy()

def run(optimizer):
  """Run an optimizer and measure it's evaluation time."""
  optimizer()  # Warmup.
  with timed_execution():
    result = optimizer()
  return np_value(result)

#%%

@make_val_and_grad_fn
def joint_log(MAP):
  return loss(MAP)

start = mu0

tolerance = 1e-10

@tf.function
def joint_log_with_bfgs():
  return tfp.optimizer.bfgs_minimize(
    joint_log,
    initial_position=tf.constant(start),
    tolerance=tolerance)

results = run(joint_log_with_bfgs)

print('BFGS Results')
print('Converged:', results.converged)
print('Location of the minimum:', results.position)
print('Number of iterations:', results.num_iterations)

BFGS_inverse_hessian = results.inverse_hessian_estimate
BFGS_hessian_estimate = tf.linalg.inv(BFGS_inverse_hessian)

BFGS_MAP = results.position

fig,ax = RB_dist.draw_post()
ax.plot(MAP[0],MAP[1],'r.')
ax.set_title('Laplacian approximation at MAP',fontsize=20)
ax.contour(x,y,rv.log_prob(pos),levels = lvls, cmap = 'Reds')

ax.plot(BFGS_MAP[0],BFGS_MAP[1],'b.')


# %%
H = explicityHessian(constant(BFGS_MAP))

# %%
Map = constant(BFGS_MAP)
with tf.GradientTape(watch_accessed_variables=False) as t:
  t.watch(Map)
  pst =  loss(Map)
grad = t.gradient(pst,Map)

g = grad.numpy().reshape(1,2)
h_approx = g.T@g

print('exact Hessian by AD:', Hess)
print('approximated Hessian by BFGS:',BFGS_hessian_estimate)