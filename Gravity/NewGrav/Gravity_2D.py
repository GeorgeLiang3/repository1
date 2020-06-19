import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pandas.plotting import autocorrelation_plot
import tensorflow_probability as tfp
import timeit
import math as m
import pandas as pd
from matplotlib.colors import from_levels_and_colors
from HessianMCMC import HessianMCMC
from Grav_polygon import constant64,Gravity_Polygon
from GaussianProcess import GaussianProcess2Dlayer

tfb = tfp.bijectors
tfd = tfp.distributions
psd_kernels = tfp.math.psd_kernels


## define some numbers
Number_para = 5
obs_N = 10
number_burnin = 3000
number_sample = 10000
steps_gradient_decent = 2000

pi = constant64(m.pi) # define PI in Tensorflow form

depth = constant64(-100)
thickness = constant64(30)

Range = constant64([-200.,200.])

# prior
mu_prior = -50.*tf.ones([Number_para],dtype = tf.float64)
cov_prior = 15.*tf.eye(Number_para,dtype = tf.float64)

x_obv = tf.linspace(Range[0],Range[1],obs_N)

rho = 2000
# likelihood
sig_e = constant64(2*1e-8)
cov=sig_e ** 2.*tf.eye(obs_N,dtype = tf.float64)

model = Gravity_Polygon(obs_N, Range, rho, thickness, Number_para)

model.set_prior(mu_prior, cov_prior, cov)
model.depth = depth

## define the ground truth
tf.random.set_seed(8)

# X- values: uniformly distributed
control_index = tf.linspace(Range[0],Range[1],Number_para)
# control_index with non-even distribution
# control_index = tf.linspace(Range[0],Range[1],Number_para)+tf.random.uniform([Number_para],-20,20,dtype = tf.float64)

## define the true z-values

True_position_sin = 20*tf.sin(0.01*control_index)+depth

### Plotting function

def pdense(x, y, sigma,M=1000,midpoint = 0.2,lable=True, **kwargs):
    """ Plot probability density of y with known stddev sigma
    """
    assert len(x) == len(y) and len(x) == len(sigma)
    N = len(x)
    # TODO: better y ranging
    ymin, ymax = min(y - 2 * sigma), max(y + 2 * sigma)

    yy = np.linspace(ymin, ymax, M)[::-1]
    a = [np.exp(-((Y - yy) / s) ** 2) / s for Y, s in zip(y, sigma)]
    A = np.array(a)

    vmax, vmin = A.max(),A.min()
    num_levels = 3000

    levels = np.linspace(vmax, vmin, num_levels)[::-1]
    midp = np.mean(np.c_[levels[:-1], levels[1:]], axis=1)

    vals = np.interp(midp, [0, midpoint, vmax], [1, 0.3, 0])
    colors = plt.cm.gray(vals)
    cmap, norm = from_levels_and_colors(levels, colors)

    A = A.reshape(N, M)
    
    if 'ax' in kwargs:
        ax = kwargs.get('ax')
        fig = kwargs.get('f')
    else: fig, ax = plt.subplots()
    img = ax.imshow(A.T, cmap=cmap, aspect='auto',
               origin='upper',norm=norm ,extent=(min(x)[0], max(x)[0], ymin, ymax))
    ax.set_title('Density plot')
    if 'title' in kwargs:
        ax.set_title(kwargs.get('title'))
    if lable == True:

        v1 = np.round(np.linspace(A.min(), A.max(), 8, endpoint=True),2)
        cbar = fig.colorbar(img,ticks=v1,ax = ax)
        cbar.ax.set_ylabel('standard diviation')

def simulated_gravity(x,z,x_obs=None,Number_=obs_N ,R=100,ax = None,style = None,**kwargs):
    """
    kwargs:
        x_obs: x coordinates of observation points
        Number_: number of observation points, if x_obs is not given
    return: 
        gravity: Tensor
    """
    if ax is None:
        f,ax=plt.subplots()
    if x_obs is None:
        x_obs = np.linspace(-R,R,Number_)
    y = np.zeros(np.shape(x_obs))
    obv = np.vstack((x_obs,y)).T

    gravity = model.calculate_gravity(x,z)
    if style is None:
        style = '-'
    ax.set_title('gravity response at surface')
    ax.set_ylabel('g_z ($m/s^2$ )')

    ax.set_xlim(-R,R)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    ax.plot(x_obs,gravity,style,**kwargs)
    return gravity

def Draw_inter(_control_index,_control_position,x_ = None, z_ = None, z_true = None,R = 100, ax = None,**kwargs):
    if ax is None:
        _,ax = plt.subplots(figsize = (14,7))
    if z_true is not None:
        ## true poistion is black
        ax.scatter(control_index,z_true, c = 'black',label = 'true position',alpha = 0.6)
    
    if x_ is None:
        x_,z_ = gp.GaussianProcess(_control_index,_control_position)
    ax.plot(x_,z_,**kwargs)
    ## proposal is red
    ax.scatter(_control_index,_control_position, label = 'model position',c = 'red')

    ax.plot(np.linspace(-R,R,obs_N),np.zeros(obs_N))
    ax.plot(np.linspace(-R,R,obs_N),np.zeros(obs_N),'k|')
    ax.set_ylim(depth-100,10)
    ax.set_xlim(-R,R)
    
def Draw(_control_index,_control_position,_x = None,_z = None,true_position = None,ax = None,R =100,**kwargs):
    if ax is None:
        f,ax = plt.subplots(2,sharex=True, figsize = (7,10))
    if _x is None:
        _x,_z = gp.GaussianProcess(_control_index,_control_position,resolution=10)

    simulated_gravity(_x,_z,R = R,ax = ax[0],**kwargs)
    Draw_inter(_control_index,_control_position,x_ = _x, z_ = _z,R = R,ax = ax[1],z_true =true_position,**kwargs)
#     plt.legend(loc = 'lower right')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(),loc = 'lower right')
    

# prior
mu_prior = True_position_sin
cov_prior = tf.constant([30**2,0.1,0.1,0.1,0.1],dtype = tf.float64)*tf.eye(Number_para,dtype = tf.float64)

x_obv = tf.linspace(Range[0],Range[1],obs_N)

rho = 2000
# likelihood
sig_e = constant64(2*1e-8)
cov=sig_e ** 2.*tf.eye(obs_N,dtype = tf.float64)


mvn_prior = tfd.MultivariateNormalTriL(
    loc=mu_prior,
    scale_tril=tf.linalg.cholesky(cov_prior))

control_index_ = tf.expand_dims(tf.linspace(Range[0], Range[1], Number_para), 1)

gp = GaussianProcess2Dlayer(Range,depth,Number_para,thickness,)
gp.amplitude =tfp.util.TransformedVariable(0.01, tfb.Exp(), dtype=tf.float64, name='amplitude')
gp.length_scale = tfp.util.TransformedVariable(
            100, tfb.Exp(), dtype=tf.float64, name='length_scale')
gp.kernel = psd_kernels.ExponentiatedQuadratic(
    gp.amplitude, gp.length_scale)
gp.observation_noise_variance = 0.

N = 1000

gravity_ = tf.TensorArray(tf.float64, size=N)

z_cord = tf.TensorArray(tf.float64, size=N)
for i in range(N):
    mu=mvn_prior.sample()
    if (mu.numpy() < 0).all():
        _x,_z = gp.GaussianProcess(control_index,mu,resolution=100)
        z_cord = z_cord.write(i,_z)
        gravity_ = gravity_.write(i,model.calculate_gravity(_x,_z))
    else: continue
x_cord = _x.numpy()

x_obv_p = np.expand_dims(x_obv, 1)

gravity = np.squeeze(gravity_.stack())
gravity_mean = np.mean(gravity,0)
gravity_std = np.std(gravity, 0)

G = gravity
Data_mean = np.mean(G, 0)

cov_matrix = np.cov(G.T)
# eigval,eigvec = np.linalg.eig(cov_matrix)

# new_cov = eigvec @ np.diag(eigval + 1e-28)@eigvec.T

## define oversample parameter number
Number_para_os = 30

# prior
mu_prior = -100.*tf.ones([Number_para_os],dtype = tf.float64)
cov_prior = 30.*tf.eye(Number_para_os,dtype = tf.float64)

# likelihood
Data = tf.convert_to_tensor(Data_mean)
# cov = tf.convert_to_tensor(new_cov, dtype=tf.float64)
cov = tf.convert_to_tensor(cov_matrix, dtype=tf.float64)

prior_x = np.expand_dims(np.linspace(Range[0],Range[1],Number_para_os),1)
mu_prior_y = np.expand_dims(mu_prior,1)
cov_prior_sigma = np.expand_dims(np.diag(cov_prior), 1)

control_index_os = tf.linspace(Range[0],Range[1],Number_para_os)

True_position_sin = 20*tf.sin(0.01*control_index_os)+depth

model1 = Gravity_Polygon(obs_N,Range,rho,thickness,Number_para_os)

model1.set_prior(mu_prior,cov_prior,cov)

mu_init = tf.random.normal([Number_para_os],mean = depth, stddev = 20,seed = 1,dtype = tf.float64) # initial parameters

gp = GaussianProcess2Dlayer(Range,depth,Number_para,thickness,)
gp.amplitude =tfp.util.TransformedVariable(0.01, tfb.Exp(), dtype=tf.float64, name='amplitude')
gp.length_scale = tfp.util.TransformedVariable(
            40, tfb.Exp(), dtype=tf.float64, name='length_scale')
gp.kernel = psd_kernels.ExponentiatedQuadratic(
    gp.amplitude, gp.length_scale)
gp.observation_noise_variance = 0.
gp.Number_para = Number_para_os

model1.gp = gp


# start = timeit.default_timer()
# steps_gradient_decent = 100000
# lost = []
# mu = mu_init
# for i in range(steps_gradient_decent):
#     with tf.GradientTape() as t:  
#         t.watch(mu)
#         loss = model1.negative_log_posterior(Data,mu) # negative log posterior
#         lost.append(loss.numpy())
#     dlossdmu = t.gradient(loss,mu)
#     mu = mu-tf.multiply(constant64(1e-4),dlossdmu)
# end=timeit.default_timer()
# print('time for find MAP by SGD: %.2f'%(end-start))
# gp.Number_para=Number_para_os

### ---------
### Find MAP
### ---------

def loss(mu):
    lost =  model1.negative_log_posterior(Data,mu)
    return lost


def loss_minimize():
    lost =  model1.negative_log_posterior(Data,mu)
    return lost
    
# cost = []
# mu = tf.Variable(mu_init)
# opt = tf.keras.optimizers.SGD(learning_rate=0.0001, name='SGD')
# steps = 100000
# start = timeit.default_timer()
# for i in range(steps):
#     opt.minimize(loss_minimize, var_list=[mu])
#     cost.append(loss(mu).numpy())
#     print ('step:',len(cost),'loss',loss(mu).numpy())
# end = timeit.default_timer()
# print('SGD: %.3f' % (end - start))

# result_100000_SGD = loss(mu).numpy()


# ## momentum
# opt_momentum = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.2, name='SGD')
# cost_M = []
# mu = tf.Variable(mu_init)
# start = timeit.default_timer()
# while loss(mu) > result_100000_SGD:
#     opt_momentum.minimize(loss_minimize, var_list=[mu])
#     cost_M.append(loss(mu).numpy())
#     print ('step:',len(cost_M),'loss',loss(mu).numpy())
# end = timeit.default_timer()
# print('SGD with momentum: %.3f' % (end - start))

## Adam

Adam = tf.keras.optimizers.Adam(
    learning_rate=0.1
)
cost_A = []
mu = tf.Variable(mu_init)
start = timeit.default_timer()
for step in range(10000):
# while loss(mu) > -22:
    Adam.minimize(loss_minimize, var_list=[mu])
    cost_A.append(loss(mu).numpy())
    print ('step:',step,'loss',loss(mu).numpy())
end = timeit.default_timer()
print('Adam: %.3f' % (end - start))
MAP = mu

### ---------
### Calculation Hessian
### ---------

def Full_Hessian():
    Hess = tf.TensorArray(tf.float64, size=Number_para_os)
    for i in range(Number_para_os):
        with tf.GradientTape() as t:
            t.watch(MAP)
            with tf.GradientTape(persistent=True) as tt:
                tt.watch(MAP)
                loss = model1.negative_log_posterior(Data,MAP)
            jac = tt.gradient(loss,MAP,unconnected_gradients='zero')[i]
        hess = t.gradient(jac,MAP,unconnected_gradients = 'none')
        Hess = Hess.write(i,hess)
    return Hess.stack()
start = timeit.default_timer()
New_Hessian = Full_Hessian()
end = timeit.default_timer()
print('time for Hessian calculation: %.3f' % (end - start))


### ---------
### MCMC
### ---------

### MCMC RNH

number_sample=10000
number_burnin = 0

num_results = number_sample
burnin = number_burnin


initial_chain_state = [
    depth * tf.ones([Number_para_os], dtype=tf.float64, name="init_t1"),
]

unnormalized_posterior_log_prob = lambda *args: model1.joint_log_post(Data,*args)

def gauss_new_state_fn(scale, dtype):
    gauss = tfd.Normal(loc=dtype(0), scale=dtype(scale))
    def _fn(state_parts, seed):
        next_state_parts = []
        seed_stream  = tfp.util.SeedStream(seed, salt='RandomNormal')
        for sp in state_parts:
            next_state_parts.append(sp + gauss.sample(
            sample_shape=sp.shape, seed=seed_stream()))
        return next_state_parts
    return _fn

dtype = np.float64
start = timeit.default_timer()
samples, kernel_results = tfp.mcmc.sample_chain(
    num_results=num_results,
    current_state=initial_chain_state,
    kernel=tfp.mcmc.RandomWalkMetropolis(
        target_log_prob_fn=unnormalized_posterior_log_prob,
        new_state_fn=gauss_new_state_fn(scale=0.1, dtype=dtype)),
    num_burnin_steps=burnin,
    num_steps_between_results=1,  # Thinning.
    parallel_iterations=1)
end = timeit.default_timer()
print('Random walk time in seconds: %.3f' % (end - start))

samples = tf.stack(samples, axis=-1)
accepted = kernel_results.is_accepted

samples = samples.numpy()
accepted = accepted.numpy()

accept_index = np.where(accepted==True)
accepted_samples_RMH = samples[accept_index]
accepted_samples_RMH = np.squeeze(accepted_samples_RMH)

x_RMH = np.expand_dims(np.linspace(Range[0],Range[1],Number_para_os),1)
y_RMH = np.expand_dims(np.mean(samples,0),1)
std_RMH = np.expand_dims(np.std(samples,0),1)
pdense(x_RMH,y_RMH,std_RMH,M=10000,midpoint =0.2,title = 'Posterior by RMH, steps:{}, accept rate:{}%'.format(number_sample,100*accepted_samples_RMH.shape[0]/number_sample))
plt.ylabel('depth (m)', fontsize=12)
plt.xlabel('position (m)', fontsize =12)
plt.savefig("/Users/zhouji/Documents/Presentations/EGU 2020/Presentation/Input/posterior_RMH.png")
# pd.DataFrame(accepted_samples_RMH).to_csv('./RMH.txt')

samples_RMH = np.squeeze(samples)
plt.figure()
plt.plot(samples_RMH[:, 0], label='point 1')
plt.plot(samples_RMH[:,1],label = 'point 2')
plt.title('Trace plot RMH')
plt.ylabel('depth')
plt.xlabel('iterations')
plt.legend(loc='lower left')
plt.savefig("/Users/zhouji/Documents/Presentations/EGU 2020/Presentation/Input/trace_plot_RMH.png")


### HMC
number_sample=10000
number_burnin = 0

@tf.function
def run_HMC():
    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=number_sample,
        current_state=initial_chain_state,
        kernel=tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=unnormalized_posterior_log_prob,
            step_size = 0.067,
            num_leapfrog_steps = 3),
        num_burnin_steps=number_burnin,
        num_steps_between_results=5,  # Thinning.
        parallel_iterations=1)
    return samples,kernel_results

start = timeit.default_timer()

samples,kernel_results = run_HMC()

end = timeit.default_timer()
print('HMC time in seconds: %.3f' % (end - start))

samples = tf.stack(samples, axis=-1)
accepted = kernel_results.is_accepted

samples = samples.numpy()
accepted = accepted.numpy()

accept_index = np.where(accepted==True)
accepted_samples_HMC = samples[accept_index]
accepted_samples_HMC = np.squeeze(accepted_samples_HMC)
x_HMC = np.expand_dims(np.linspace(Range[0],Range[1],Number_para_os),1)
y_HMC = np.expand_dims(np.mean(samples,0),1)
std_HMC = np.expand_dims(np.std(samples,0),1)

pdense(x_HMC,y_HMC,std_HMC,M=10000,midpoint =0.2,title = 'Posterior by HMC, steps:{}, accept rate:{}%'.format(number_sample,100*accepted_samples_HMC.shape[0]/number_sample))
plt.ylabel('depth (m)', fontsize=12)
plt.xlabel('position (m)', fontsize =12)
plt.savefig("/Users/zhouji/Documents/Presentations/EGU 2020/Presentation/Input/posterior_HMC.png")

plt.figure()
samples_HMC = np.squeeze(samples)
plt.plot(samples_HMC[:, 0], label='point 1')
plt.plot(samples_HMC[:,1],label = 'point 2')
plt.title('Trace plot HMC')
plt.ylabel('depth')
plt.xlabel('iterations')
plt.legend(loc='lower left')
plt.savefig("/Users/zhouji/Documents/Presentations/EGU 2020/Presentation/Input/trace_plot_HMC.png")


### gpCN

beta = constant64(0.5)
number_sample = 10000
number_burnin = 0
h = HessianMCMC(Number_para_os,model1.negative_log_posterior,
                Data,MAP,cov_prior,number_sample,number_burnin,mu_init = MAP,beta = beta)

start = timeit.default_timer()
accepted_samples_gpCN, rejected_samples_gpCN, samples_gpCN = h.run_chain_hessian(New_Hessian)
accepted_samples_gpCN = np.array(accepted_samples_gpCN)
print('Acceptance rate = %0.2f%%' % (100 * accepted_samples_gpCN.shape[0] / number_sample))
end = timeit.default_timer()
print('gpCN time in seconds: %.3f' % (end - start))

samples_gpCN = np.array(samples_gpCN)
x_gpCN = np.expand_dims(np.linspace(Range[0],Range[1],Number_para_os),1)
y_gpCN = np.expand_dims(np.mean(samples_gpCN,0),1)
std_gpCN = np.expand_dims(np.std(samples_gpCN,0),1)

pdense(x_gpCN, y_gpCN, std_gpCN, M=10000, midpoint=0.2, title='Posterior by gpCN, steps:{}, accept rate:{}%'.format(number_sample,100*accepted_samples_gpCN.shape[0]/number_sample))
plt.ylabel('depth (m)', fontsize=12)
plt.xlabel('position (m)',fontsize=12)
plt.savefig("/Users/zhouji/Documents/Presentations/EGU 2020/Presentation/Input/posterior_gpcn.png")


plt.figure()
plt.plot(samples_gpCN[:,0], label='point 1')
plt.plot(samples_gpCN[:,1], label='point 2')
plt.title('Trace plot gpCN')
plt.ylabel('depth')
plt.xlabel('iterations')
plt.legend(loc = 'lower left')
plt.savefig("/Users/zhouji/Documents/Presentations/EGU 2020/Presentation/Input/trace_plot_gpCN.png")

### Plot Autocorrelation


def label(ax, string):
    ax.annotate(string, (1, 1), xytext=(-8, -8), ha='right', va='top',
                size=14, xycoords='axes fraction', textcoords='offset points')

ax1 = autocorrelation_plot(samples_RMH[:, 0])
ax1.set_xlim(0, 300)
label(ax1, 'RMH')
plt.savefig("/Users/zhouji/Documents/Presentations/EGU 2020/Presentation/Input/acrr_RMH.png")


ax2=autocorrelation_plot(samples_HMC[:, 0])
ax2.set_xlim(0, 300)
label(ax2, 'HMC')
plt.savefig("/Users/zhouji/Documents/Presentations/EGU 2020/Presentation/Input/acrr_HMC.png")

ax3=autocorrelation_plot(samples_gpCN[:, 0])
ax3.set_xlim(0, 300)
label(ax3, 'gpCN')
plt.savefig("/Users/zhouji/Documents/Presentations/EGU 2020/Presentation/Input/acrr_gpCN.png")

