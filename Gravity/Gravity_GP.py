import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import math as m


tfb = tfp.bijectors
tfd=tfp.distributions
psd_kernels = tfp.math.psd_kernels


class GravityModel(object):
    def __init__(self,mu_prior,cov_prior,cov,fix_point1,fix_point2,Number_para,obs_N):
        self.cov = cov
        self.mu_prior = mu_prior
        self.cov_prior = cov_prior
        self.fix_point1 = fix_point1
        self.fix_point2 = fix_point2
        self.Number_para = Number_para
        self.obs_N = obs_N

    def gravity_calculate(self,x,z):
        self.D = grav(x,z,self.obs_N)


def constant64(i):
    return(tf.constant(i,dtype = tf.float64))

pi = constant64(m.pi)    

def A(x,z,p1,p2):
    numerator = (x[p2]-x[p1])*(x[p1]*z[p2]-x[p2]*z[p1])
    denominator = (x[p2]-x[p1])**2 + (z[p2]-z[p1])**2
    return (numerator/denominator)


def B(x,z,p1,p2):
    '''
    x : array, x coordinate
    z : array, z coordinate
    p1, p2 : int, position
    
    '''
    return ((z[p1]-z[p2])/(x[p2]-x[p1]))


def theta(x,z, p):
    if tf.math.not_equal(x[p], 0) :
        if tf.less(tf.atan(tf.divide(z[p],x[p])),0):
            return(tf.atan(tf.divide(z[p],x[p]))+pi)
        else:
            return(tf.atan(tf.divide(z[p],x[p])))
    elif tf.math.logical_and(tf.math.equal(x[p], 0), tf.math.not_equal(z[p], 0)):
        return(pi/2)
    else: return(0.)


def r(x,z,p):
    '''
    x : array, x coordinate
    z : array, z coordinate
    p : int, position
    
    '''
    return(tf.sqrt(x[p]**2+z[p]**2))


def Z(x,z,p1,p2):
    
    if tf.logical_or(tf.logical_and(tf.equal(x[p1],z[p1]),tf.equal(x[p1],0.)), tf.logical_and(tf.equal(x[p2],z[p2]),tf.equal(x[p2],0.))):
        return(0.)

    elif tf.equal(x[p1], x[p2]):
        return((x[p1]*tf.math.log(r(x,z,p2)/r(x,z,p1))))
    
    else:
    
        theta1 = theta(x,z, p1)
        theta2 = theta(x,z, p2)

        r1 = r(x,z,p1)
        r2 = r(x,z,p2)

        _A = A(x,z,p1,p2)
        _B = B(x,z,p1,p2)

        Z_result = _A*((theta1-theta2)+_B*tf.math.log(r1/r2))
        return(Z_result)


def g(x,z,loc):

    G = constant64(6.67 * 10**(-11)) # gravitational constant  m^3 kg ^-1 s^-2
    rho = constant64(2000.)        # density difference   kg/m^3

    _x = x-loc[0]
    _z = z-loc[1]

    Z_sum = constant64(0.)

    for i in tf.range(_x.shape[0]-1):
        Z_sum = tf.add(Z_sum, Z(_x,_z,i,i+1))

    Z_sum = tf.add(Z_sum, Z(_x,_z,-1,0))

    g = 2*G*rho * Z_sum

    return(g)


def grav(x,z,obs_N):

    x_obv = tf.linspace(constant64(-70.),constant64(70.),obs_N)
    y_obv = tf.zeros(tf.shape(x_obv),dtype = tf.float64)
    obv = tf.stack((x_obv,y_obv),axis = 1)

    gravity = tf.TensorArray(tf.float64, size=obv.shape[0])

    j = tf.constant(0)
    for i in obv:
        gravity=gravity.write(j,-g(x,z,i))
        j = tf.add(j,1)
    return tf.reshape(gravity.stack(),shape = [obs_N])



def joint_log_post(D,mu_prior,cov_prior,cov,fix_point1,fix_point2,Number_para,obs_N,_control_position):
    """
    D: is the observation data
    ps: Positions,Variable(N elements vector)
    """
    # define random variables prior

    mvn_prior = tfd.MultivariateNormalTriL(
            loc = mu_prior,
            scale_tril=tf.linalg.cholesky(cov_prior))
    # define likelihood

    _control_index = tf.linspace(constant64(-70),constant64(70),Number_para)
    __x,__z = GaussianProcess_model(fix_point1,fix_point2,Number_para,_control_index,_control_position)
    
    Gm_ = grav(__x,__z,obs_N)
    
    mvn_likelihood = tfd.MultivariateNormalTriL(
            loc = Gm_,
            scale_tril= tf.linalg.cholesky(cov))
    
    # return the posterior probability
    return (mvn_prior.log_prob(_control_position)
          +mvn_likelihood.log_prob(D))

def negative_log_posterior(D,mu_prior,cov_prior,cov,fix_point1,fix_point2,Number_para,obs_N,_control_position):
    return -joint_log_post(D,mu_prior,cov_prior,cov,fix_point1,fix_point2,Number_para,obs_N,_control_position)

def GaussianProcess_model(fix_point1,fix_point2,Number_para,_control_index,_control_position,visual = False,resolution = None):
    if resolution is None:
        resolution=3

    if thickness is None:
        thickness = constant64(10)
    points = tf.stack([_control_index,_control_position],axis = -1)

    points = tf.concat([tf.concat([fix_point1,points],axis = 0),fix_point2],axis = 0)

    M = 3 # controling the index's density out of the box

    observation_index_points = tf.reshape(points[:,0],[Number_para+2*M,1])
    amplitude = tfp.util.TransformedVariable(
      10, tfb.Exp(), dtype=tf.float64, name='amplitude')
    length_scale = tfp.util.TransformedVariable(
      1000, tfb.Exp(), dtype=tf.float64, name='length_scale')
    kernel = psd_kernels.ExponentiatedQuadratic(amplitude, length_scale)

    observation_noise_variance = tfp.util.TransformedVariable(
        np.exp(-50), tfb.Exp(),dtype = tf.float64, name='observation_noise_variance')

    # We'll use an unconditioned GP to train the kernel parameters.
    gp = tfd.GaussianProcess(
        kernel=kernel,
        index_points=observation_index_points,
        observation_noise_variance=observation_noise_variance)
    
    optimizer = tf.optimizers.Adam(learning_rate=.05, beta_1=.5, beta_2=.99)
    @tf.function
    def optimize():
        with tf.GradientTape() as tape:
            loss = -gp.log_prob(points[:,1])
        grads = tape.gradient(loss, gp.trainable_variables)
        optimizer.apply_gradients(zip(grads, gp.trainable_variables))
        return loss
    # First train the model, then draw and plot posterior samples.
    for _ in range(1000):
        neg_log_likelihood_ = optimize()

    ### discretize the geometry
    if visual == True:
        visual_index = tf.expand_dims(tf.linspace(observation_index_points[0,0],observation_index_points[-1,0],200),axis =1)
    model_index = tf.expand_dims(tf.linspace(observation_index_points[0,0],observation_index_points[-1,0],resolution*Number_para+4),axis =1)

    gprm = tfd.GaussianProcessRegressionModel(
        kernel=kernel,
        index_points=model_index,
        observation_index_points=observation_index_points,
        observations=points[:,1],
        observation_noise_variance=observation_noise_variance)
    
    tf.random.set_seed(1)
    model_position = gprm.sample(1,seed = 1)
    
    if visual == True:
        gprm_visual = tfd.GaussianProcessRegressionModel(
            kernel=kernel,
            index_points=visual_index,
            observation_index_points=observation_index_points,
            observations=points[:,1],
            observation_noise_variance=observation_noise_variance)
        tf.random.set_seed(1)
        visual_position = gprm_visual.sample(1,seed =1)
        plt.figure(figsize = (14,7))
        plt.scatter(_control_index,_control_position,c = 'r')
        plt.plot(tf.transpose(visual_index).numpy()[0],visual_position[0])
        
    model_position_complete = tf.reshape([tf.concat([model_position - thickness,
                                                     tf.reverse(model_position,axis = [-1])],axis = -1)],
                                         shape = [model_position.shape[1]*2,1])

    model_index_complete = tf.concat([model_index,tf.reverse(model_index,axis = [0])],axis = 0)
    
    return model_index_complete,model_position_complete


def grav_calculate(x,z,Number_, x_obs=None,R=70,ax = None,style = None,**args):
    if ax is None:
        _,ax=plt.subplots()
    if x_obs is None:
        x_obs = np.linspace(-R,R,Number_)
    y = np.zeros(np.shape(x_obs))
    obv = np.vstack((x_obs,y)).T
    gravity = []
    for i in obv:
        gravity.append(-g(x,z,i))
    gravity = np.array(gravity)
    if style is None:
        style = '-'
    ax.set_title('gravity response at surface')
    ax.set_ylabel('g_z ($m/s^2$ )')
#     ax.set_ylim(4e-6,10e-6)
    ax.set_xlim(-70,70)
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,2))
    ax.plot(x_obs,gravity,style,**args)
    return gravity


def Draw_inter(fix_point1,fix_point2,obs_N,Number_para,_control_index=None,_control_position=None,True_position = None,x_true = None,z_true = None,R = 70, ax = None):
    if ax is None:
        _,ax = plt.subplots(figsize = (14,7))
    if z_true is None:
        x_true,z_true = GaussianProcess_model(fix_point1,fix_point2,Number_para,_control_index,_control_position)
    ax.scatter(_control_index,_control_position, c = 'red')
    ax.scatter(_control_index,True_position, c = 'black',alpha = 0.6)
    ax.plot(x_true,z_true)
    ax.plot(np.linspace(-R,R,obs_N),np.zeros(obs_N))
    ax.plot(np.linspace(-R,R,obs_N),np.zeros(obs_N),'k|')
    ax.set_ylim(-100,10)
    ax.set_xlim(-R,R)



def Draw(fix_point1,fix_point2,obs_N,Number_para,_control_index,_control_position,Number_,True_position,ax = None):
    if ax is None:
        _,ax = plt.subplots(2,sharex=True, figsize = (7,10))
    _x,_z = GaussianProcess_model(fix_point1,fix_point2,Number_para,_control_index,_control_position)
    grav_calculate(_x,_z,Number_,ax = ax[0])
    Draw_inter(obs_N,Number_para,_control_index,_control_position,True_position,ax = ax[1])




