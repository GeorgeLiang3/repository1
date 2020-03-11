import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

## find MAP point
@tf.function()
def gradient_decent(unnormalized_posterior_log_prob,steps = 1000,learning_rate = 0.001):

    mu = tf.constant([-1., -1.])

    for i in tf.range(steps):
        with tf.GradientTape() as t:  
            t.watch(mu)
            theta1 = mu[0]
            theta2 = mu[1]
            loss = tf.negative(unnormalized_posterior_log_prob(theta1,theta2))
            dlossdmu = t.gradient(loss,mu)
            mu = mu - learning_rate*dlossdmu
    return mu



## Hessian
@tf.function
def Full_Hessian():
    Hess = tf.TensorArray(tf.float32, size=2)
    j=0
    for i in range(2):
        with tf.GradientTape() as t:
            t.watch(MAP)
            with tf.GradientTape() as tt:
                tt.watch(MAP)
                loss = -joint_log_post(D,MAP[0],MAP[1])
            jac = tt.gradient(loss,MAP,unconnected_gradients='zero')[i]
        hess = t.gradient(jac,MAP,unconnected_gradients = 'none')
        Hess = Hess.write(j,hess)
        j = j+1
    return Hess.stack()


@tf.function
def matrixcompute(matrix1,matrix2,Cov):
    matrix1 = tf.cast(matrix1,tf.float32)
    matrix2 = tf.cast(matrix2,tf.float32)
    matrix = tf.subtract(matrix1, matrix2)
    matrix = tf.reshape(matrix,[matrix.shape[0],1])
    matrix_T = tf.transpose(matrix)
    Cov_inv = tf.linalg.inv(Cov)
    result = tf.multiply(tf.constant(1/2),tf.matmul(tf.matmul(matrix_T,Cov_inv),matrix))
    return result

@tf.function
def negative_log_post(vars):
    return(tf.negative(joint_log_post(D,vars[0],vars[1])))


@tf.function
def acceptance_gpCN(m_current , m_proposed):
    delta_current = tf.add(negative_log_post(m_current),matrixcompute(m_current,MAP,C_post))
    delta_proposed = tf.add(negative_log_post(m_proposed),matrixcompute(m_proposed,MAP,C_post))

    ## calculate accept ratio if exp()<1
    accept_ratio = tf.exp(tf.subtract(delta_current,delta_proposed))
    acceptsample = tfd.Sample(
    tfd.Uniform(0., 1.),
    sample_shape=[1,1])
    sample = acceptsample.sample()
    
    if(accept_ratio > sample):
        return True
    else:
        return False



@tf.function
def draw_proposal(m_current):
    
    beta = tf.constant(0.25)
    _term1 = MAP
    
    ## sqrt term
    tem_1 = tf.convert_to_tensor(tf.sqrt(1-beta**2),dtype = tf.float32)
    ## sqrt(1-beta^2)()
    _term2 = tf.multiply(tem_1,(tf.subtract(m_current,MAP)))
    
    Xi = tfd.MultivariateNormalFullCovariance(
            loc = 0,
            covariance_matrix= C_post)

    Xi_s = tfd.Sample(Xi)
    _term3 = tf.multiply(beta,Xi_s.sample())
    
    m_proposed = tf.add(MAP,tf.add(_term2,_term3))
    
    return m_proposed


def run_chain():
    MAP = gradient_decent()
    New_Hessian = Full_Hessian()
    burn_in = 100
    steps = number_of_steps
    k = 0
    accepted = []
    rejected = []

    m_current = mu_init  # init m
    
    
    for k in range(steps+burn_in):

        m_proposed = draw_proposal(m_current)

        if acceptance_gpCN(m_current,m_proposed):
            m_current = m_proposed
            if k > burn_in:
                accepted.append(m_proposed.numpy())
        else:
            m_current = m_current
            rejected.append(m_proposed.numpy())
    
    return accepted,rejected










cov= [[1.,0.],[0.,1.]]
cov = tf.convert_to_tensor(cov,dtype = tf.float32)
tf.linalg.inv(cov)

Sum = 0
Sum = tf.add(New_Hessian,tf.linalg.inv(cov))
C_post = tf.linalg.inv(Sum)
C_post 