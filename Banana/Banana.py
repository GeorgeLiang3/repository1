## define the 2D banana-shape distribution

#import library
from scipy.stats import norm
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import math
import matplotlib.pyplot as plt
import seaborn as sns

class Banana_dist:

    def __init__(self, mu= [0.,0.], cov = [[ 1,  0.],[ 0.,  1]]):

        ## initiate the model
        self.mu = mu
        self.cov = cov
        self.c = 0 # mean of observations
        self.N = 100 # number of observation data
        self.sigma2y = 1 # standard deviation of observation data
        ## generate the observation data
        np.random.seed(121)
        self.y_ = np.random.normal(loc = self.c, scale = self.sigma2y, size = self.N)
        self.D = tf.convert_to_tensor(self.y_,dtype = tf.float32)

    @tf.function
    def joint_log_post(self,theta1,theta2):
        # define random variables prior
        mvn = tfd.MultivariateNormalTriL(
                loc = self.mu,
                scale_tril=tf.linalg.cholesky(self.cov))
        z = tf.stack([theta1, theta2], axis=-1)
        # define likelihood
        y = tfd.Normal(loc = tf.add(theta2,tf.pow(theta1,2.)), scale = self.sigma2y)
        # return the posterior probability
        return(mvn.log_prob(tf.squeeze(z))
            +tf.reduce_sum(y.log_prob(self.D)))


## calculate the posterior density 
    def full_post(self):
        self.x_1, self.y_1 = np.mgrid[-2:2:.03, -2:2:.03]
        pos = np.empty(self.x_1.shape + (2,),dtype = np.float32) 
        pos[:, :, 0] = self.x_1; pos[:, :, 1] = self.y_1
        pos = tf.convert_to_tensor(pos)
        post = np.empty(self.x_1.shape)
        for i in range(np.arange(-2,2,.03).shape[0]):
            for j in range(np.arange(-2,2,.03).shape[0]):
                post[i][j] = self.joint_log_post(pos[i][j][0],pos[i][j][1])
        return post

    def draw_post(self,post = None):
        levels = np.arange(self.joint_log_post(-0.,-0.2), self.joint_log_post(0.0099,0.04), 
                    (self.joint_log_post(0.0099,0.04)- self.joint_log_post(-0.,-0.2))/50)
        if post is None:
            post = self.full_post()
        plt.contourf(self.x_1, self.y_1, post,levels = levels,alpha = 0.7)

        plt.xlim(-1.5,1.5)
        plt.ylim(-2,0.5)
        plt.xlabel("x1",fontsize = 15)
        plt.ylabel("x2",fontsize = 15)




