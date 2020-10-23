# define the 2D banana-shape distribution

#import library
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import matplotlib.colors as colors
from matplotlib.colors import from_levels_and_colors


def constant32(m):
    return tf.convert_to_tensor(m,dtype = tf.float32)

class Gaussian_dist: 

    def __init__(self, mu=[0., 0.], cov=[[1.,  0.], [0.,  1.]]):

        # initiate the model

        # prior
        self.mu = constant32(mu)
        self.cov = constant32(cov)
        self.post = None
        self.range = constant32(2)
        self.c = constant32(tf.reduce_sum(mu))  # mean of observations
        self.N = 30  # number of observation data
        self.sigma2y = constant32(1)  # standard deviation of observation data
        # generate noisy observation data
        np.random.seed(121)
        self.y_ = np.random.normal(loc=self.c, scale=self.sigma2y, size=self.N)
        # self.y_ = np.random.uniform(low = -3, high = 3,size = self.N)
        self.D = tf.convert_to_tensor(self.y_, dtype=tf.float32)

    # @tf.function
    # def distribution(self,theta):
    #      y = tfp.distributions.MultivariateNormalFullCovariance(
    #             loc=[0,0], covariance_matrix=self.cov
    #         )
    #      log_prob = y.log_prob(theta)
    #      return log_prob

    @tf.function
    def joint_log_post(self, theta):
        """Calculate the joint posterior of a given point

        Arguments:
            theta {[tensor:float32]} -- [2xn] tensor! eg: tf.constant([[1.,1.]])

        Returns:
            [tensor] -- value of the posterior
        """
        # define random variables prior

        # D_n = tf.reshape(self.D, [self.D.shape[0], 1])
        # D_n = tf.tile(D_n, [1, theta.shape[0]])
        D_n = self.D

        mvn = tfd.MultivariateNormalTriL(
            loc=self.mu,
            scale_tril=tf.linalg.cholesky(self.cov))

        # define likelihood
        # y = x1-0.7x2
        y = tfd.Normal(loc=(theta[0]-0.7*theta[1]), scale=self.sigma2y)
        # return the posterior probability
        return(mvn.log_prob(theta)
               + tf.reduce_sum(y.log_prob(D_n), axis=0))


# calculate the posterior density

    def full_post(self):
        self.x_extent = [self.mu[0]-self.range,self.mu[0]+self.range]
        self.y_extent = [self.mu[1]-self.range,self.mu[1]+self.range]
        
        self.x_1, self.y_1 = np.mgrid[self.x_extent[0]:self.x_extent[1]:.02, self.y_extent[0]:self.y_extent[1]:.02]
        pos = np.empty(self.x_1.shape + (2,), dtype=np.float32)
        pos[:, :, 0] = constant32(self.x_1)
        pos[:, :, 1] = constant32(self.y_1)
        pos = constant32(pos)
        post = np.empty(self.x_1.shape)
        for i in range(self.x_1.shape[0]):
            for j in range(self.y_1.shape[0]):
                post[i][j] = self.joint_log_post(
                    constant32(pos[i][j]))
        return post

    def draw_post(self,fig = None,ax = None, title=None):


        if self.post is None:
            self.post = self.full_post()
        
        if ax is None:
            fig,ax = plt.subplots(figsize = (10,10))

        Min = constant32([self.mu[0]-0.3, self.mu[1]+0.5])
        # Max = tf.constant([[0.68, 0.48]])

        ## define a log space for better contour plot
        N = 10
        U = 50 # upper value
        L = 1  # lower value
        space = np.logspace(np.log10(U), np.log10(L),N) # create a logspace
        space = space/(U-L)*(self.joint_log_post(Min)-np.max(self.post))
        space = space - space[-1] + np.max(self.post) # map the logspace to the target range, can also use np.interp

        vmax = np.max(self.post)
        vmin = self.joint_log_post(Min)


        f = ax.contourf(self.x_1, self.y_1, self.post,

                    levels=space, alpha=0.7,cmap =  'Blues')


        if title is not None:
            ax.title(title)
        ax.set_xlim(self.x_extent[0], self.x_extent[1])
        ax.set_ylim(self.y_extent[0], self.y_extent[1])
        ax.set_xlabel("$x_1$", fontsize=25)
        ax.set_ylabel("$x_2$", fontsize=25)

        return fig,ax