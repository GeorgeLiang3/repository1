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


class Rosenbrock_dist:

    def __init__(self, mu=[0., 0.], cov=[[1,  0.], [0.,  1]]):

        # initiate the model

        # prior
        self.mu = mu
        self.cov = cov
        self.post = None
        self.c = 0  # mean of observations
        self.N = 100  # number of observation data
        self.sigma2y = 3  # standard deviation of observation data
        # generate the observation data
        np.random.seed(121)
        self.y_ = np.random.normal(loc=self.c, scale=self.sigma2y, size=self.N)
        self.D = tf.convert_to_tensor(self.y_, dtype=tf.float32)

    @tf.function
    def joint_log_post(self, theta):
        """Calculate the joint posterior of a given point

        Arguments:
            theta {[tensor:float32]} -- [2xn] tensor! eg: tf.constant([[1.,1.]])

        Returns:
            [tensor] -- value of the posterior
        """
        # define random variables prior

        D_n = tf.reshape(self.D, [self.D.shape[0], 1])
        # D_n = tf.tile(D_n, theta.shape[0])

        mvn = tfd.MultivariateNormalTriL(
            loc=self.mu,
            scale_tril=tf.linalg.cholesky(self.cov))

        # define likelihood
        ## y = -(0.3 - x1^2 + 80*(x2-x1^2))
        # y = tfd.Normal(loc=tf.negative((0.3-theta[0])**2+80*(theta[1]-*theta[0]**2)**2), scale=self.sigma2y)
        ## y = -(0.3 - x1^2 + 30*(x2-0.1x1^2))
        y = tfd.Normal(loc=tf.negative((0.3-theta[0])**2+30*(theta[1]-0.1*theta[0]**2)**2), scale=self.sigma2y)
        # return the posterior probability
        return(mvn.log_prob(tf.squeeze(theta))
               + tf.reduce_sum(y.log_prob(D_n), axis=0))


# calculate the posterior density

    def full_post(self):
        self.x_1, self.y_1 = np.mgrid[-1:1.3:.01, -1:1:.01]
        pos = np.empty(self.x_1.shape + (2,), dtype=np.float32)
        pos[:, :, 0] = self.x_1
        pos[:, :, 1] = self.y_1
        pos = tf.convert_to_tensor(pos)
        post = np.empty(self.x_1.shape)
        for i in range(np.arange(-1, 1.3, .01).shape[0]):
            for j in range(np.arange(-1, 1, .01).shape[0]):
                post[i][j] = self.joint_log_post(
                    tf.convert_to_tensor(pos[i][j]))
        return post

    def draw_post(self, title=None):


        if self.post is None:
            self.post = self.full_post()

        fig,ax = plt.subplots()

        Min = tf.constant([-0.7, 0.3])
        # Max = tf.constant([[0.68, 0.48]])

        ## define a log space for better contour plot
        N = 10
        U = 50 # upper value
        L = 1  # lower value
        space = np.logspace(np.log10(U), np.log10(L),N) # create a logspace
        space = space/(U-L)*(self.joint_log_post(Min)[0]-np.max(self.post))
        space = space - space[-1] + np.max(self.post) # map the logspace to the target range, can also use np.interp
        # space = np.squeeze(space, axis=1)


        vmax = np.max(self.post)
        vmin = self.joint_log_post(Min)[0]

        # midpoint = 1.005*vmax
        # midp = np.mean(np.c_[space[:-1], space[1:]], axis=1)

        # vals = np.interp(midp, [vmin, midpoint, vmax], [0, 0.4, 1])

        # cmap,norm = from_levels_and_colors(space,plt.cm.PuBu(vals))

        ax.contourf(self.x_1, self.y_1, self.post,
                    # norm=colors.SymLogNorm(linthresh=0.03, linscale=0.03,
                    #                           vmin=self.joint_log_post(Min), vmax=np.max(self.post)), 
                    levels=space, alpha=0.7,cmap =  'PuBu')
                    # levels=space, alpha=0.7, cmap=cmap)
        # plt.colorbar(f)

        if title is not None:
            ax.title(title)
        ax.set_xlim(-1, 1.2)
        ax.set_ylim(-0.6, 1.)
        ax.set_xlabel("x1", fontsize=15)
        ax.set_ylabel("x2", fontsize=15)

        return fig,ax