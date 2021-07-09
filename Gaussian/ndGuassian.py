from re import L
import tensorflow as tf
import os
import numpy as np
import sys
import timeit
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import pandas as pd
tfd = tfp.distributions
import seaborn as sns

from scipy import stats, optimize
import numpy.matlib


def constant32(x):
    return tf.convert_to_tensor(x,dtype=tf.float32)

class BBDdist:

    def __init__(self):

        # initiate the model

        self.DoF = 4 # dimension must be even number
        assert self.DoF % 2 == 0, "Dimension must be even"
        self.nData = 100
        self.mu0 = np.zeros((self.DoF, 1))
        self.std0 = np.ones((self.DoF,1))
        self.var0 = self.std0 ** 2
        self.stdn = 2
        self.varn = self.stdn ** 2
        
        np.random.seed(40)
        self.A = np.ones( (self.DoF, 1) )
        self.thetaTrue = constant32(np.random.normal(size = self.DoF))
        self.data = self.simulateData()

    def getForwardModel(self, thetas):
        # nSamples = tf.size(thetas) // self.DoF
        # thetas = tf.reshape(thetas,(self.DoF, nSamples))
        thetas = tf.transpose(thetas)
        
        tmp = tf.reduce_sum(thetas[::2],axis = 0)+tf.reduce_sum(thetas[1::2]**2,axis = 0)
        return tmp
        
    def simulateData(self):
        noise = np.random.normal( scale = self.stdn, size = (1, self.nData) )
        thetaTrue = tf.expand_dims(self.thetaTrue,axis=0)
        return self.getForwardModel(self.thetaTrue) + noise
    

    def getADMinusLogPrior(self, thetas):
        # thetas = tf.reshape(thetas,[-1,self.DoF])
        thetas = tf.transpose(thetas)

        mvn = tfd.MultivariateNormalDiag(
            loc=tf.tile(tf.expand_dims(constant32(self.mu0.squeeze()),0),[thetas.shape[0],1]),
            scale_diag = tf.tile(tf.expand_dims(constant32(self.std0.squeeze()),0),[thetas.shape[0],1]))
        return -mvn.log_prob(thetas)
    
    def getMinusLogLikelihood(self, thetas, *arg):

        # thetas = tf.reshape(thetas,[-1,self.DoF])
        thetas = tf.transpose(thetas)
        y = tfd.Normal(loc=thetas[:,0]+(thetas[:,1])**2, scale=constant32(self.stdn))
        self.data =constant32([[1]])
        return -tf.squeeze(y.log_prob(constant32(self.data)))

    def getADMinusLogLikelihood(self, thetas):

        # thetas = tf.reshape(thetas,[-1,self.DoF])
        thetas = tf.transpose(thetas)
        
        y = tfd.MultivariateNormalDiag(
            loc=tf.tile(tf.expand_dims(self.getForwardModel(thetas),1),[1,self.data.shape[1]]), 
            scale_diag=tf.tile(tf.expand_dims(tf.expand_dims(constant32(self.stdn),0),0),[thetas.shape[0],self.data.shape[1]]))

        # y = tfd.Normal(loc=tf.tile(tf.expand_dims(self.getForwardModel(thetas),1),[1,self.data.shape[1]]), 
        #             #    scale=tf.tile(tf.expand_dims(tf.expand_dims(constant32(self.stdn),0),0),[thetas.shape[0],self.data.shape[1]]))
        #             scale=self.stdn)

        data = tf.tile(self.data,[tf.shape(thetas)[0],1])
        return -y.log_prob(constant32(data))
    
    
    @tf.function
    def getADMinusLogPosterior(self, thetas):
        """
        Calculate the joint posterior of a given point

        Arguments:
            theta {[tensor:float32]} 

        Returns:
            [tensor] -- value of the posterior
        """
        
        minusLogPrior = self.getADMinusLogPrior(thetas)
        minusLogLikelihood = self.getADMinusLogLikelihood(thetas)

        return minusLogPrior+minusLogLikelihood


    def getADGradientMinusLogPosterior(self, thetas):

        thetas = constant32(thetas)
        with tf.GradientTape() as tt:
            tt.watch(thetas)
            loss = self.getADMinusLogPosterior(thetas)
        grad = tt.gradient(loss, thetas, unconnected_gradients='zero')

        return grad

    def getADHessianMinusLogPosterior(self, thetas): 
        thetas = constant32(thetas)
        Hess = tf.TensorArray(tf.float32, size=2)
        j = 0
        for i in range(2):
            with tf.GradientTape() as t:
                t.watch(thetas)
                with tf.GradientTape() as tt:
                    tt.watch(thetas)
                    loss = self.getADMinusLogPosterior(thetas)
                jac = tt.gradient(loss, thetas, unconnected_gradients='zero')[i]
            hess = t.gradient(jac, thetas, unconnected_gradients='none')
            Hess = Hess.write(j, hess)
            j = j + 1
        hessian = tf.squeeze(Hess.stack())
        return hessian
    
# Banana = BANANA()

# ngrid = 100
# x = np.linspace(-2, 2, ngrid)
# y = np.linspace(-2, 2, ngrid)
# X, Y = np.meshgrid(x,y)
# Z = tf.reshape(Banana.getADMinusLogPosterior( tf.convert_to_tensor(np.vstack( (np.ndarray.flatten(X), np.ndarray.flatten(Y)) ) ,dtype=tf.float32) ) \
#     ,(ngrid, ngrid))

# plt.figure(figsize = (10,10))
# plt.contourf(X, Y, Z, 10)

BBD = BBDdist()

np.random.seed(1)
mu = tf.constant(np.random.normal(size = [2,99]), dtype = tf.float32)
print(mu)


######## Visualization 2D #########

# BBD.data = constant32([[1.3]])

ngrid = 100
x = np.linspace(-2, 2, ngrid)
y = np.linspace(-2, 2, ngrid)
X, Y = np.meshgrid(x,y)
Z_true =  tf.reshape(BBD.getADMinusLogLikelihood( tf.convert_to_tensor(np.vstack( (np.ndarray.flatten(X), np.ndarray.flatten(Y)) ) ,dtype=tf.float32) )\
    ,(ngrid, ngrid))

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))

axes.contourf(X, Y, Z_true, 10)


######## Visualization 4D #########

ngrid = 40
x1 = np.linspace(-2, 2, ngrid)
x2 = np.linspace(-2, 2, ngrid)
x3 = np.linspace(-2, 2, ngrid)
x4 = np.linspace(-2, 2, ngrid)

X1,X2,X3,X4 = np.meshgrid(x1,x2,x3,x4)


Z_true =  tf.reshape(BBD.getADMinusLogLikelihood( constant32(np.vstack(
                                        (np.ndarray.flatten(X1),
                                        np.ndarray.flatten(X2),
                                        np.ndarray.flatten(X3),
                                        np.ndarray.flatten(X4))))) ,
                     (ngrid, ngrid, ngrid, ngrid))

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))

# global_max = np.max(Z_true)
# global_min = np.min(Z_true)

axes[0].contourf(X1[:,:,0,0], X2[:,:,0,0], Z_true[:,:,0,0], 10)
axes[1].contourf(X1[:,:,0,0], X2[:,:,0,0], Z_true[20,:,:,20], 10)
axes[2].contourf(X1[:,:,0,0], X2[:,:,0,0],  Z_true[:,0,0,:], 10)
plt.show(fig)
