from Banana import Banana_dist
import tensorflow as tf
import numpy as np
from RMH import run_chain_RMH
from HMC import run_HMC


class MCMC:
    
    def __init__(self,target_distribution = Banana_dist()):


        self.target_distribution = target_distribution
        self.num_results = 500
        self.brunin = 100
        self.initial_chain_state = [
            -1.7 * tf.ones([], dtype=tf.float32, name="init_t1"),
            -1.5 * tf.ones([], dtype=tf.float32, name="init_t2")
            ]

    @tf.function
    def unnormalized_posterior_log_prob(self,*args):
        return self.target_distribution.joint_log_post(*args)
    
    
    def run_chain(self,method = 'RMH'):
        self.method = method
        if self.method == 'RMH': # Random Walk Matroplis Hasting algorithm
            scale = 0.1
            samples,kernel_results = run_chain_RMH(scale,self.num_results,self.brunin,
                                        self.initial_chain_state,self.unnormalized_posterior_log_prob)
            return samples,kernel_results

        if self.method == 'HMC': # Hamiltonian Monte Carlo algoritem
            samples,kernel_results = run_HMC(self.num_results,self.brunin,
                                        self.initial_chain_state,self.unnormalized_posterior_log_prob)
            return  samples,kernel_results

        
        if self.method == 'HessianMC':
            accepted
        #     return accepted, rejuected
