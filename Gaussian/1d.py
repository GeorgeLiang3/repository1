import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
import matplotlib.pyplot as plt

dist = tfd.Normal(loc=0., scale=1.)

xs = np.linspace(-40,40,1000)

log_prob = dist.log_prob(xs)

prob = dist.prob(xs)

plt.plot(xs, prob)
plt.show()
plt.plot(xs, log_prob)
plt.show()