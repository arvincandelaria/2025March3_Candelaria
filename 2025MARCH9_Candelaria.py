# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 01:16:51 2025

@author: Arvin Candelaria
"""

import scipy.stats as sts
import numpy as np
import matplotlib.pyplot as plt

# Define mu range
mu = np.linspace(1.65, 1.8, num=50)

# Define uniform distribution correctly
uniform_dist = sts.uniform.pdf(mu, loc=1.65, scale=0.2) + 1
uniform_dist = uniform_dist / uniform_dist.sum()

# Define beta distribution
beta_dist = sts.beta.pdf(mu, 2, 5, loc=1.65, scale=0.2)
beta_dist = beta_dist / beta_dist.sum()

# Plot distributions
plt.plot(mu, beta_dist, label='Beta Dist')
plt.plot(mu, uniform_dist, label='Uniform Dist')
plt.xlabel("Value of $\mu$ in meters")
plt.ylabel("Probability density")
plt.legend()
plt.show()

# Define likelihood function
def likelihood_func(datum, mu):
    likelihood_out = sts.norm.pdf(datum, mu, scale=0.1)
    return likelihood_out / likelihood_out.sum()

# Call function with correct arguments
likelihood_out = likelihood_func(1.7, mu)

# Plot likelihood
plt.plot(mu, likelihood_out, label="Likelihood")
plt.title("Likelihood of $\mu$ given observation 1.7m")
plt.ylabel("Probability Density/Likelihood")
plt.xlabel("Value of $\mu$")
plt.legend()
plt.show()

# Compute posterior
unnormalized_posterior = likelihood_out * uniform_dist
plt.plot(mu, unnormalized_posterior)
plt.xlabel("$\mu$ in meters")
plt.ylabel("Unnormalized Posterior")
plt.show()
