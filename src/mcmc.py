import random
import numpy as np
from scipy.stats import norm


def p_laplace(x):
    # f(x) with f being proportional to the density of P(x)
    # This is thus the numerator of the bayes rule formula
    return 0.5*np.exp(-abs(np.sum(x*x)))

def p_proxy(x,theta,sigma):
    # g(x | theta) on normal distribution with
    # mean theta and std sigma
    return norm.pdf(x,theta,sigma)

def mcmc_generator(n, size=1, p=p_laplace):
    random_samples = []
    # Initialize the starting point
    theta_i = np.zeros(shape=size)
    while len(random_samples) < n:
        # The new (proposal) sample is taken from a normal distribution
        # with mean theta_i and std 1
        # sample taken from distribution g(proposal | theta_i)
        proposal = np.random.normal(loc=theta_i,scale=1, size=size)
        # Generate a random number v uniformly between 0 and 1
        v = np.random.uniform(0,1)
        # 
        #assert p_proxy(theta_i,proposal,1) == p_proxy(proposal,theta_i,1)
        a = p(proposal) #*p_proxy(theta_i,proposal,1)
        b = p(theta_i) #*p_proxy(proposal,theta_i,1)
        if v <= (a/b):
            # Accept
            random_samples.append(proposal)
            theta_i = proposal
        else:
            # Reject
            random_samples.append(theta_i)
    return random_samples
