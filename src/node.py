import numpy as np
from scipy.stats import *
from scipy.special import logsumexp
import matplotlib.pyplot as plt

from functools import reduce
import operator

import mcmc

class Node:

    def __init__(self, children=None, *args, **kwargs):
        if children is None:
            children = []
        self.parents = []
        self.children = children
        if isinstance(self, Leaf):
            self.scopes = set([self.scope])
        else:
            self.scopes = set([sc for c in self.children for sc in c.scopes])
        self.parents = []
        [c.add_parent(self) for c in self.children]
        self._ll = np.nan
        self._grad = np.nan
        self.id = None

    def add_parent(self, parent):
        self.parents.append(parent)

    def add_children(self, children):
        # Add self as the parent of the new children
        [c.add_parent(self) for c in children]
        self.children += children
        self.scopes.update([sc for c in children for sc in c.scopes])

    def set_index(self, id):
        self.id = id

    def __str__(self):
        children_str = [self.__child_str__(str(c), i) for i, c in enumerate(self.children)]
        children_str = ['\n'.join([f'|   {p}' for p in c.split('\n')]) for c in children_str]
        out = self.__node_str__(f'{type(self).__name__}') + ':'
        if len(self.children) > 0:
            out += '\n'
        out += '\n'.join(children_str)
        return out

    def __node_str__(self, name):
        if self.id != None:
            return f'{name}({self.id})'
        return name

    def __child_str__(self, name, i):
        return name

    def __repr__(self):
        if self.id != None:
            return f'{type(self).__name__}({self.id})'
        return type(self).__name__

    def value(self, evidence, ll=False):
        v = self._value(evidence, ll=ll)
        if ll:
            self._ll = v
        return v

    def _value(self, evidence, ll=False):
        raise ValueError('Not implemented yet')

    def sample(self, param, data=None, rand_gen=None):
        raise ValueError('Not implemented yet')

    def mpe(self, param, data=None, rand_gen=None):
        raise ValueError('Not implemented yet')

    def gradient(self, param):
        raise ValueError('Not implemented yet')

    def sgd(self, lr=0.05, **kwargs):
        raise ValueError('Not implemented yet')


class Sum(Node):
    def __init__(self, weights=None, *args, **kwargs):
        if weights is None:
            # TODO: Initialize weights
            weights = []
        super().__init__(*args, **kwargs)
        if len(weights) != len(kwargs.get('children', [])):
            raise ValueError(f'weights and children lenghts are not equal {len(weights)} != {len(kwargs.get("children", []))}')
        self.weights = weights
        self.initialized = len(weights) > 0

    def add_children(self, children):
        super().add_children(children)
        self.weights += [0 for c in children]

    def __child_str__(self, name, i):
        return f'{self.weights[i]} {name}'

    def _value(self, evidence, ll=False):
        child_values = [c.value(evidence, ll=ll) for c in self.children]
        if ll:
            return logsumexp(child_values, b=self.weights) # np.log(np.sum(w * np.exp(a)))
        return sum([c * w for c, w in zip(child_values, self.weights)]) # np.dot(c, w)

    def sample(self, param, data=None, rand_gen=None):
        if param is None:
            return None
        param = np.concatenate(param)

        size = (len(param), len(self.children))
        w_children_log_probs = np.zeros(size)
        for i, c in enumerate(self.children):
            w_children_log_probs[:,i] = c._ll + np.log(self.weights[i])

        z_gumbels = rand_gen.gumbel(loc=0, scale=1, size=size)
        g_children_log_probs = w_children_log_probs + z_gumbels
        rand_child_branches = np.argmax(g_children_log_probs, axis=1)

        results = {}
        for i, c in enumerate(self.children):
            results[c] = param[rand_child_branches == i]

        return results

    def mpe(self, param, data=None, rand_gen=None):
        if param is None:
            return None
        param = np.concatenate(param)

        size = (len(param), len(self.children))
        w_children_log_probs = np.zeros(size)
        for i, c in enumerate(self.children):
            w_children_log_probs[:,i] = c._ll + np.log(self.weights[i])

        max_child_branches = np.argmax(w_children_log_probs, axis=1)

        results = {}
        for i, c in enumerate(self.children):
            results[c] = param[max_child_branches == i]

        return results

    def gradient(self, param):
        # Store the node gradient as the logarithmic version
        self._grad = logsumexp(param)

        messages_to_children = {}
        wlog = np.log(self.weights)

        for i, c in enumerate(self.children):
            # Compute the gradient from this node to the child node c
            # This is done by multiplying the gradient with the weight
            # Or in the log domain, by adding the values together
            child_gradient = self._grad + wlog[i]
            if np.isinf(child_gradient):
                child_gradient = np.finfo(float).min
            messages_to_children[c] = child_gradient
        
        return messages_to_children

    def sgd(self, lr=0.05, **kwargs):
        self.weights = [w + lr * np.exp(self._grad + c._ll) for w, c in zip(self.weights, self.children)]
        self.weights = [w / sum(self.weights) for w in self.weights]


class Product(Node):
    
    def _value(self, evidence, ll=False):
        if ll:
            return sum([c.value(evidence, ll=ll) for c in self.children])
        return reduce(operator.mul, [c.value(evidence, ll=ll) for c in self.children], 1)

    def sample(self, param, data=None, rand_gen=None):
        if param is None:
            return None
        param = np.concatenate(param)

        results = {}
        for c in self.children:
            results[c] = param

        return results

    def mpe(self, param, data=None, rand_gen=None):
        if param is None:
            return None
        param = np.concatenate(param)

        results = {}
        for c in self.children:
            results[c] = param

        return results

    def gradient(self, param):
        # Store the node gradient as the logarithmic version
        self._grad = logsumexp(param)

        messages_to_children = {}

        for i, c in enumerate(self.children):
            # Compute the gradient from this node to the child node c
            # This is equal to the node gradient multiplied by the child values
            # Except for the value of child c.
            # g * prod(S_i(x) for i in self.children)
            # In the log domain, this can be computed as follows
            # self._ll is the product (logsum) of all child likelihood values
            # but we also want to ignore the ll of node c
            child_gradient = self._grad + self._ll - c._ll
            if np.isinf(child_gradient):
                child_gradient = np.finfo(float).min
            messages_to_children[c] = child_gradient
        
        return messages_to_children

    def sgd(self, lr=0.05, **kwargs):
        pass


# WIP: Non-decomposable product node
class NDProduct(Node):

    def eval(self, x):
        return reduce(operator.mul, [c.value(x, ll=False) for c in self.children], 1)

    def _value(self, evidence, ll=False):
        random_samples = mcmc.mcmc_generator(10000, size=len(self.scopes), p=self.eval)
        # print(random_samples)
        print(np.mean(random_samples, axis=0))

        hists = np.swapaxes(random_samples, 0, 1)

        def sc0(x):
            return norm.pdf(x, loc=5, scale=1) * 0.3 + norm.pdf(x, loc=9, scale=2) * 0.7

        def sc(x):
            return sc0(x) * norm.pdf(x, loc=7.5, scale=2) / 0.1144

        plt.title('Metropolis algorithm (N=10000)')
        plt.hist(hists[0], bins=30, histtype='bar', ec='black', density=True)
        x = np.linspace(0, 15, 1000)
        plt.plot(x, sc(x))
        plt.grid(True)
        plt.savefig('sc0.png')
        plt.close()

        # plt.title('Scope 1')
        # plt.hist(hists[1], bins=30, histtype='bar', ec='black')
        # plt.grid(True)
        # plt.savefig('sc1.png')
        # plt.close()

        print('done')
        return 0


class Leaf(Node):
    def __init__(self, scope=None, *args, **kwargs):
        if scope == None:
            raise ValueError('Scope needs to be an integer')
        self.scope = scope
        super().__init__(*args, **kwargs)

    def __str__(self):
        if self.id != None:
            return f'{type(self).__name__}({self.id}) scope={self.scope} {self.properties()}'
        return f'{type(self).__name__} scope={self.scope} {self.properties()}'

    def properties(self):
        return ''

    def gradient(self, param):
        # Store the node gradient as the logarithmic version
        self._grad = logsumexp(param)

    def sgd(self, lr=0.05, **kwargs):
        pass


class Bernoulli(Leaf):
    # src: https://en.wikipedia.org/wiki/Bernoulli_distribution

    def __init__(self, p=0.5, scope=None):
        super().__init__(scope=scope)
        self.p = p
        assert 0 <= p and p <= 1

    def properties(self):
        return f'p={self.p}'

    def _value(self, evidence, ll=False):
        xi = evidence[self.scope]
        # If the evidence is not present (np.nan), return 1
        if np.isnan(xi):
            v = 1
            if ll:
                v = np.log(v)
            return v
        
        assert xi in [0, 1]
        # Else, return the probability of this evidence occuring
        v = self.p if xi == 1 else 1 - self.p
        if ll:
            if v == 0:
                v = np.finfo(float).min
            else:
                v = np.log(v)
        return v

    def sample(self, param, data=None, rand_gen=None):
        if param is None:
            return None
        param = np.concatenate(param)
        if len(param) == 0:
            return None

        data_nans = np.isnan(data[self.scope])

        n_samples = np.sum(data_nans)
        if n_samples == 0:
            return None
        
        data[self.scope] = bernoulli.rvs(p=self.p, size=1, random_state=rand_gen)[0]

    def mpe(self, param, data=None, rand_gen=None):
        if param is None:
            return None
        param = np.concatenate(param)
        if len(param) == 0:
            return None

        data_nans = np.isnan(data[self.scope])

        n_samples = np.sum(data_nans)
        if n_samples == 0:
            return None
        
        data[self.scope] = 1 if self.p > 0.5 else 0

    def sgd(self, lr=0.05, data=None):
        return
        # if data is None:
        #     return
        # v = data[self.scope]
        # # print(self._grad, self._ll)
        # dp = (self.p - v) / ((1 - self.p) * self.p)
        # # dp = 1/self.p if v == 1 else 0 - 1/(1-self.p) if v == 0 else 0
        # self.p = self.p - lr * np.exp(self._grad + dp)
        # self.p = max(0.00000000000001, min(0.9999999999, self.p))


class Categorical(Leaf):
    # src: https://en.wikipedia.org/wiki/Categorical_distribution

    def __init__(self, p, scope=None):
        super().__init__(scope=scope)
        self.p = p
        self.k = len(p)
        assert all([pi > 0 for pi in p])
        assert sum(p) == 1
        assert k > 0

    def properties(self):
        return f'p={self.p}'

    def sample(self, param, data=None, rand_gen=None):
        raise ValueError('Not implemented yet')


class Gaussian(Leaf):


    def __init__(self, mean=0, stdev=1, scope=None):
        super().__init__(scope=scope)
        self.mean = mean
        self.stdev = stdev
        assert stdev > 0

    def properties(self):
        return f'mean={self.mean} stdev={self.stdev}'

    def _value(self, evidence, ll=False):
        x = evidence[self.scope]
        if np.isnan(x):
            v = 1
            if ll:
                v = np.log(v)
            return v
        
        if ll:
            return norm.logpdf(x, self.mean, self.stdev)
        return norm.pdf(x, self.mean, self.stdev)

    def sample(self, param, data=None, rand_gen=None):
        if param is None:
            return None
        param = np.concatenate(param)
        if len(param) == 0:
            return None

        data_nans = np.isnan(data[self.scope])

        n_samples = np.sum(data_nans)
        if n_samples == 0:
            return None
        
        data[self.scope] = norm.rvs(loc=self.mean, scale=self.stdev, size=1, random_state=rand_gen)[0]

    def mpe(self, param, data=None, rand_gen=None):
        if param is None:
            return None
        param = np.concatenate(param)
        if len(param) == 0:
            return None

        data_nans = np.isnan(data[self.scope])

        n_samples = np.sum(data_nans)
        if n_samples == 0:
            return None
        
        data[self.scope] = self.mean

    def sgd(self, lr=0.05, data=None):
        if data is None:
            return
        v = data[self.scope]
