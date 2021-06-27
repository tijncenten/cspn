import numpy as np

# source: https://github.com/tijncenten/GeFs/blob/master/gefs/signed.py
class Signed:
    def __init__(self, value, sign=None):
        if sign is None:
            self.value = np.log(np.absolute(value))
            self.sign = np.sign(value).astype(np.float64)
        else:
            self.value = value
            self.sign = sign
        self.size = len(value)
        self.sign[self.value==-np.Inf] = 0.

    def __repr__(self):
        return f'Signed({self.value}, {self.sign})'

    def get(self, i):
        assert i < self.size, "Index out of range."
        return Signed(self.value[i:i+1], self.sign[i:i+1])

    def insert(self, sig, i):
        assert i < self.size, "Index out of range."
        assert sig.size == 1, "Can only insert Signed with size 1."
        self.value[i] = sig.value[0]
        self.sign[i] = sig.sign[0]

    def argsort(self, increasing=True):
        negatives = np.zeros(self.size, dtype=np.bool)
        order_positives = []
        order_negatives = []
        for i in range(self.size):
            if self.sign[i] < 0:
                negatives[i] = 1
                order_negatives.append(i)
            else:
                order_positives.append(i)
        if increasing:
            delta = 1
        else:
            delta = -1
        order_negatives = np.asarray([x for x, y in sorted(zip(order_negatives, -delta*self.value[negatives]), key = lambda x: x[1])], dtype=np.int64)
        order_positives = np.asarray([x for x, y in sorted(zip(order_positives, delta*self.value[~negatives]), key = lambda x: x[1])], dtype=np.int64)
        if increasing:
            return np.concatenate((order_negatives, order_positives))
        else:
            return np.concatenate((order_positives, order_negatives))

    def __mul__(self, other):
        if not isinstance(other, Signed):
            other = Signed(other, sign=np.array([1.]))
        res = Signed(self.value + other.value, self.sign * other.sign)
        if res.value[0] == -np.Inf: res.sign[0] = 0.
        return res

    def concat(self, other):
        if other is None:
            return self
        return Signed(np.concatenate((self.value, other.value)), np.concatenate((self.sign, other.sign)))

    def sum(self):
        # Sum all signed values together using the log-sum-exp trick
        # since all values are in log domain
        if self.size == 1:
            return self
        max_i = np.argmax(self.value)
        if np.isinf(self.value[max_i]):
            return self.get(max_i)
        # Perform LSE with max value trick
        indicators = np.ones(self.size, dtype=np.bool)
        indicators[max_i] = 0
        # Compute the linear domain sum (and apply max value trick)
        r = np.sum(np.exp(self.value[indicators] - self.value[max_i]) * (self.sign[indicators] * self.sign[max_i]))
        # Now, the linear domain sum value needs to be converted to log domain (and complete max value trick)
        # The max value is not included in the sum r, which is equal to np.exp(0) = 1
        r += 1.
        # So to check whether the sum is negative, compare with -1
        if r < 0:
            # The value is negative
            return Signed(np.array([self.value[max_i]]) + np.log(-r), np.array([-self.sign[max_i]]))
        # The value is positive
        return Signed(np.array([self.value[max_i]]) + np.log(r), np.array([self.sign[max_i]]))


def signed_econtaminate(vec, signed_logprs, eps, ismax):
    econt = np.asarray(vec) * (1 - eps)
    room = 1 - np.sum(econt)
    if ismax:
        order = signed_logprs.argsort(False)
    else:
        order = signed_logprs.argsort(True)
    for i in order:
        if room > eps:
            econt[i] = econt[i] + eps
            room -= eps
        else:
            econt[i] = econt[i] + room
            break
    return Signed(econt, None)
