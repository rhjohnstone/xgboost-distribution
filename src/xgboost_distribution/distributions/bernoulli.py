"""Normal distribution
"""
import numpy as np
from scipy.stats import bernoulli

from xgboost_distribution.distributions.base import BaseDistribution


def sigmoid(r):
    return 1 / (1 + np.exp(-r))


def inverse_sigmoid(p):
    return np.log(p) - np.log(1-p)


class Bernoulli(BaseDistribution):
    """Bernoulli distribution with log scoring

    Definition:

        f(x) = (p^x) * (1-p)^(1-x)

    We reparameterize:

        r = log(p / (1-p))  |  p = 1 / (1 + exp(-r))

    (Note: reparameterizing ensures that 0 <= p <= 1, regardless
    of what the xgboost booster internally outputs.)

    The gradient is:

        d/dr -log[f(x)] = -(e^r (x - 1) + x)/(e^r + 1)

    The Fisher Information:

        I(p) = 1 / (p * (1-p))

    In reparameterized form, we find I(r) (abusing notation):

        I(r) = e^r / (1 + e^r)^2

    Ref:

        https://www.wolframalpha.com/input/?i2d=true&i=D%5B-ln%5C%2840%29ReplaceAll%5C%2891%29Power%5Bp%2Cx%5DPower%5B%5C%2840%291-p%5C%2841%29%2C1-x%5D%5C%2844%29+p-%3EDivide%5B1%2C1%2BExp%5B-r%5D%5D%5C%2893%29%5C%2841%29%2Cr%5D
        https://www.wolframalpha.com/input/?i=Simplify%5BReplaceAll%5B1%2F%28p*%281-p%29%29%2C+p-%3E1%2F%281%2Bexp%28-r%29%29%5D+*+D%5B1%2F%281%2Bexp%28-r%29%29%2C+r%5D%5E2%5D

    """

    @property
    def params(self):
        return ("p",)

    def gradient_and_hessian(self, y, params, natural_gradient=True):
        """Gradient and diagonal hessian"""

        e_r = np.exp(params)

        grad = -(e_r * (y - 1) + y) / (e_r + 1)

        if natural_gradient:
            fisher_matrix = e_r / (1 + e_r)**2

            grad /= fisher_matrix

            hess = np.ones(len(y))  # we set the hessian constant
        else:
            hess = e_r / (1 + e_r)**2

        return grad, hess

    def loss(self, y, r):
        p = self.predict(r)
        return "BernoulliError", -bernoulli.logpmf(y, p=p).mean()

    def predict(self, params):
        return self.Predictions(p=sigmoid(params))

    def starting_params(self, y):
        return (inverse_sigmoid(np.mean(y)),)
