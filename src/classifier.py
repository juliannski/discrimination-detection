import numpy as np
from scipy.special import expit as logistic
from src.debug import ipsh
INTERCEPT_NAME = 'free_param'

class LinearClassificationModel:

    def __init__(self, coefficients, intercept = 0.0):
        coefficients = np.array(coefficients)
        assert len(coefficients) > 1
        assert np.isfinite(coefficients).all()
        assert np.isfinite(intercept)
        self._coefficients = np.array(coefficients)
        self._intercept = float(intercept)

    @staticmethod
    def from_dict(**kwargs):
        t = kwargs.get(INTERCEPT_NAME, 0.0)
        w = [kwargs[k] for k in kwargs.keys() if k != INTERCEPT_NAME]
        return LinearClassificationModel(coefficients = w, intercept = t)

    @property
    def coefficients(self):
        return self._coefficients

    @property
    def intercept(self):
        return self._intercept

    def score(self, X):
        scores = X.dot(self._coefficients) + self._intercept
        return scores

    def predict_proba(self, X):
        scores = self.score(X)
        phat = logistic(scores)
        return phat

    def predict(self, X):
        return np.multiply(np.greater_equal(self.predict_proba(X), 0.5), 1)

    def predict_score(self, X):
        return self.score(X), self.predict(X)


class DistributionClassificationModel:

    def __init__(self, p_cond_inputs):
        self.distribution = np.array(p_cond_inputs)

    def predict(self, X):
        p_Y = self.distribution[tuple(X.T)]
        val_y = np.argmax(p_Y)
        return val_y

    def predict_proba(self, X):
        p_Y = self.distribution[tuple(X.T)]
        p_y = np.max(p_Y)
        return p_y


if __name__ == "__main__":

    f = {'proxy_weight': 6,
         'weight_1': 4,
         'weight_2': 4,
         'weight_3': 3,
         'free_param': -8}
    coefficients =[f[k] for k in f.keys() if k != INTERCEPT_NAME]
    intercept = f[INTERCEPT_NAME]
    o = LinearClassificationModel(coefficients = coefficients, intercept = intercept)
    p = o.predict(np.array([[1,1,1,0]]))
    s = o.predict_proba(np.array([[1,1,1,0]]))
    print(p)  # [ 1 ]
    print(s)  # [ 0.997 ]
