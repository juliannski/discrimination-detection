import numpy as np
import pandas as pd
from src.classifier import INTERCEPT_NAME, LinearClassificationModel, DistributionClassificationModel


class FinalBooleanVariable(object):
    """
    Class for a boolean terminal Variable inside the RobotCatalog. It has a string name, classification model that is
    used to determine its value (either through logistic or linear regression), and the ordering of variables for the
    model.
    """

    def __init__(self, name, vars_ordering, model_parameters, vars_vals_dict = None):
        """
        Enable sampling the value of a terminal variable in a RobotCatalog.

        Use the causal variables with assigned values to compute a score being their linear combination.

        Use the score in a logistic function, and sample a binary value or use a signum function and assign a value.

        :param name: text name of the variable
        :param vars_ordering: variable names as used in the linear function for computing the score
        :param model_parameters: coefficients for the linear/logistic regression
        :param vars_vals_dict: dictionary with values of the causal variables
        """
        assert isinstance(model_parameters, dict) or isinstance(model_parameters, np.ndarray)
        assert INTERCEPT_NAME in model_parameters
        self.name = name
        self.variable_vals = None
        if vars_vals_dict:
            self.variable_vals = [vars_vals_dict[var] for var in vars_ordering]
        self.vars_ordering = vars_ordering
        if isinstance(model_parameters, np.ndarray):
            self.model = DistributionClassificationModel(p_cond_inputs=model_parameters)
        else:
            self.model = LinearClassificationModel.from_dict(**model_parameters)

    def sample(self, prior_values):
        """
        Sample a value for the Variable

        :return: sampled value
        """
        prior_values = {k.name if isinstance(k, Variable) else k: v for k, v in prior_values.items()}
        self.variable_vals = [prior_values[var] for var in self.vars_ordering]
        phat = self.model.predict_proba(np.array([self.variable_vals]))
        yhat = self.model.predict(np.array([self.variable_vals]))
        return yhat[0], phat[0]

    def recompute_prob(self, var_val_dict):
        """
        Get probability for the predicted outcome

        :param var_val_dict: dict with values for all the conditional variables keyed with their names
        :return prob:
        """
        _, prob = self.sample(var_val_dict)
        return prob

    def __repr__(self):
        return '<{}>'.format(self.name)


class Variable(object):
    """
    Class for a discrete random variable in a RobotCatalog. It has a string name, a set of values it attains and a
    conditional probability.

    The latter is represented as a nested dictionary with keys being string representations of conditional variables,
    and the last value being a list modeling the prob distribution.
    """

    def __init__(self, name, values, probabilities):
        self.name = name
        self.values = values
        self.conditional_prob = probabilities
        self.var_val_dict = None
        self.proxy_of = []
        self.is_source = isinstance(probabilities, list)

    def __repr__(self):
        return '<{}>'.format(self.name)

    def resolve(self, prior_values):
        """
        :param prior_values:
        :return:
        """
        distribution = dict(self.conditional_prob)
        while not isinstance(distribution, list):
            for variable, value in prior_values.items():
                condition = f'{variable.name}_{value}'
                if condition in distribution.keys():
                    distribution = distribution[condition]
                    break
        assert np.isclose(sum(distribution), 1), "Probabilities do not sum to 1"
        return distribution

    def sample(self, var_val_dict):
        """
        Go through a probability dictionary by applying variable-value pairs that could condition the prob distribution
        of Variable. Once the condition results in a prob distribution, not another dict that encodes conditionable prob
        distribution, sample a value.

        :param var_val_dict: Dictionary where keys are variables, and [ any ] values are values attained by those
                             variables
        :return to_return: Value of the Variable that was drawn
        """
        if self.is_source:
            cond_prob = self.conditional_prob
        else:
            cond_prob = self.resolve(var_val_dict)
        self.var_val_dict = var_val_dict
        idx = np.random.choice(np.arange(len(cond_prob)), 1, cond_prob)
        val = self.values[idx[0]]
        prob = cond_prob[idx[0]]
        return val, prob

    def recompute_prob(self, var_val_dict):
        """
        Recompute the probability for the current value of the Variable under new values of the conditioning variables.

        :param var_val_dict: Dictionary where keys are variables, and [ any ] values are values attained by those
                             variables
        :return p: new probability
        :return probabilities: probability distribution for the values in general
        """
        cond_prob = self.conditional_prob
        self.var_val_dict = var_val_dict
        i = 0
        evaluated_vars = list(var_val_dict.items())
        while type(cond_prob) is not list:
            var, val = evaluated_vars[i]
            condition = f'{var}_{val}'
            i += 1
            if condition in cond_prob.keys():
                condition = f'{var}_{val}'
                cond_prob = cond_prob[condition]
                i = 0
        curr_val = var_val_dict[self.name]
        ind = self.values.index(curr_val)
        p = cond_prob[ind]
        return p

    @property
    def cond_prob(self):
        return self.conditional_prob

    @cond_prob.setter
    def cond_prob(self, value):
        assert isinstance(value, dict) or (isinstance(value, list) and sum(value)) == 1, \
            "New conditional probability needs to be a dictionary keyed by conditional variable names_values or a list"
        self.conditional_prob = value


class FairnessDAG(object):

    def __init__(self, features, outcome, proxy, protected):

        assert isinstance(protected, Variable)
        assert isinstance(proxy, Variable)

        # initialize nodes
        nodes = {
            protected.name: protected,
            proxy.name: proxy,
            }

        for k, v in features.items():
            if k != proxy.name:
                nodes[k] = v

        nodes[outcome.name] = outcome
        self.nodes = nodes

        # pull names
        self._feature_names = [proxy.name] + [k for k in features.keys() if k != proxy.name]
        self._outcome_name = outcome.name
        self._protected_name = protected.name
        self._proxy_name = proxy.name

        # initialize dag
        self.edges = {k: [] for k in self.nodes.keys()}

        # sampling order
        self._sampling_order = [self._proxy_name] + self._feature_names
        self._attribution_order = [[self.protected], [self.proxy], self.features]

        # initialize distinct values
        variables = [self.nodes[k] for k in self.measured_names]
        variable_space = {var.name: var.values for var in variables}
        index = pd.MultiIndex.from_product(variable_space.values(), names=variable_space.keys())
        self._values = pd.DataFrame(index=index).reset_index()

        ## add predictions
        self._values[self._outcome_name] = self._values.apply(lambda x: self.outcome.sample(x)[0], axis=1)

        # add their likelihoods
        self._get_likelihoods()


    def add_edges_from_coefficients(self, coefficients):
        assert isinstance(coefficients, dict)
        assert set(self._feature_names).issubset(coefficients.keys())
        new_edges = {k: [] for k in self.nodes.keys()}
        for name in self._feature_names:
            w = coefficients[name]
            if np.not_equal(w, 0.0):
                new_edges[name] = [self.outcome]
        self.edges = new_edges

    @property
    def outcome(self):
        return self.nodes[self._outcome_name]

    @property
    def protected(self):
        return self.nodes[self._protected_name]

    @property
    def proxy(self):
        return self.nodes[self._proxy_name]

    @property
    def features(self):
        return [self.nodes[k] for k in self._feature_names]

    @property
    def node_names(self):
        return list(self.nodes.keys())

    @property
    def feature_names(self):
        return self._feature_names

    @property
    def measured_names(self):
        return [k for k in self.node_names if k is not self._outcome_name]

    @property
    def values(self):
        return self._values

    @property
    def n_distinct(self):
        return len(self._values)

    def _recompute_likelihood(self, point):
        """
        Recompute likelihood of a point given it as a dictionary/row in a dataframe

        :param point:
        :return likelihood:
        """
        assert (self._values == point).all(1).any(), "Provided point does not come from the DAG"
        relevant_elements = {k: v for k, v in point.items() if k in self._values.columns}
        probs = [self.nodes[el].recompute_prob(relevant_elements) for el in relevant_elements]
        likelihood = np.exp(np.log(probs).sum())
        return likelihood

    def _get_likelihoods(self):
        """
        Get likelihoods for each point in self._values
        """
        self._values['likelihood'] = self._values.apply(lambda x: self._recompute_likelihood(x), axis=1)

    def sample(self):
        """
        Sample one point from the DAG with its likelihood

        :return: dict with the sampled point and its likelihood
        """
        values = {}
        collected = {}
        probabilities = []
        # Sample values of the causal variables
        for nodelist in self._attribution_order:
            for node in nodelist:
                sample, probability = node.sample(collected)
                collected[node] = sample
                values[node.name] = sample
                probabilities.append(probability)

        # sample the outcome
        sample, probability = self.outcome.sample(collected)
        values[self._outcome_name] = sample
        probabilities.append(probability)
        values['likelihood'] = np.exp(np.log(probabilities).sum())
        return values

    def find_ids(self, elem):
        """
        Return all ids of rows in the FairnessDAG.values dataframe that share the same values for columns in elem with
        elem

        :param elem: dict with column names and values found in FairnessDAG.values
        :return: list with all the ids of rows in FairnessDAG.values that satisfy the property above
        """
        relevant_elem = {k: v for k, v in elem.items() if k in self.measured_names}
        key_names, values = list(relevant_elem.keys()), list(relevant_elem.values())
        fitting_rows = self.values[(self.values[key_names] == values).all(1)]
        return list(fitting_rows.index)