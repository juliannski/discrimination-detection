import copy
import warnings
import random
import numpy as np
import pandas as pd
from src.utils import power_set
from src.robots import Robot


class RobotCatalog(object):
    """
    RobotCatalog is defined through a dictionary that defines which variables have an outgoing arrow to which other variables. It
    allows sampling example points from itself based on the conditional distributions implemented by the variables.

    The samples are stored in a dataframe which conceptually can be thought of as the robot catalog for the experiment.

    The class also implements likelihood computation and set plausibility computation.

    Plausibility of a sequence of points is its likelihood divide by the likelihood of the sequence containing the most
    probable examples.
    """

    def __init__(self, dag, model, random_state=np.random.RandomState(seed=1111)):
        """
        Enable sampling from the RobotCatalog using its name in form of a dictionary.

        Create a list of variable lists so that each next list of variables contains variables that depend on the
        previously sampled ones.

        :param dag: FairnessDAG
        :param model: LinearClassificationModel instance used to sample yhat
        :param random_state: RandomState
        """
        self.model = model
        self.df = None
        self.dag = dag
        self.rng = random_state

    #### built-ins ####
    def __str__(self):
        return self.df.loc[:, self.df.columns != 'CEs'].to_string(index=False)

    def __repr__(self):
        return str(self)

    def generate(self, n, with_replacement=True, prediction_type='label'):
        """
        Sample multiple points from the RobotCatalog and save them in a DataFrame.

        :param n: Number of points to sample from the RobotCatalog
        :param with_replacement: Whether to sample with or without replacement
        :param prediction_type: Whether the model returns a prob distribution or a label
        :return samples: DataFrame containing the sampled points.
        """
        variable_names = self.dag.node_names
        feature_names = self.dag._feature_names

        if n > self.dag.n_distinct and with_replacement is False:
            warnings.warn(f"Can only produce {self.dag.n_distinct} points, not {n}.")
            n = self.dag.n_distinct

        df = self.dag.values.sample(n=n, replace=with_replacement,
                                    weights=self.dag.values['likelihood'], random_state=self.rng)
        ids = list(df.index)
        df['id'] = ids
        # assign "mid" = model_id based on the unique feature combinations, i.e. 0 0 0 0 is 0, 0 0 0 1 is 1, etc.
        df['mid'] = df.apply(lambda row: int(sum([2 ** i * row[feat] for i, feat in enumerate(feature_names)])), axis=1)
        df = df.reindex(columns=['id', 'mid'] + variable_names + ['likelihood'])
        X = df[feature_names].values.astype(float)
        df['yhat_score'] = self.model.score(X)
        df['phat'] = self.model.predict_proba(X)
        df['yhat'] = self.model.predict(X)
        self._assign_deltas(df, prediction_type)
        self.df = df.sort_values(by='id')
        self.df['color_scheme'] = 1
        self.unique_df = copy.deepcopy(self.df)

        return df

    def adapt(self, to_size):
        """
        Extend the catalog by re-adding points from the original catalog up to the desired size.
        :param to_size:
        :return: DataFrame containing the (extended) sampled points.
        """
        current_size = self.df.shape[0]

        if to_size < current_size:
            return self.df[self.df.imdex.isin(range(to_size))]

        color_scheme = 2
        while to_size > current_size:
            n = to_size - current_size
            new_df = self.df[self.df.index.isin(range(n))]
            new_df['color_scheme'] = color_scheme
            self.df = pd.concat([self.df, new_df], ignore_index=True)
            current_size = self.df.shape[0]
            color_scheme += 1
        ids = list(self.df.index)
        self.df['id'] = ids

        return self.df

    def recompute_likelihood(self, subset=False):
        """
        Compute likelihood of each point in a given subset using the distribution of each variable.

        :param subset: column name used for filtering the dataframe for 1s
        """
        filtered_data = self.df
        if subset:
            filtered_data = self.df[self.df[subset] == 1]

        new_liks = []
        for i, point in filtered_data.iterrows():
            whole_p = dict(point)
            rel_p = {k: v for k, v in whole_p.items() if k in self.dag.measured_names}
            point_lik = 1
            for part in self.dag._attribution_order:
                for var in part:
                    p = var.recompute_prob(rel_p)
                    point_lik *= p
            new_liks.append(point_lik)
            self.df.at[i, 'likelihood'] = point_lik
        if subset is not False:
            self.df.loc[self.df[subset] == 1, 'likelihood'] = new_liks
        else:
            self.df.loc[:, 'likelihood'] = new_liks

    def compute_seq_likelihood(self, subset):
        """
        Compute the likelihood of a chosen subset by multiplying likelihoods of each point in the set.

        :param subset: which subset of points (define by 1 in the columns 'subset') to compute the likelihood of
        :return: computed likelihood
        """
        chosen_points = self.df.loc[self.df[subset] == 1]
        seq_lik = np.prod(list(chosen_points['likelihood']))
        return seq_lik

    def _assign_deltas(self, df, prediction_type='label'):
        """
        Compute delta counterfactual fairness for all the points and add a column to the dataframe
        """
        df['delta'] = df.apply(lambda row:
                               self._compute_delta({k: v for k, v in dict(row).items() if k in self.dag.node_names},
                                                   self.dag.proxy, prediction_type), axis=1)
        return df

    def _compute_delta(self, point, proxy, prediction_type='label'):
        """
        Compute delta-fairness for a point given a proxy with respect to which that fairness is to be evaluated.

        Delta is equal to the biggest difference in the distribution under different protected attribute values

        P(Yhat_{A<-a} = pred | X=x, Proxy=val, A=a) = 1
        P(Yhat_{A<-b} = pred | X=x, Proxy=val, A=a) = SUM_val P(Yhat_{A<-a} = pred | X=x, Proxy=val, A=b) * P(Proxy = val | A = b)

        :param point: point for which delta is to be found
        :param proxy: name of the proxy variable
        :return: float/integer defining the delta counterfactual fairness
        """
        # generating distributions for different values of the protected attributes that affect the proxy
        # supports one protected attribute now
        A = self.dag.protected
        cond_probs_across_protected_values = {v:proxy.resolve(prior_values = {A:v}) for v in A.values}

        # current value of the protected att
        curr_a = point[A.name]

        # looking through the distributions across all pairs of protected attribute values where 1 element is the
        # current value
        all_var_combs = power_set(list(cond_probs_across_protected_values.keys()))
        all_var_comparisons = [ac for ac in all_var_combs if len(ac) == 2]
        all_rel_var_comparisons = [ac for ac in all_var_comparisons if curr_a in ac]
        curr_val_indices = [ac.index(curr_a) for ac in all_rel_var_comparisons]
        # always set the first distribution for A=a
        all_rel_cond_probs_comparisons = [(cond_probs_across_protected_values[ac[curr_val_indices[i]]],
                                           cond_probs_across_protected_values[ac[1-curr_val_indices[i]]]) for i, ac in enumerate(all_rel_var_comparisons)]

        # we use the model prob predictions or actual predictions
        if prediction_type == 'label':
            apply_model = lambda x: self.model.predict(x)
        else:
            apply_model = lambda x: self.model.predict_proba(x)

        pred = self.model.predict(np.array([[int(point[var.name]) for var in self.dag.features]]))[0]
        prob_yhat_a_gets_curr_a = 1

        # computing the maximum difference in the categorical probability distribution and setting that as delta
        max_delta = 0
        for pair_prob_dists in all_rel_cond_probs_comparisons:
            # pair_prob_dists contains P(proxy | protected_att = curr_a) and P(proxy | protected_att = b) for b != curr_a
            prob_yhat_A_gets_b_mid_proxy_equal_val = []
            for num, proxy_val in enumerate(proxy.values):
                proxy_equal_val_comma_x = np.array([[int(point[var.name]) if var.name != proxy.name else proxy_val for var in self.dag.features]])
                prob_proxy_equal_val_mid_a_equal_b = pair_prob_dists[1][num]
                # y_hat is binary hence this is P(\hat{Y} = 1 | proxy = val, X)
                cf_pred_mid_proxy_equal_val_comma_x = apply_model(proxy_equal_val_comma_x)[0]

                if prediction_type == "label":
                    if pred != cf_pred_mid_proxy_equal_val_comma_x:
                        prob_yhat_equal_pred_mid_proxy_equal_val_comma_x = 0
                    else:
                        prob_yhat_equal_pred_mid_proxy_equal_val_comma_x = 1
                else:
                    prob_yhat_equal_pred_mid_proxy_equal_val_comma_x = cf_pred_mid_proxy_equal_val_comma_x if pred == 1 else 1 - cf_pred_mid_proxy_equal_val_comma_x

                # P(Yhat_{A<-b} = pred | Proxy = val) = P(Yhat = pred | Proxy = val, x) * P(Proxy = val | protected_att = b)
                prob_yhat_A_gets_b_mid_proxy_equal_val.append(prob_yhat_equal_pred_mid_proxy_equal_val_comma_x * prob_proxy_equal_val_mid_a_equal_b)

            prob_yhat_a_gets_b = sum(prob_yhat_A_gets_b_mid_proxy_equal_val)
            delta = abs(prob_yhat_a_gets_curr_a - prob_yhat_a_gets_b)

            # searching for delta to limit the difference across different values of A from above;
            if delta > max_delta:
                max_delta = delta

        return max_delta

    def force_protected_values(self, mid_value_dict):
        """
        Force the protected attribute to take a certain value based on the robot mid (model id assigned based on features)

        :param mid_value_dict: dictionary of mid to protected value
        :return:
        """
        self.df[self.dag.protected.name] = self.df['mid'].map(mid_value_dict)
        print(self.df['delta'])
        df = self._assign_deltas(self.df)
        print(df['delta'])
        self.df = df
        print(self.df['delta'])

    def draw_fairness(self, delta=0.9, threshold=None):
        """
        Assign fairness to points based on the value of CF-fairness.

        For CF-fairness of delta, each points has delta probability of being unfair.

        Fair == 1.

        :param threshold: threshold on the delta parameter to account a point as fair or unfair; sampled if None
        :param delta: delta cf-fairness of the DAG
        :return fair_labels: vector of fairness for each point in the robot catalog
        """
        df = self.df
        U = self.rng.uniform(0, 1, df.shape[0])
        if threshold is None:
            fair_labels = np.less_equal(U, delta)
        else:
            if 'delta' not in self.df.columns:
                df = self._assign_deltas(df)
            fair_labels = np.less_equal(self.df['delta'], threshold)
        df['fairness'] = fair_labels
        self.df = df
        return fair_labels

    def find_ids(self, cf, col_scheme):
        """
        Find the IDs of the points in the dataframe that match the counterfactuals

        :param cf: counterfactuals to match
        :param col_scheme: color scheme to match
        :return: list of IDs
        """
        cf_ids = []
        for i, row in self.df.iterrows():
            if all([row[feat] == val for feat, val in cf.items()]) and row['color_scheme'] == col_scheme:
                cf_ids.append(row['id'])
        return cf_ids

    def get_robot(self, id_):
        """
        Create a Robot object out of a row in the dataframe.

        :param id_: ID of the robot to turn into the object
        :return: Robot object
        """
        assert 'CEs' in self.df.columns, "Have to sample the CEs first"
        assert 'Anchoring' in self.df.columns, 'Have to sample the sets first'
        assert 'Auditing' in self.df.columns, 'Have to sample the sets first'
        # Take mappers and the point out
        robot_row = self.df.loc[self.df['id'] == id_].to_dict('list')
        robot_row = {k: v[0] for k, v in robot_row.items()}  # flatten
        protected_atts = {self.dag.protected.name: robot_row[self.dag.protected.name]}
        features = {col: robot_row[col] for col in self.dag.measured_names}

        # Retrieve important characteristics
        fairness = robot_row['fairness']
        uid = robot_row['mid']
        delta = robot_row['delta']
        prediction = robot_row['yhat']
        anchoring = True if robot_row['Anchoring'] == 1 else False
        probing = True if robot_row['Anchoring'] in [2, 3] else False
        repeated_stage2 = True if robot_row['Anchoring'] == 3 else False
        auditing = True if robot_row['Auditing'] == 1 else False
        color_scheme = robot_row['color_scheme']
        outcome = robot_row[self.dag.outcome.name]

        # Retrieve previously sampled CEs
        cfs = robot_row['Sampled CEs']
        cfs_robot_row = [{k: v if k not in cf else 1-v for k, v in robot_row.items()} for cf in cfs]
        cfs_robot_row = [{k: v for k, v in cf.items() if k in self.dag.feature_names} for cf in cfs_robot_row]
        cf_list, cf_list_all = [], []
        for cf in cfs_robot_row:
            cf_list += self.find_ids(cf, col_scheme=color_scheme)

        non_cfs = [[feat for feat in features.keys() if feat not in c] for c in cfs]
        # randomize the order of unchanged features
        len_cfs = len(non_cfs)
        list_indices_cf = list(range(len_cfs))
        random.shuffle(list_indices_cf)
        non_cfs = [non_cfs[i] for i in list_indices_cf]

        proxy = self.dag.proxy.name

        return Robot(id_, uid, protected_atts, features, delta, fairness, anchoring, prediction, outcome, probing,
                     repeated_stage2, auditing, cf_list, non_cfs, proxy)

    def get_robot_set(self, robot_ids='auditing'):
        """
        Output a list of Robot class instance given ids

        :param robot_ids: list of robot ids/string
        :return: list of Robot instances
        """
        robot_list = []
        if robot_ids == 'auditing':
            robot_ids = list(self.df.loc[self.df['Auditing'] == 1]['ID'])
        elif robot_ids == 'probing':
            robot_ids = list(self.df.loc[self.df['Anchoring'] in [2, 3]]['ID'])
        for id_ in robot_ids:
            rob = self.get_robot(id_)
            robot_list.append(rob)
        return robot_list