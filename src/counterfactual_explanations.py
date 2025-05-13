import random
import numpy as np
import pandas as pd
from src.utils import power_set


class CategoricalCounterfactualExplainer(object):
    """
    Class for generating all the CEs given a catalog with all the points and the model. It also provides some CE
    sampling functions and returns a list of Robot objects with sampled CEs.
    """
    def __init__(self, model, ordered_model_input, catalog, proxy_name, random_state=np.random.RandomState(seed=123)):
        """
        :param model: LinearClassificationModel instance
        :param catalog: RobotCatalog instance
        :param ordered_model_input: ordered list containg the names of the columns in catalog so that the model could be
               applied on that representation
        """
        self.model = model  # needs to have a 'predict' function
        self.point_catalog = catalog
        self.cfs = None
        self.closest_cfs = None
        self.proxy_rep = proxy_name
        self.sampled_cfs = None
        self.cf_dataframe = None
        self.ordered_model_input = ordered_model_input
        self.random_state = random_state

    def compute_cfs(self):
        """
        Specify counterfactual explanations for each point in a sampled RobotCatalog.

        Naive method. Go over each point and check all possible changes of features and select those that result in
        flipping the prediction.

        :return cfs_column: a dict of the point catalog column names or tuples of column names; the column names
                            indicate counterfactuals
        """
        cfs_dict = {}
        print('Computing CFs...')
        for _, point in self.point_catalog.iterrows():
            whole_p = dict(point)
            rel_p = {k: v for k, v in whole_p.items() if k in self.ordered_model_input}
            ordered_point = [rel_p[var] for var in self.ordered_model_input]
            pred = self.model.predict(np.array([ordered_point]))[0]
            id_ = whole_p['id']
            all_combinations = power_set(list(rel_p.keys()))
            cfs = []
            for n, ks in enumerate(all_combinations):
                point_cf = {k: rel_p[k] if k not in ks else 1 - rel_p[k] for k in list(rel_p.keys())}
                point_cf = self.point_catalog.loc[(self.point_catalog[list(point_cf)] ==
                                                   pd.Series(point_cf)).all(axis=1)]
                point_cf = point_cf.iloc[0]
                ordered_point_cf = [point_cf[var] for var in self.ordered_model_input]
                pred_cf = self.model.predict(np.array([ordered_point_cf]))[0]
                if pred_cf != pred:
                    cfs.append(ks)
            cfs_dict[id_] = cfs
        self.cfs = cfs_dict
        return cfs_dict

    def compute_closest_cfs(self):
        """
        Compute the closest counterfactuals for each point in the RobotCatalog.

        Select the shortest CFs among the computed CFs.
        """
        ccfs_dict = {}
        if self.cfs is None:
            self.compute_cfs()
        for id_, all_cfs in self.cfs.items():
            min_len = min([len(c) for c in all_cfs])
            closest_cfs = [c for c in all_cfs if len(c) == min_len]
            ccfs_dict[id_] = closest_cfs
        self.closest_cfs = ccfs_dict

    def sample_cfs(self, method=lambda x: random.sample(x, k=1)):
        """
        Compute closest CEs to show to people during the experiment according to some method

        :param method:
        :return:
        """
        sampled_cfs_dict = {}
        assert type(method) is callable or method in ['proxy', 'non_proxy', 'competing', 'random', 'multiple'], \
            "method needs to be a function or be 'proxy', 'no_proxy' 'competing', 'multiple' or 'random'"
        if type(method) is not callable:
            if method == 'proxy':
                method = self._sample_proxy_ce
            elif method == 'non_proxy':
                method = self._sample_non_proxy_ce
            elif method == 'competing':
                method = self._sample_proxy_non_proxy_ce
            elif method == 'random':
                method = self._sample_random_cf
            elif method == 'multiple':
                method = self._sample_proxy_and_ce

        if self.closest_cfs is None:
            self.compute_closest_cfs()
        for id_, _ in self.closest_cfs.items():
            sampled_cfs = method(id_)
            sampled_cfs_dict[id_] = sampled_cfs
        self.sampled_cfs = sampled_cfs_dict

    def _sample_random_cf(self, robot_id):
        """
        Given robot ID, sample one closest CE randomly

        :param robot_id:
        :return: [ CE ]
        """
        closest_cfs = self.closest_cfs[robot_id]
        self.random_state.shuffle(closest_cfs)
        return [closest_cfs[0]]

    def _sample_proxy_and_ce(self, robot_id):
        """
        Given robot ID, the smallest sample CE that contains the proxy with at least one additional feature

        :param robot_id:
        :return: [ CE ]
        """
        cfs = self.cfs[robot_id]
        cfs_valid = [cf for cf in cfs if len(cf) > 1]
        cfs_valid = sorted(cfs_valid, key=lambda x: len(x))
        min_len = len(cfs_valid[0])
        cfs_valid_filtered = [cf for cf in cfs_valid if len(cf) == min_len]
        self.random_state.shuffle(cfs_valid_filtered)
        out = [cfs_valid_filtered[0]]
        for cf in cfs_valid_filtered:
            if self.proxy_rep in cf:
                return [cf]
        return out

    def _sample_proxy_ce(self, robot_id):
        """
        Given robot ID, sample one closest CE that contains the proxy

        :param robot_id:
        :return: [ CE ]
        """
        closest_cfs = self.closest_cfs[robot_id]
        self.random_state.shuffle(closest_cfs)
        for cf in closest_cfs:
            if self.proxy_rep in cf:
                return [cf]
        return [closest_cfs[0]]

    def _sample_non_proxy_ce(self, robot_id):
        """
        Given robot ID, sample one closest CE that does not contain the proxy

        :param robot_id:
        :return: [ CE ]
        """
        closest_cfs = self.closest_cfs[robot_id]
        self.random_state.shuffle(closest_cfs)
        for cf in closest_cfs:
            if self.proxy_rep not in cf:
                non_proxy_cf = cf
                return non_proxy_cf
        return [closest_cfs[0]]

    def _sample_proxy_non_proxy_ce(self, robot_id):
        """
        Given robot ID, sample, possibly, 2 CEs: 1 with the proxy and 1 without it.

        :param robot_id:
        :return: [ CE ]
        """
        closest_cfs = self.closest_cfs[robot_id]
        proxy_cf, non_proxy_cf = None, None
        self.random_state.shuffle(closest_cfs)
        for cf in closest_cfs:
            if self.proxy_rep in cf and proxy_cf is None:
                proxy_cf = cf
            elif non_proxy_cf is None:
                non_proxy_cf = cf
        cfs = [proxy_cf, non_proxy_cf]
        cfs = [el for el in cfs if el is not None]
        return cfs

    def define_cf_dataframe(self):
        """
        Create a dataframe that contains a CF for each point in the point_catalog as a separate row, and 2 indicators:
        if that is the closest CF or if that is CF with the proxy

        :return: pd.DataFrame
        """
        if self.closest_cfs is None:
            self.compute_closest_cfs()
        df = pd.DataFrame(columns=['id', 'CE', 'is_closest', 'is_proxy'])
        for id_, cfs in self.cfs.items():
            for cf in cfs:
                df.loc[len(df.index)] = [id_, cf, cf in self.closest_cfs[id_], self.proxy_rep in cf]
        df.sort_values(by='id')

        self.cf_dataframe = df
        return self.cf_dataframe

