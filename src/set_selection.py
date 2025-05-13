import itertools
import warnings

import numpy as np
import pandas as pd
import math

from src.dag import Variable
from src.utils import power_set, abs_list


class SetSelector(object):
    """
    Object handling Anchoring and Auditing set generation, statistics computation given user labels, and Robot objects
    generation.
    """

    def __init__(self, robot_catalog, set_codes, random_state_seed=1111):
        self.robot_catalog = robot_catalog
        self.robot_catalog.df['Anchoring'] = ['?'] * self.robot_catalog.df.shape[0]
        self.robot_catalog.df['Probing'] = ['?'] * self.robot_catalog.df.shape[0]
        self.robot_catalog.df['Auditing'] = ['0'] * self.robot_catalog.df.shape[0]

        self.anchoring_var = self.robot_catalog.dag.protected

        self.anchoring_set = None
        self.probing_set = []
        self.probing_repeat = None
        self.auditing_set = None
        self.num_auditing_set = 0
        self.codes = set_codes

        self.random_state_seed = random_state_seed
        self.random_state = np.random.RandomState(seed=random_state_seed)

    def sample_model_anchoring(self, num):
        """
        Select points that show 2 levels of the predicated variable Y selected at random

        :param num:
        :return: { y: [ id ] } dict with keys being the values of the predicted variable and values being lists of ids
        """
        assert num % 2 == 0, "The size of the model anchoring set needs to be divisible by 2"
        to_select = num / 2
        value_0_points = self.robot_catalog.df[self.robot_catalog.df[self.robot_catalog.dag.outcome.name] == 0]
        value_1_points = self.robot_catalog.df[self.robot_catalog.df[self.robot_catalog.dag.outcome.name] == 1]
        value_0_points_num = list(value_0_points['id'])
        value_1_points_num = list(value_1_points['id'])
        p_0 = [p / sum(list(value_0_points['likelihood'])) for p in list(value_0_points['likelihood'])]
        p_1 = [p / sum(list(value_1_points['likelihood'])) for p in list(value_1_points['likelihood'])]
        sampled_0_points = self.random_state.choice(value_0_points_num, size=int(to_select), p=p_0)
        sampled_1_points = self.random_state.choice(value_1_points_num, size=int(to_select), p=p_1)

        outcome_dict = {0: sampled_0_points, 1: sampled_1_points}
        return outcome_dict

    def sample_anchoring(self, anchoring_var, num):
        """
        Select points for the anchoring set. Assumes the anchoring variable is binary.

        Choose n/2 points at random where the protected variable is == 0 and choose n/2 points that are identical in all
        but, possibly, the proxy from where the protected variable is == 1.

        :param anchoring_var: the anchoring variable in the RobotCatalog
        :param num: number of points in the anchoring set
        :return: [ id ] list chosen for the anchoring set
        """
        assert not isinstance(self.robot_catalog.df, type(None)), \
            "Cannot choose the auditing set before sampling the data!"
        if num % 2 != 0:
            raise Exception("The size of the anchoring set needs to be divisible by 2")
        if num > self.robot_catalog.df.shape[0]:
            raise Exception("Cannot sample so many points for the anchoring set")

        rep = anchoring_var.name if type(anchoring_var) is Variable else anchoring_var
        assert rep in self.robot_catalog.dag.measured_names, \
               "Cannot use that as the anchoring Variable, it is not a feature"

        all_region_a = self.robot_catalog.df.loc[(self.robot_catalog.df[rep] == 0)]
        all_region_b = self.robot_catalog.df.loc[(self.robot_catalog.df[rep] == 1)]
        all_region_a_num = list(all_region_a['id'])
        all_region_b_num = list(all_region_b['id'])

        # by default Region A needs to have no proxy
        region_a = self.robot_catalog.df.loc[(self.robot_catalog.df[rep] == 0) & (self.robot_catalog.df['yhat'] == self.robot_catalog.df['Y']) & (self.robot_catalog.df[self.robot_catalog.dag.proxy.name] == 0)]
        region_b = self.robot_catalog.df.loc[(self.robot_catalog.df[rep] == 1) & (self.robot_catalog.df['yhat'] == self.robot_catalog.df['Y'])]
        region_a_num = list(region_a['id'])
        region_b_num = list(region_b['id'])

        # make sure there are at least floor(num/4) robots with 1 color scheme
        colors_half_set_a, defective_half_set_a = 0, 0
        while colors_half_set_a < int(num/4) or colors_half_set_a >= math.ceil(num/4) or defective_half_set_a < int(num/4) or defective_half_set_a >= math.ceil(num/4):
            half_set_a = sorted(list(self.random_state.choice(region_a_num, int(num/2),
                                                              p=[p/sum(list(region_a['likelihood'])) for p in list(region_a['likelihood'])],
                                                              replace=False)))
            colors_half_set_a = [region_a[region_a['id'] == i]['color_scheme'].values[0] for i in half_set_a].count(2)
            defective_half_set_a = [region_a[region_a['id'] == i]['yhat'].values[0] for i in half_set_a].count(0)
            print("Num defective", defective_half_set_a)
            print("Num color", colors_half_set_a)

        half_set_b = []
        print(half_set_a)

        measured_no_proxy = [v for v in self.robot_catalog.dag.measured_names
                             if v not in [self.robot_catalog.dag.proxy.name, self.robot_catalog.dag.protected.name]]
        for _, row_a in region_a.iterrows():
            ind_a = row_a['id']
            if ind_a in half_set_a:
                temp_half_b = []
                p_temp = []
                for _, row_b in region_b.iterrows():
                    ind_b = row_b['id']
                    # same point but for the proxy, different protected attribute
                    if list(row_b[measured_no_proxy] - row_a[measured_no_proxy]) == [0] * len(measured_no_proxy)\
                            and row_b['color_scheme'] == row_a['color_scheme']:
                        if ind_b not in half_set_b:
                            temp_half_b.append(ind_b)
                            p_temp.append(row_b['likelihood'])
                half_set_b.append(self.random_state.choice(temp_half_b, p=[p/sum(p_temp) for p in p_temp]))
        if len(half_set_b) != len(half_set_a):
            raise Exception(f"The anchoring set has unequal number of examples from Region A ({len(half_set_a)}) vs " +
                            f"Region B ({len(half_set_b)}). Generate more points.")

        anchor_a = [self.codes["stage 1"] if r in half_set_a else 0 for r in all_region_a_num]
        anchor_b = [self.codes["stage 1"] if r in half_set_b else 0 for r in all_region_b_num]

        self.robot_catalog.df.loc[self.robot_catalog.df[anchoring_var.name] == 0, 'Anchoring'] = anchor_a
        self.robot_catalog.df.loc[self.robot_catalog.df[anchoring_var.name] == 1, 'Anchoring'] = anchor_b
        self.robot_catalog.df = self.robot_catalog.df.sort_values(by=['Anchoring', 'id'])

        # Turn the sampled values into a dictionary based on the values of the anchoring variable
        anchoring_ids = half_set_a+half_set_b
        self.anchoring_var = anchoring_var
        values = self.anchoring_var.values
        dct = {}
        num_values = len(values)
        parts = int(len(anchoring_ids) / num_values)
        for i, v in enumerate(values):
            dct[v] = anchoring_ids[i*parts:(i+1)*parts]
        self.anchoring_set = dct

        return anchoring_ids

    def _get_variable_value(self, var):
        """
        Obtain the value that is added to the points ID whenever the value of the input variable changes from 0 to 1

        Based on the 'binary_to_num' function.

        :param var: String name or an instance of the Variable class
        :return: the change in the point ID whenever the anchoring variable flips its value
        """
        if not isinstance(var, str):
            var = var.name
        position = self.robot_catalog.ordering.index(var)
        power = position+1
        difference = 2 ** power
        return difference

    def _check_samples(self, list_of_ids, checks=None):
        """
        Check if the sampled values do not contain two identical points with different values of the protected attribute

        :param list_of_ids: list of integer indicating point id computed via 'binary_to_number' function
        :param checks: names of columns in the dataframe whose values are irrelevant and do not count in determining
                       similarity
        :return: boolean indicating whether there are no 2 identical points
        """
        all_pairs = [p for p in power_set(list_of_ids) if len(p) == 2]
        all_point_pairs = [(self.robot_catalog.df[self.robot_catalog.df['id'] == p[0]][self.robot_catalog.dag.measured_names].drop(columns=checks+[self.robot_catalog.dag.protected.name]),
                            self.robot_catalog.df[self.robot_catalog.df['id'] == p[1]][self.robot_catalog.dag.measured_names].drop(columns=checks+[self.robot_catalog.dag.protected.name]))
                           for p in all_pairs]
        all_point_pairs_prot_att = [(self.robot_catalog.df[self.robot_catalog.df['id'] == p[0]][self.robot_catalog.dag.protected.name].values[0],
                                     self.robot_catalog.df[self.robot_catalog.df['id'] == p[1]][self.robot_catalog.dag.protected.name].values[0])
                                    for p in all_pairs]
        all_point_pairs_dict = [(p[0].to_dict('list'), p[1].to_dict('list')) for p in all_point_pairs]

        if any(np.array([p[0] == p[1] for p in all_point_pairs_dict]) & np.array([p[0] == p[1] for p in all_point_pairs_prot_att])):
            return False
        return True

    @staticmethod
    def _draw_matching_points(dataframe, measured_reps, var_name, k, random_state=np.random.RandomState(seed=1111),
                              sample_check_func=None):
        """
        Find k points sampled form the dataframe according to their likelihood and return 2 * k points where the other k
        points are counterparts of the sampled k points but with a different value on column var_name keeping other
        values from measured_reps intact.

        :param dataframe: dataframe to sample from
        :param measured_reps: list of relevant columns in the dataframe
        :param var_name: name of the variable whose value needs to be different between k and another k sampled points
        :param k: number of points to initially sample
        :param sample_check_func: function that checks if the initially drawn k points are ok
        :return: list of indices in dataframe that were sampled
        """
        available_points = dataframe
        available_points_num = list(available_points.index)

        checkmark = False
        sampled_indices_available_points = None
        while checkmark is not True:
            sampled_indices_available_points = sorted(random_state.choice(available_points_num, k,
                                                      p=[p / sum(list(available_points['likelihood'])) for p in
                                                         list(available_points['likelihood'])],
                                                      replace=False))
            print(sampled_indices_available_points)
            print([p / sum(list(available_points['likelihood'])) for p in list(available_points['likelihood'])])
            sampled_ids = [available_points.loc[index]['id'] for index in sampled_indices_available_points]
            checkmark = sample_check_func(sampled_ids, checks=[var_name])

        result = [1 if col == var_name else 0 for col in measured_reps]
        sampled_indices_match_points = []

        for ind_row, row in available_points.iterrows():
            if ind_row in sampled_indices_available_points:
                temp_half_b = []
                p_temp = []
                for ind_match, row_match in available_points.iterrows():
                    # same point but for the proxy, different/same protected attribute
                    if abs_list(list(row_match[measured_reps] - row[measured_reps])) == result:
                        if ind_match not in sampled_indices_match_points + sampled_indices_available_points:
                            temp_half_b.append(ind_match)
                            p_temp.append(row_match['likelihood'])
                if len(p_temp) == 0:
                    raise Exception("Can't find a match for all the necessary points, reduce k.")
                sampled_indices_match_points.append(random_state.choice(temp_half_b, p=[p / sum(p_temp) for p in p_temp]))
        return sampled_indices_available_points+sampled_indices_match_points

    def sample_probing(self, omit=None, repeat=0):
        """
        Sample points for probing.

        Go over feature, num_points pairs and sample pairs of points that differ on the feature and are otherwise
        the same.

        Update the dataframe in the column "Anchoring" with the proper code for the 2nd stage.

        :param omit: { col : val } dictionary used to filter out unwanted points from the dataframe
        :param repeat: number of points to be used twice in the anchoring testing stage
        :return: IDs of the points selected for the second stage of anchoring and IDs of points that are to be repeated
        """
        if omit is None:
            omit = {}

        # Filter out unwanted points
        available_points = self.robot_catalog.df.copy()
        for column, values in omit.items():
            if not isinstance(values, list):
                values = [values]
            available_points = available_points[~available_points[column].isin(values)]
        measured_reps = self.robot_catalog.dag.feature_names
        measured_reps_no_proxy = [v for v in measured_reps if v != self.robot_catalog.dag.proxy.name]
        proxy = self.robot_catalog.dag.proxy.name

        available_points = available_points.sample(frac=1, random_state=self.random_state, replace=False)
        available_points.drop_duplicates(subset=measured_reps, keep='first', inplace=True)
        selected_indices = list(available_points['id'])

        selected_pair_indices = []
        for rid in selected_indices:
            robot_row = self.robot_catalog.df[self.robot_catalog.df['id'] == rid].iloc[0]
            copies = self.robot_catalog.df.query(' & '.join([f"{m} == {str(robot_row[m])}" for m in measured_reps_no_proxy]) +
                                                 ' & color_scheme == ' + str(robot_row['color_scheme']))
            for ind, pair_robot in copies.iterrows():
                if pair_robot[proxy] != robot_row[proxy]:
                    selected_pair_indices.append((rid, pair_robot['id']))
                    break

        self.probing_set = selected_pair_indices

        # add codes to all selected ids
        self.robot_catalog.df.loc[selected_indices, 'Probing'] = [self.codes["stage 2"]] * len(selected_indices)

        # add 0 codes for all the remaining points
        self.robot_catalog.df.loc[self.robot_catalog.df['Probing'] == '?', 'Probing'] = [0] * self.robot_catalog.df[self.robot_catalog.df['Probing'] == '?'].shape[0]

        # Sampling robots to repeat in the 2nd stage
        idx = self.random_state.choice(len(self.probing_set), size=repeat, replace=False)
        # Use indices to get the tuples
        repeat_ids = [self.probing_set[i] for i in idx]
        flattened_repeat_ids = [r for ri in repeat_ids for r in ri]
        self.robot_catalog.df.loc[flattened_repeat_ids, 'Probing'] = [self.codes["repeated stage 2"]] * repeat * 2

        self.probing_repeat = repeat_ids

        return self.probing_set, repeat_ids

    def _select_audit_candidates(self):
        """
        Select points for the auditing set as all the points that are not in the Anchoring set and have decision == 0.

        :return: DataFrame updated with which points are in the auditing set and nicely sorted
        """
        anchoring_var = self.anchoring_var
        assert anchoring_var is not None, "Cannot sample the auditing set before sampling the Anchoring set"
        assert not isinstance(self.robot_catalog.df, type(None)), \
            "Cannot choose the auditing set before sampling the data!"

        rep = anchoring_var.name if type(anchoring_var) is Variable else anchoring_var
        assert rep in self.robot_catalog.dag.measured_names, \
            f"Cannot use {rep} as the anchoring Variable, it is not a feature: {self.robot_catalog.dag.measured_names}"

        all_a_idx = ((self.robot_catalog.df['Anchoring'] == 0) &
                     (self.robot_catalog.df[rep] == 0) & (self.robot_catalog.df['yhat'] == 0))
        all_b_idx = ((self.robot_catalog.df['Anchoring'] == 0) &
                     (self.robot_catalog.df[rep] == 1) & (self.robot_catalog.df['yhat'] == 0))

        # Choose only points which have a negative decision, and from each protected variable level
        all_a = self.robot_catalog.df.loc[all_a_idx]
        all_b = self.robot_catalog.df.loc[all_b_idx]

        # Every point whose decision is 0 is in the Auditing set
        new_audit_a = [1] * len(list(range(all_a.shape[0])))
        new_audit_b = [1] * len(list(range(all_b.shape[0])))

        self.robot_catalog.df.loc[all_a_idx, 'Auditing'] = new_audit_a
        self.robot_catalog.df.loc[all_b_idx, 'Auditing'] = new_audit_b
        self.robot_catalog.df = self.robot_catalog.df.sort_values(by=['Anchoring', 'Probing', 'Auditing', 'id'])

        return self.robot_catalog.df

    def sample_auditing(self, num):
        """
        Select random num points for auditing that were selected as candidates for the auditing set by _select_audit.

        :param num: number of points to sample
        :return: [ id ] randomly shuffled list of num point ids sampled from the auditing set
        """
        self._select_audit_candidates()
        temp_df = self.robot_catalog.df.copy()
        temp_df["likelihood"] = round(temp_df["likelihood"], 4)
        print(temp_df.to_string())
        # save to excel
        #temp_df.to_excel('test.xlsx')
        anchoring_var = self.anchoring_var
        assert anchoring_var is not None, "Cannot sample the auditing set before sampling the Anchoring set"
        candidates = self.robot_catalog.df
        max_num = candidates[(candidates['Anchoring'] == 0) & (candidates['yhat'] == 0)].shape[0]
        req_num = num
        if max_num < num:
            warnings.warn(f"Can only sample {max_num} points for auditing")
            num = max_num

        rep = anchoring_var.name if type(anchoring_var) is Variable else anchoring_var

        half_num_a = int(float(num) / 2)
        half_num_b = num - half_num_a

        all_a = self.robot_catalog.df.loc[(self.robot_catalog.df['Auditing'] == 1) & (self.robot_catalog.df[rep] == 0)].copy()
        all_b = self.robot_catalog.df.loc[(self.robot_catalog.df['Auditing'] == 1) & (self.robot_catalog.df[rep] == 1)].copy()

        # We sample num / 2 points with one level of the protected attributed and num / 2 with another level
        ids_all_a = list(all_a['id'])
        ids_all_b = list(all_b['id'])

        chosen_indices_a, chosen_indices_b = [], []
        replace_a, replace_b = False, False
        if half_num_a > len(ids_all_a):
            chosen_indices_a = ids_all_a
            replace_a = True
        if half_num_b > len(ids_all_b):
            chosen_indices_b = ids_all_b
            replace_b = True
        chosen_indices_a += list(self.random_state.choice(ids_all_a, half_num_a-len(chosen_indices_a), replace=replace_a))
        # choose the rest with the preference for robots that were not previously selected (have different features)
        selected_a = all_a.loc[chosen_indices_a]
        # create a new column in all_b that says if robot with antenna body head legs is in selected_a
        measured_reps = self.robot_catalog.dag.feature_names
        all_b.loc[:, 'in_selected_a'] = [1 if any([all([r == s for r, s in zip(row[measured_reps], selected_row[measured_reps])])
                                           for _, selected_row in selected_a.iterrows()]) else 0
                                 for _, row in all_b.iterrows()]
        remaining_b = list(all_b[all_b['in_selected_a'] == 0]['id'])
        chosen_indices_b += list(self.random_state.choice(remaining_b, min(half_num_b, len(remaining_b)), replace=False))
        chosen_indices_b += list(self.random_state.choice(ids_all_b, half_num_b-len(chosen_indices_b), replace=replace_b))

        all_indices = chosen_indices_a + chosen_indices_b
        self.random_state.shuffle(all_indices)
        self.auditing_set = all_indices

        if req_num-max_num > 0:
            self.auditing_set += all_indices[:req_num-max_num]

        return all_indices


    def _assign_sets(self, anchoring, probing, repeat, audit_num):
        """
        Given anchoring sets as special dicts, assign proper values to the DataFrame and update the class variables.

        All the points which are not in the anchoring sets and have Y' == 0 are labeled as the auditing set.

        :param anchoring: { anchoring_var : [ id ] } dict with ids of the points in the anchoring set
        :param probing: [ (id, id) ] list with ids of the points in the probing set
        :param repeat: [ (id, id) ] of points that are to be repeated during probing
        :param audit_num: number of points to sample for the auditing set
        """
        codes = self.codes
        for num, point in self.robot_catalog.df.iterrows():
            id_ = point['id']
            if id_ in sum(list(anchoring.values()), []):
                point['Anchoring'] = codes['stage 1']
                point['Auditing'] = 0
            elif id_ in [p[0] for p in probing] + [p[1] for p in probing]:
                all_id_repeat = [p[0] for p in repeat] + [p[1] for p in repeat]
                point['Probing'] = codes['repeated stage 2'] if id_ in all_id_repeat else codes['stage 2']
                point['Anchoring'] = 0
                point['Auditing'] = 0
            elif point[self.robot_catalog.dag.outcome.name] == 0:
                point['Auditing'] = 1
                point['Probing'] = 0
                point['Anchoring'] = 0
            else:
                point['Anchoring'] = 0
                point['Probing'] = 0
                point['Auditing'] = 0
            self.robot_catalog.df.iloc[num] = point
        self.robot_catalog.df = self.robot_catalog.df.sort_values(by=['Anchoring', 'Probing', 'Auditing', 'id'])
        self.anchoring_set = anchoring
        self.probing_set = probing
        self.probing_repeat = repeat
        self.auditing_set = self.sample_auditing(audit_num)

    def sample_sets(self, num_anchoring, omit_probing, num_repeat_probing, num_audit,
                    sampled_anchoring, sampled_probing, repeat_probing, sets_fixed=False, **kwargs):
        """
        Sample points for the experiment or assign points to the sets provided in the input.

        :param num_anchoring: number of points to sample for the anchoring set
        :param omit_probing: { col : val } dictionary used to filter out unwanted points from the dataframe
        :param num_repeat_probing: number of points to be used twice in probing
        :param num_audit: number of points to sample for the auditing set
        :param sampled_anchoring: { val : [ id ] } dict with keys being the values of the anchoring variable and
                                      values being lists of ids to show during anchoring for that value of the
                                      anchoring variable
        :param sampled_probing: [ (id, id) ] list of pairs to show during probing
        :param repeat_probing: [ (id, id) ] list of pairs to repeat during probing
        :param sets_fixed: whether to regenerate the sets or used the data passed to the function
        """
        self.num_auditing_set = num_audit
        if not sets_fixed:
            # Draw robots to show in the experiment
            anc_var = self.anchoring_var if self.anchoring_var is not None else kwargs['anchoring_var']
            if omit_probing is None:
                omit_probing = {}

            unique_robots = self.robot_catalog.df.drop_duplicates(subset=self.robot_catalog.dag.feature_names,
                                                                  keep='first', inplace=False)
            num_unique_robots = unique_robots.shape[0]
            while len(self.probing_set) < num_unique_robots:
                self.robot_catalog.df['Anchoring'] = ['?'] * self.robot_catalog.df.shape[0]
                self.robot_catalog.df['Probing'] = ['?'] * self.robot_catalog.df.shape[0]
                self.sample_anchoring(anc_var, num_anchoring)
                self.sample_probing(omit_probing, num_repeat_probing)
                self.random_state_seed += 1
                self.random_state = np.random.RandomState(seed=self.random_state_seed)
                print(len(self.probing_set))
                print(self.random_state_seed)

            self.sample_auditing(num_audit)

        else:
            # Assign proper sets that were previously hand-picked/generated
            self._assign_sets(anchoring=sampled_anchoring,
                              probing=sampled_probing,
                              repeat=repeat_probing,
                              audit_num=num_audit)
        self.sort_df()

    def get_sets(self):
        """
        Return ids of robots in the anchoring and auditing sets.

        :return anchoring: dict with keys being the values of the anchoring variable and values being lists of
                                  ids to show during the first stage of anchoring for that value of the anchoring var
        :return probing: a list of pairs to show during probing
        :return repeat_probing: a list of pairs from probing to repeat
        :return: auditing: [ id ] ordered list of point ids for auditing

        """
        anchoring = self.anchoring_set
        probing = self.probing_set
        repeat_probing = self.probing_repeat
        auditing = self.auditing_set
        return anchoring, probing, repeat_probing, auditing

    def sort_df(self):
        self.robot_catalog.df = self.robot_catalog.df.sort_values(by=['Anchoring', 'Probing', 'Auditing', 'id'])
