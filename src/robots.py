"""
This file contains code for printing the robots and the anchoring set image as well as the Robot class
"""
import random

from src.paths import image_dir, static_dir, repo_dir

PROTECTED_ATT_NAMES = ['COMPANY X', 'COMPANY S']


class Robot:
    """
    Class representing features relevant for a robot used in the experiment
    """
    def __init__(self, id_, uid, protected_att_val, feature_vals, delta, fairness, anchoring, prediction, outcome,
                 probing, repeated_stage2, auditing, cf_list, normal_feats, proxy):
        self.robot_id = id_
        self.robot_uid = uid
        self.region_num = list(protected_att_val.values())[0]
        self.region_name = PROTECTED_ATT_NAMES[list(protected_att_val.values())[0]]
        self.region_names = PROTECTED_ATT_NAMES
        self.feature_values = feature_vals
        self.used_anchoring = anchoring
        self.used_probing = probing
        self.repeated_stage2 = repeated_stage2
        self.prediction = prediction
        self.prediction_str = 'DEFECTIVE' if prediction == 0 else "RELIABLE"
        self.cf_prediction_str = 'RELIABLE' if prediction == 0 else "DEFECTIVE"
        self.used_auditing = auditing

        self.counterfactuals = cf_list
        self.unchanged_cf_features = normal_feats

        self.proxy = proxy
        self.outcome = outcome
        self.outcome_str = 'DEFECTIVE' if outcome == 0 else "RELIABLE"

        if auditing:
            self.counterfactual_urls = []
            for cf in self.counterfactuals:
                zero_cf = "%03d" % cf
                counterfactual_url = str(image_dir.relative_to(repo_dir) / f'robot_{zero_cf}.png')
                self.counterfactual_urls.append(counterfactual_url)
        self.zero_robot_id = "%03d" % self.robot_id
        self.robot_url = str(image_dir.relative_to(repo_dir) / f'robot_{self.zero_robot_id}.png')

        self.fairness_judgement = None
        self.ground_truth_fairness = fairness
        self.delta = delta

        num_features = len(self.feature_values.keys())
        for unchanged_feats in self.unchanged_cf_features:
            if self.proxy not in unchanged_feats and len(unchanged_feats) == num_features-1:
                self.observed_fairness = False
                break
            self.observed_fairness = True
        self.region_decision = None
        self.cf_exclusive_proxy = not self.observed_fairness
