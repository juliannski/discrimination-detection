"""
This file contains the Experiment class for assigning experiment sets, computing results and outputting Robot objects.
"""
import pickle
import numpy as np
import json
from src.paths import static_dir
from src.set_selection import SetSelector
from src.classifier import INTERCEPT_NAME, LinearClassificationModel
from src.counterfactual_explanations import CategoricalCounterfactualExplainer
from src.dag import Variable, FinalBooleanVariable, FairnessDAG
from src.robot_catalog import RobotCatalog


def check_parameters(p):

    needed_params = ['y_func', 'yhat_func', 'proxy', 'n_auditing', 'CE_method', 'n_probing',
                     'n_probing_repeat', 'n_anchoring', 'anchoring', 'probing', 'repeat',
                     'sample_sets', 'fairness_threshold']
    assert all([n in p.keys() for n in needed_params]), "Some parameter is missing..."

    assert p['n_probing_repeat'] < p['n_probing'], \
        "Can't repeat more robots than are sampled"

    if p['proxy_strength'] in p['probing'].keys() and not p['sample_sets']:
        assert p['n_probing'] == len(p['probing'][p['proxy_strength']]) + len(p['repeat']), \
            "Number of robots in the anchoring set stage 2 doesn't match the 'probing' and 'repeat' parameter"

    if len(p['repeat']) > 0:
        assert p['n_probing_repeat'] == len(p['repeat']), \
                "Number of robots to repeat in anchoring stage 2 does not match 'n_probing_repeat' parameter"

    if len(p['repeat']) > 0 and p['proxy_strength'] in p['probing'].keys():
        assert all([i in p['probing'][p['proxy_strength']] for i in p['repeat']]), \
            "ID mismatch. Cannot repeat a robot which is not selected for anchoring stage 2"

    if not p['sample_sets'] and p['proxy_strength'] not in p['probing'].keys():
        raise Exception("Cannot fix the sets if the anchoring set stage 2 is not defined")

    if not p['sample_sets'] and p['proxy_strength'] not in p['anchoring'].keys():
        raise Exception("Cannot fix the sets if the anchoring set stage 1 is not defined")

    return True


def setup_experiment(settings):

    random_state = np.random.RandomState(seed=settings['seed'])

    # Transform script_parameters to sth usable
    assert check_parameters(settings)

    proxy_strength = settings['proxy_strength']

    # Variable definition and DAG definition
    feature_nodes = [k for k in settings['yhat_func'].keys() if k not in (INTERCEPT_NAME, settings['proxy'])]
    aprobs = settings.get('custom_a', [0.5, 0.5])
    A = Variable(name='A', values=[0, 1], probabilities=aprobs)
    B = Variable(name=settings['proxy'], values=[0, 1],
                 probabilities=dict({'A_0': settings['proxy_params'][proxy_strength][0],
                                     'A_1': settings['proxy_params'][proxy_strength][1]}))
    X = {k: Variable(name=k, values=[0, 1], probabilities=[0.5, 0.5]) for k in feature_nodes}
    Y = FinalBooleanVariable(name='Y', vars_ordering=[B.name] + [x.name for x in X.values()],
                             model_parameters=settings['y_func'])
    dag = FairnessDAG(features=X, outcome=Y, protected=A, proxy=B)
    dag.add_edges_from_coefficients(coefficients=settings['yhat_func'])

    # Define the predictor
    feature_ordering = dag.feature_names
    yhat_func = {k: settings['yhat_func'][k] for k in feature_ordering + [INTERCEPT_NAME]}
    model = LinearClassificationModel.from_dict(**yhat_func)

    # Create the robot catalog
    # Generate a giant data frame of attributes, robots, likelihoods, true labels
    catalog = RobotCatalog(dag=dag, model=model, random_state=random_state)
    catalog.generate(n=dag.n_distinct, with_replacement=False)
    catalog.adapt(to_size=settings['n_robots'])
    catalog.draw_fairness(threshold=settings['fairness_threshold'])

    # Compute CEs based on the model and the sampled data in the RobotCatalog
    cf_explainer = CategoricalCounterfactualExplainer(model=model,
                                                      ordered_model_input=catalog.dag.feature_names,
                                                      catalog=catalog.df,
                                                      proxy_name=catalog.dag.proxy.name,
                                                      random_state=random_state)

    cf_explainer.sample_cfs(method=settings['CE_method'])
    all_cfs = cf_explainer.define_cf_dataframe()

    catalog.df['CEs'] = list(cf_explainer.cfs.values())
    catalog.df['Closest CEs'] = list(cf_explainer.closest_cfs.values())
    catalog.df['Sampled CEs'] = list(cf_explainer.sampled_cfs.values())
    print(catalog.df.to_string())
    print(f"Accuracy: {1-(catalog.df['Y'] - catalog.df['yhat']).mean()}")

    experiment_handle = SetSelector(robot_catalog=catalog,
                                    set_codes={"stage 1": 1, "stage 2": 1, "repeated stage 2": 2},
                                    random_state_seed=settings['seed'])

    # Establish anchoring and auditing sets by creating an Experiment object that sets it
    experiment_handle.sample_sets(num_anchoring=settings['n_anchoring'],
                                  omit_probing={'Anchoring': 1, 'Auditing': 1},
                                  num_repeat_probing=settings['n_probing_repeat'],
                                  num_audit=settings['n_auditing'],
                                  sampled_anchoring=settings['anchoring'].get(proxy_strength, None),
                                  sampled_probing=settings['probing'].get(proxy_strength, None),
                                  repeat_probing=settings['repeat'],
                                  sets_fixed=not settings['sample_sets'])

    return experiment_handle, settings


DEFAULT_SETTINGS = {
    'w_ground_truth': 'standard_gt_weights',
    'w_model': 'strong_proxy',
    'proxy': 'antenna',
    'fairness_threshold': 0.2,
    'CE_method': 'competing',                             # method for sampling CEs; "random", "competing" and "proxy"
    'n_auditing': 20,
    'anchoring_codes': 'default_codes',                   # to put in the dataframe
    'excluded_stage2': 'no_anchoring_auditing',           # column : value
    'included_stage2': 'proxy_8',                         # feature : number of robots to extract,
    'n_probing_repeat': 2,                               # number of robots that repeat in the 2nd anchoring stage
    'n_anchoring': 8,
    'anchoring': 'anchoring_strong_proxy_stage1',  # ids of the robots in the Anchoring set for stage 1
    'probing': 'anchoring_strong_proxy_stage2',  # ids of the robots in the Anchoring set for stage 2
    'repeat': 'consistency_points_strong_proxy',          # ids of the robots to repeat in the Anchoring set for stage 2
    'sets_fixed': True                                    # whether the sets in the parameters are to be used or not
    }

if __name__ == "__main__":
    experiment_handle, settings = setup_experiment(DEFAULT_SETTINGS)

    with open(static_dir / 'experiment_handle.pkl', 'wb') as f:
        pickle.dump(experiment_handle, f)

    print("\n\nAFTER SAMPLING THE EXPERIMENTAL SETS:\n")
    print(experiment_handle.robot_catalog)

    # Sanity checks
    print("\n\nSANITY CHECKS:")
    anchoring1, anchoring2, repeated, audit = experiment_handle.get_sets()
    print(f'\nAnchoring stage 1: {anchoring1}\nAnchoring stage 2: {anchoring2}\nRepeat: {repeated}\nAudit: {audit}')
    audit = experiment_handle.sample_auditing(settings['n_auditing'])
    print(f'Resampled audit: {audit}')

    # Robot creation checks
    print("\n\nCREATING ROBOTS:\n")
    _, _, _, audit = experiment_handle.get_sets()
    audit_samp = experiment_handle.sample_auditing(settings['n_auditing'])
    print(f'Audit sampled again: {audit_samp}')
    audit_robots = experiment_handle.get_robots(robot_ids=audit_samp)
    print(f'List of robot classes: {audit_robots}')
    json_str = json.dumps(audit_robots[0].__dict__)
    print(f'Sample Robot JSON: {json_str}')
