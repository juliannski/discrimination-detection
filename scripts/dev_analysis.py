import pickle
import time
import copy
import pandas as pd
import matplotlib.pyplot as pp
from pathlib import Path
from src.paths import static_dir
from src.utils import cond_probs
from reporting.processing import compute_consistency, compute_stats


def get_accuracy_range(rob_cat, deltas_list, fairness_labels, min_l, draws, precision):
    """
    Compute minimum and maximum accuracy of fairness judgements given by people in the experiment across all possible
    DAGs (up to 'precision') for predefined values of fairness in 'deltas_list'.

    :param rob_cat: RobotCatalog object
    :param deltas_list: List of deltas for delta-CF fairness to compute fairness judgements accuracy range for
    :param fairness_labels: List of ({col_name: value}, fairness judgement) pairs provided by participants
    :param min_l: minimum likelihood for the anchoring and auditing set
    :param draws: number of times to sample the set RobotCatalog for ground truth fairness
    :param precision: how granular should the probabilities defining the proxy be to satisfy CF fairness value
    :return: min/max accuracy of fairness_labels across all possible DAGs with a given delta-CF fairness
    """

    min_accuracies, max_accuracies = [], []
    for cf in deltas_list:
        st = time.time()
        probs = cond_probs(cf, precision=precision)
        print(f"CF fairness: {cf}. Iterate over {len(probs)} probs")
        min_p = 1
        max_p = 0
        rob_cat.delta = cf
        plausible = 0
        for i, prob in enumerate(probs):
            print(f"\r{i+1}|{len(probs)}", end='')
            rob_cat.dag.proxy.cond_prob = dict({'A_0': prob[0],
                                                'A_1': prob[1]})
            rob_cat.recompute_likelihood(subset='Anchoring')
            rob_cat.recompute_likelihood(subset='Auditing')

            lik_anchor = rob_cat.compute_seq_likelihood(subset='Anchoring')

            print(lik_anchor)

            if lik_anchor >= min_l:
                plausible += 1
                acc = 0
                for _ in range(draws):
                    # compute fairness
                    rob_cat.draw_fairness()
                    a = compute_stats(rob_cat, fairness_labels, set_='Auditing')['accuracy']
                    acc += a
                acc = acc / draws
                if acc < min_p:
                    min_p = round(acc, 5)
                if acc > max_p:
                    max_p = round(acc, 5)

        if min_p != 1:
            min_accuracies.append(min_p)
        if max_p != 0:
            max_accuracies.append(max_p)
        e = time.time()
        if min_p != 1 and max_p != 0:
            print(f"\n\nAccuracies: {min_p} up to {max_p} for {plausible} sets computed in {e - st}s")
        else:
            print(f"\n\nNo plausible sets for this CF fairness computed in {e - st}s.")
    return min_accuracies, max_accuracies


def get_minimax_deltas(robot_catalog, deltas_list, max_l, precision):

    min_deltas_ = [1 for _ in range(robot_catalog.df.shape[0])]
    max_deltas_ = [0 for _ in range(robot_catalog.df.shape[0])]
    for cf in deltas_list:
        st = time.time()
        probs = cond_probs(cf, precision=precision)
        print(f"CF fairness: {cf}. Iterate over {len(probs)} probs")
        for i, prob in enumerate(probs):
            robot_catalog.dag.proxy.cond_prob = dict({'A_0': prob[0],
                                                      'A_1': prob[1]})
            robot_catalog.recompute_likelihood(subset='Anchoring')
            robot_catalog.recompute_likelihood(subset='Auditing')
            robot_catalog._assign_deltas(df=robot_catalog.df)

            lik_anchor = robot_catalog.compute_seq_likelihood(subset='Anchoring')

            print(f"\r{i+1}|{len(probs)}   {lik_anchor} vs {max_l}", end='')
            if lik_anchor <= max_l:
                deltas = list(robot_catalog.df['delta'])
                for j in range(len(deltas)):
                    if deltas[j] < min_deltas_[j]:
                        min_deltas_[j] = deltas[j]
                    if deltas[j] > max_deltas_[j]:
                        max_deltas_[j] = deltas[j]

        e = time.time()
        print(f"\n\nNew deltas: {max_deltas_} computed in {e - st}s.")
    return min_deltas_, max_deltas_


def generate_stats_arrays(list_user_labels):
    """
    Go over a list of user labels and output a list of statistics associated with those labels in a dictionary keyed
    with the name of the statistic (Set in Experiment)

    :param list_user_labels: [ { robot_id : judgement } ] list
    :return stats: { statistic_name : [ value ] } dictionary
    """
    stats = compute_stats(list_user_labels[0])
    stats = {name: [] for name in stats.keys()}
    for labels in list_user_labels:
        s = compute_stats(labels)
        for name in s.keys():
            stats[s].append(s[name])
    return stats


def wrapper_accuracy_range(robot_catalog, user_labels):
    """
    Perform get_minmax_Delta, then get_accuracy_range.
    """
    robot_catalog_copy = copy.deepcopy(robot_catalog)
    lik_map = robot_catalog_copy.compute_seq_likelihood(subset='Anchoring')
    print(lik_map)
    deltas = list(np.arange(0, 1, 0.01)[1:])
    draws = 50
    precision = 0.01

    for i in range(1):
        max_l = lik_map / 2
        min_deltas, max_deltas = get_minimax_deltas(robot_catalog=robot_catalog_copy,
                                                    deltas_list=deltas,
                                                    max_l=max_l,
                                                    precision=precision)
        print(min_deltas)
        print(max_deltas)

    plaus_acc_min, plaus_acc_max = {}, {}

    robot_catalog_copy = copy.deepcopy(robot_catalog)
    for i in range(10):
        min_l = 1e-50 * 10 ** i
        print(f"\n\n\nNEW PLAUSIBILITY: {min_l}\n\n\n")
        min_acc, max_acc = get_accuracy_range(rob_cat=robot_catalog_copy,
                                              deltas_list=deltas,
                                              fairness_labels=user_labels,
                                              min_l=min_l,
                                              draws=draws,
                                              precision=precision)
        if len(min_acc) != 0:
            plaus_min = min(min_acc)
        else:
            break
        if len(max_acc) != 0:
            plaus_max = max(max_acc)
        else:
            break

        print(f"\n\n\n\n MIN: {plaus_min}")
        print(f" MAX: {plaus_max}")

        plaus_acc_min[min_l] = plaus_min
        plaus_acc_max[min_l] = plaus_max

    return plaus_acc_min, plaus_acc_max


def plot_accuracy_range(plaus_acc_min, plaus_acc_max):

    xs = list(plaus_acc_min.keys())
    xs = [str(x) for x in xs]
    ticks = [10 * i for i in range(1, len(xs) + 1)]
    pp.xticks(ticks, xs)
    ys = list(plaus_acc_min.values())
    pp.plot(ticks, ys)
    pp.show()

    xs = list(plaus_acc_max.keys())
    xs = [str(x) for x in xs]
    pp.xticks(ticks, xs)
    ys = list(plaus_acc_max.values())
    pp.plot(ticks, ys)
    pp.show()


def load_data(params):
    filename = "experiment_handle_app_" + params['w_model'] + "_CE_" + params['CE_method'] + ".pkl"
    with open(static_dir.parent / filename, 'rb') as f:
        experiment = pickle.load(f)
    sessions = Path(static_dir.parent / 'results').glob('*')
    df = pd.DataFrame()
    for participant_data in sessions:
        with open(participant_data, 'rb') as f:
            session = pickle.load(f)
            results = session['results']
        if results['quiz_attempts'] < 3:  # rejection criterion
            df = df.append(results, ignore_index=True)
    print(df.to_string())
    return experiment, df


CONDITION = 0
MODEL = {0: 'strong_proxy', 1: 'medium_proxy', 2: 'weak_proxy'}
METHOD = {0: 'proxy_non_proxy', 1: 'competing', 2: 'multiple'}
SHOWN = {0: 'has_proxy', 1: "has_no_proxy"}
CE_TYPE = 0
PARAMS = {
    'condition': CONDITION,
    'w_model': MODEL[int(CONDITION / 3)],                     # underlying probabilities for the DAG
    'CE_method': METHOD[CONDITION % 3],
    'CE_shown': SHOWN[CE_TYPE]}


import numpy as np
import itertools
# function that generates all binary vectors of length n into a matrix
def get_points(n):
    return np.array(list(itertools.product([0, 1], repeat=n)))
# function for computing logistic function
def get_ly(v):
    return 1 / (1 + np.exp(-v))
# function for turning a logit vector into binary classification
def get_y(weights, intercept):
    vecs = get_points(len(weights))
    l = get_ly(np.matmul(vecs, weights) + intercept)
    return np.array([1 if i >= 0.5 else 0 for i in l])
# function for computing the accuracy of a classifier
def get_acc(y, yhat):
    return np.mean(y == yhat)

GTWEIGHTS = np.array([1, 1, 1, 1, 1, 0])
GTINTERCEPT = -3
FWEIGHTS = np.array([0.16, 0.16, 0.16, 0.16, 0.36, 0])
FINTERCEPT = -0.48
Y = get_y(GTWEIGHTS, GTINTERCEPT)
YHAT = get_y(FWEIGHTS, FINTERCEPT)
ACC = get_acc(Y, YHAT)

if __name__ == "__main__":
    experiment_handle, experiment_data = load_data(params=PARAMS)

    combined_plaus_min, combined_plaus_max = {}, {}
    global_statistics = {}

    for _, participant_data in experiment_data.iterrows():
        # User fairness judgements
        # { robot_id : judgement }
        # in proxy_non_proxy we assume the first 20 robots have the proxy, and the other 20 do not
        len_rob = len(participant_data['auditing']['robots'])
        if METHOD == 'proxy_non_proxy':
            ind_s = CE_TYPE*int(len_rob/2)
            ind_e = (CE_TYPE+1)*int(len_rob/2)
        else:
            ind_s = 0
            ind_e = len_rob
        ids = [participant_data['auditing']['robots'][i]['robot_id'] for i in
               range(len_rob)][ind_s:ind_e]
        judgements = [0 if val == 'FAIR' else 1 for val in participant_data['auditing']['judgements']][ind_s:ind_e]
        user_labels = {k: v for k, v in zip(ids, judgements)}

        # User anchoring stage2 proxy strength predictions
        # [ ( (robot_ids), decision ) ]
        ids = [participant_data['probing']['robots'][i*2]['pair_probing'] for i in
               range(int(len(participant_data['probing']['robots'])/2))]
        decisions = [0 if val == 'COMPANY X' else 1 for val in participant_data['probing']['decisions']]
        proxy_strength = float(participant_data['proxy_strength']) / \
                         (len(participant_data['probing']['robots']) / 2)
        stage2_decisions = list(zip(ids, decisions))
        inconsistency = compute_consistency(stage2_decisions)

        statistics = compute_stats(experiment_handle.robot_catalog, user_labels)
        if global_statistics == {}:
            global_statistics = {k: [v] for k, v in statistics.items()}
        else:
            global_statistics = {k: global_statistics[k] + [val] for k, val in statistics.items()}

        # Compute accuracy range per participant
        plaus_min, plaus_max = wrapper_accuracy_range(experiment_handle.robot_catalog, user_labels)

        if combined_plaus_min == {}:
            combined_plaus_min = {k: [v] for k, v in plaus_min.items()}
            combined_plaus_max = {k: [v] for k, v in plaus_max.items()}
        else:
            combined_plaus_min = {k: combined_plaus_min[k] + [val] for k, val in plaus_min.items()}
            combined_plaus_max = {k: combined_plaus_max[k] + [val] for k, val in plaus_max.items()}

    # Compute mean plausibility values
    combined_plaus_min = {k: np.mean(v)[0] for k, v in combined_plaus_min.items()}
    combined_plaus_max = {k: np.mean(v)[0] for k, v in combined_plaus_max.items()}
    print(combined_plaus_min)
    print(combined_plaus_max)

    # Plot this stuff
    plot_accuracy_range(combined_plaus_min, combined_plaus_max)

    # Average out the statistics
    global_statistics = {k: np.mean(v) for k, v in global_statistics.items()}

    # Sample results computation
    print("SAMPLE RESULTS: \n\n")
    print(f'\nFairness accuracy is {global_statistics["accuracy"]}\nFPR: {global_statistics["FPR"]}\nFNR: {global_statistics["FNR"]}')
    print(f'Inconsistency: {inconsistency}')
    print(f'Observed proxy strength: {proxy_strength}')


