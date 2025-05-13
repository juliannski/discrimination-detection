from datetime import datetime as timer
from itertools import chain
from scripts.dev_experiment import setup_experiment
from src.dag import INTERCEPT_NAME
import dill
import pandas as pd
import src.paths as paths
import warnings

warnings.filterwarnings('ignore')


def flatten(listlist):
    flat_list = [item for sublist in listlist for item in sublist]
    return flat_list


def peek(iterable):
    try:
        first = next(iterable)
    except StopIteration:
        return None
    return first, chain([first], iterable)


def compute_stats(robot_catalog, fairness_labels, set_=None):
    """
    Compute accuracy, false negative rate and false positive rate of human fairness judgements with respect to
    previously sampled ground truth.

    :param robot_catalog: RobotCatalog object
    :param fairness_labels: dict of { id : value } pairs, where the first element is the ID of the relevant point
                            and the second element is human-given fairness label
    :param set_: name of the column used to filter all rows but for that have a value of 1 for that column
    :return: stats: dictionary containing the accuracy, FPR and FNR of human judgements
    """
    corr, inc, fn, fp, tn, tp = 0, 0, 0, 0, 0, 0
    if set_ is not None:
        dataframe = robot_catalog.df[robot_catalog.df[set_] == 1]
    else:
        dataframe = robot_catalog.df
    for id_, pred_label in fairness_labels.items():
        point = dataframe.loc[dataframe['id'] == id_]
        corr_label = list(point['fairness'])[0]
        if pred_label == corr_label and corr_label == 0:
            tn += 1
            corr += 1
        elif pred_label == corr_label and corr_label == 1:
            tp += 1
            corr += 1
        elif pred_label != corr_label and corr_label == 1:
            fn += 1
            inc += 1
        elif pred_label != corr_label and corr_label == 0:
            fp += 1
            inc += 1
    acc = round(float(corr) / (corr + inc), 4)
    fpr = round(float(fp) / (fp + tn), 4) if (fp + tn) != 0 else 0
    fnr = round(float(fn) / (fn + tp), 4) if (fn + tp) != 0 else 0

    stats = {'accuracy': acc, 'FPR': fpr, 'FNR': fnr}
    return stats


def compute_consistency(protected_attribute_decisions):
    """
    Compute the number of times participant's judgement in the second stage of anchoring differed between the same
    pair of points

    :param protected_attribute_decisions: list [ ( (id, id), int/str ) where ids represent pairs of points sampled
                                          for the second stage of anchoring and integers/strings the decision
    :return: number of times a person was inconsistent
    """
    inconsistent_num = 0
    repeated_pairs, unique_pairs = [], []
    for pair in protected_attribute_decisions:
        if pair not in unique_pairs:
            unique_pairs.append(pair)
        else:
            repeated_pairs.append(pair)

    for r in repeated_pairs:
        decisions = []
        for pair, dec in protected_attribute_decisions:
            if pair == r:
                decisions.append(dec)
        mean = [d-decisions[0] for d in decisions]
        if sum(mean) != 0:
            inconsistent_num += 1
    return inconsistent_num


def find_copy_based_on_subset(df, el_id, subset):
    el_row = df.loc[df['id'] == el_id].iloc[0]
    for i, robot_row in df.iterrows():
        if robot_row['id'] != el_id and robot_row[subset].equals(el_row[subset]):
            return robot_row['id']


def compute_observed_proxy_strength(robot_catalog, protected_attribute_decisions):
    """
    Compute observed proxy strength.

    Go over all the protected var decisions and count the prediction as indicating proxy dependence on the anchoring
    variable whenever adding/removing the proxy causes change in the protected attribute.

    Indifferent to the direction of proxy change (appear/disappear)!

    :param robot_catalog: RobotCatalog object
    :param protected_attribute_decisions: [ ( (robot_ids), decision ) ] list of pairs
    :return strength: float indicating proxy strength. i.e. how many points were labeled as changing the protected
                      attribute when the proxy changed
    """
    repeated_pairs, unique_pairs = [], []
    for pair in protected_attribute_decisions:
        if pair not in unique_pairs:
            unique_pairs.append(pair)
        else:
            repeated_pairs.append(pair)

    all_pairs = unique_pairs + repeated_pairs
    proxy = robot_catalog.dag.proxy.name
    strength = 0
    all_examples = 0

    for id_pair, pred_protected_var in all_pairs:
        p1 = dict(robot_catalog.df.loc[robot_catalog.df['ID'] == id_pair[0]])
        p2 = dict(robot_catalog.df.loc[robot_catalog.df['ID'] == id_pair[1]])
        p1_proxy = list(p1[proxy.name])[0]
        p2_proxy = list(p2[proxy.name])[0]
        if p1_proxy != p2_proxy:  # the pair differs by the proxy
            start_protected_var = list(p1[robot_catalog.dag.protected.name])[0]
            all_examples += 1
            if start_protected_var != pred_protected_var:
                strength += 1

    strength = float(strength) / all_examples

    return strength


def process_participant_data(results, params):
    """
    results = {
     participant_df

     anchoring_df
     - robot_id
     - robot_features
     - robot_protected_attribute

     probing_df:
     - robot_id
     - robot_features
     - robot_protected_attribute
     - robot_has_proxy
     - company
     - reliability
     - prediction_time

     round_df
     - robot_id
     - robot_features
     - robot_protected_attribute
     - cf_num
     - cf_with_proxy
     - cf_exclusive_proxy
     - robot_observed_fairness
     - robot_gt_fairness
     - judgement
     }
    """

    # exclusion criteria
    havent_started = results.get('completed', []) in [['consent'], []]
    finished = results.get('finished', False) and len(results['completed']) > 1
    attention_check = False if havent_started or results.get('completed', [])[-1] == 'debrief_failed' else True
    quiz_attempts = results.get('quiz_attempts', 0)
    bonus = results.get('bonus', 0)

    print(f"pid: {results['pid']}, {bonus}")
    print(f"quizzes: {quiz_attempts}, attention: {attention_check}, finished: {finished}")
    if not havent_started:
        print(f"completed: {results['completed']}")
        print(f"probing: {results['probing']['companies']}")
        print(f"probing: {results['probing']['reliability']}")
        print(f"auditing: {results['auditing']['judgements']}")

    participant_id = results['pid']
    date = timer.strptime(results['time_start'], '%Y-%m-%d %H:%M:%S.%f').replace(microsecond=0)

    proxy_strength = params['proxy_strength']
    proxy = params['proxy']
    fairness_threshold = params['fairness_threshold']
    CE_method = params['CE_method']

    pvisited = results.get('completed', [])
    if not havent_started:
        early_exit = results.get('early_exit', False) or 'debrief' not in pvisited[-1] or "debrief_incomplete" in pvisited
    else:
        early_exit = False

    # disregard whomever who hasn't started
    if havent_started:
        return None, participant_id
    elif early_exit:
        return None, participant_id
    # exclude participant but save stats
    elif not all([finished, attention_check]):
        output = {'participant_id': participant_id,
                  'completion_check': finished,
                  'attention_check': attention_check,
                  'proxy': proxy,
                  'proxy_strength': proxy_strength,
                  'fairness_threshold': fairness_threshold,
                  'CE_method': CE_method,
                  'rejected': True,
                  'date': date,
                  'experimental_parameters': params,
                  'time': results['total_time'] if "total_time" in results.keys() else "NA",
                  'bonus': bonus
                  }

    else:

        robot_catalog = results['robot_catalog']

        # turn questionnaires to strings
        questionnaires = [k for k in results.keys() if 'questionnaire' in k]
        for q in questionnaires:
            results[q] = {k.text if not isinstance(k, str) else k: ', '.join(v) if type(v) is list else v
                          for num, (k, v) in enumerate(results[q].items()) if v is not None}

        anchoring_robots = [robot for v in results['anchoring'].values() for robot in v]
        anchoring_df = pd.DataFrame({'robot_id': [r['robot_id'] for r in anchoring_robots],
                                     'robot_protected_attribute': [k for k, v in results['anchoring'].items()
                                                                   for _ in range(len(v))]})
        for feat in anchoring_robots[0]['feature_values'].keys():
            anchoring_df[feat] = [r['feature_values'][feat] for r in anchoring_robots]

        print(f"TIMES: {len(results['probing']['times'])}, ROBOTS: {int(len(results['probing']['robots']) / 2)}")

        ########### AD HOC, judgements sometimes are saved up multiple times ###########################################
        len_dep = len(results['auditing']['robots'])
        times_dep = results['auditing']['times']
        len_prob = int(len(results['probing']['robots']) / 2)
        times_prob = results['probing']['times']
        sorted_times_prob_indices = sorted(range(len(times_prob)), key=lambda i: times_prob[i])
        sorted_times_dep_indices = sorted(range(len(times_dep)), key=lambda i: times_dep[i])


        if len(results['probing']['first_robots']) > len_prob:
            k = len(results['probing']['first_robots']) - len_prob
            indices_to_remove = sorted_times_prob_indices[:k]
            # remove k examples which have the lowest time
            results['probing']['first_robots'] = [rg for i, rg in enumerate(results['probing']['first_robots']) if
                                                   i not in indices_to_remove]
        if len(results['probing']['companies']) > len_prob:
            k = len(results['probing']['companies']) - len_prob
            indices_to_remove = sorted_times_prob_indices[:k]
            # remove k examples which have the lowest time
            results['probing']['companies'] = [rg for i, rg in enumerate(results['probing']['companies']) if
                                                   i not in indices_to_remove]
        if len(results['probing']['reliability']) > len_prob:
            k = len(results['probing']['reliability']) - len_prob
            indices_to_remove = sorted_times_prob_indices[:k]
            # remove k examples which have the lowest time
            results['probing']['reliability'] = [rg for i, rg in enumerate(results['probing']['reliability']) if
                                                   i not in indices_to_remove]
        if len(results['probing']['determinism']) > len_prob:
            k = len(results['probing']['determinism']) - len_prob
            indices_to_remove = sorted_times_prob_indices[:k]
            # remove k examples which have the lowest time
            results['probing']['determinism'] = [rg for i, rg in enumerate(results['probing']['determinism']) if
                                                   i not in indices_to_remove]
        if len(results['probing']['times']) > len_prob:
            k = len(results['probing']['times']) - len_prob
            indices_to_remove = sorted_times_prob_indices[:k]
            # remove k examples which have the lowest time
            results['probing']['times'] = [rg for i, rg in enumerate(results['probing']['times']) if
                                           i not in indices_to_remove]

        if len(results['auditing']['judgements']) > len_dep:
            k = len(results['auditing']['judgements']) - len_dep
            indices_to_remove = sorted_times_dep_indices[:k]
            # remove k examples which have the lowest time
            results['auditing']['judgements'] = [rg for i, rg in enumerate(results['auditing']['judgements']) if
                                                   i not in indices_to_remove]
        if len(results['auditing']['times']) > len_dep:
            k = len(results['auditing']['times']) - len_dep
            indices_to_remove = sorted_times_dep_indices[:k]
            # remove k examples which have the lowest time
            results['auditing']['times'] = [rg for i, rg in enumerate(results['auditing']['times']) if
                                           i not in indices_to_remove]
        ##############################################################################################################




        rids = [r['robot_id'] for i, r in enumerate(results['probing']['robots']) if i % 2 == 0]
        measured_reps = [k for k in params['y_func'].keys() if k != params['proxy'] and k != INTERCEPT_NAME]
        df = robot_catalog.df[robot_catalog.df['id'].isin(rids)]
        pair_rids = [find_copy_based_on_subset(df, rid, measured_reps) for rid in rids]
        print(pair_rids)
        company_dict = {r: results['probing']['companies'][i] for i, r in enumerate(rids)}
        reliability_dict = {r: results['probing']['reliability'][i] for i, r in enumerate(rids)}
        letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't']
        Letters = [l.upper() for l in letters]
        probs_company, probs_reliability = {}, {}
        for i, r in enumerate(rids):
            reliability_choice = reliability_dict[r]
            company_choice = company_dict[r]
            pair_company_choice = company_dict[pair_rids[i]]
            if r not in probs_company.keys():
                probs_company[r] = letters[i]
                if company_choice == pair_company_choice:
                    probs_company[pair_rids[i]] = probs_company[r]
            if r not in probs_reliability.keys():
                probs_reliability[r] = Letters[i]
                probs_reliability[pair_rids[i]] = Letters[i] if reliability_choice == 'same' else \
                    Letters[-1-i] if reliability_choice == 'idk' else \
                        Letters[i] + " - eps" if reliability_choice == 'less' else Letters[i] + " + eps"

        # assuming users know the protected attribute
        participant_protected_att_beliefs = {results['probing']['first_robots'][i]['robot_uid']:
                                             results['probing']['first_robots'][i]['region_names'].index(results['probing']['companies'][i]) if
                                             results['probing']['companies'][i] != 'idk' else results['probing']['first_robots'][i]['region_num']
                                             for i in range(len(results['probing']['first_robots']))}
        robot_catalog.force_protected_values(participant_protected_att_beliefs)
        robot_catalog.draw_fairness(threshold=fairness_threshold)
        robot_user_delta = {r['id']: r['delta'] for _, r in robot_catalog.df.iterrows()}
        robot_user_fairness = {r['id']: r['fairness'] for _, r in robot_catalog.df.iterrows()}

        # Participant's beliefs in probabilities P(A) and P(B|A). If they said "I don't know", use ground truth
        pba = {0: [0,0], 1: [0,0]}
        pa = [0, 0]
        for i, r in enumerate(results['probing']['first_robots']):
            val_b = r['feature_values'][r['proxy']]
            belief_company = results['probing']['companies'][i]
            belief_num = r['region_names'].index(belief_company) if belief_company != 'idk' else r['region_num']
            pa[belief_num] += 1
            pba[val_b][belief_num] += 1
        print(f"pba: {pba}, pa: {pa}")
        pba = {k: [float(pba[k][0]) / sum(pba[k]), float(pba[k][1]) / sum(pba[k])] for k in pba.keys()}
        # make sure the distribution is not [0,1] to [1,0] and add a small espilon to 0 adn remove from 1 if ti is
        pba = {k: [pba[k][0] + 0.0001 if pba[k][0] == 0 else pba[k][0],
                   pba[k][1] - 0.0001 if pba[k][1] == 1 else pba[k][1]] for k in pba.keys()}
        pba = {k: [pba[k][0] - 0.0001 if pba[k][0] == 1 else pba[k][0],
                   pba[k][1] + 0.0001 if pba[k][1] == 0 else pba[k][1]] for k in pba.keys()}
        pa = [float(pa[0]) / sum(pa), float(pa[1]) / sum(pa)]
        belief_params = params.copy()
        belief_params['proxy_strength'] = 'custom'
        belief_params['proxy_params']['custom'] = pba
        belief_params['custom_a'] = pa
        print(f"P(B|A): {pba}, P(A): {pa}")
        experiment_handle, _ = setup_experiment(belief_params)
        belief_robot_catalog = experiment_handle.robot_catalog
        def pab(robot, pba, pa):
            b_val = robot['feature_values'][robot['proxy']]
            p0_mid_bval = pba[0][b_val] * pa[0] / (1/2)
            p1_mid_bval = pba[1][b_val] * pa[1] / (1/2)
            return [p0_mid_bval, p1_mid_bval]

        def get_a_based_on_beliefs(robot, pba, pa):
            pab_ = pab(robot, pba, pa)
            return 0 if pab_[0] > pab_[1] else 1

        participant_protected_att_prob_beliefs = {results['probing']['first_robots'][i]['robot_uid']:
                                                  results['probing']['first_robots'][i]['region_names'].index(results['probing']['companies'][i]) if
                                                  results['probing']['companies'][i] != 'idk' else get_a_based_on_beliefs(results['probing']['first_robots'][i], pba, pa)
                                                  for i in range(len(results['probing']['first_robots']))}
        belief_robot_catalog.force_protected_values(participant_protected_att_prob_beliefs)
        robot_belief_delta = {r['id']: r['delta'] for _, r in belief_robot_catalog.df.iterrows()}

        probing_df = pd.DataFrame({'robot_id': [r['robot_id'] for r in results['probing']['first_robots']],
                                   'robot_protected_attribute': [r['region_name'] for r in results['probing']['first_robots']],
                                   'robot_has_proxy': [r['feature_values'][r['proxy']] for r in results['probing']['first_robots']],
                                   'company': results['probing']['companies'],
                                   'reliability': results['probing']['reliability'],
                                   'p(Company = S|X)': [probs_company[r] for r in rids],
                                   'p(Reliable|X, Company)': [probs_reliability[r] for r in rids],
                                   'prediction_time': results['probing']['times']})

        rel_to_num = lambda string: -1 if string == 'less' else 1 if string == 'more' else 0 if string != 'idk' else 100
        prot_to_num = lambda string: 100 if string == 'idk' else 0 if string == "COMPANY X" else 1
        if results['pid'] == '5e3eb0bcd3d69e10901f0cb3':
            for i, r in enumerate(results['probing']['first_robots']):
                print(
                    f"company: {results['probing']['companies'][i]}, region_name: {r['region_name']}, comparison: {results['probing']['companies'][i] == r['region_name']}, inside: {results['probing']['companies'][i] in [r['region_name'], 'idk']},"
                    f" A: {r['region_num'] if results['probing']['companies'][i] in [r['region_name'], 'idk'] else (r['region_num'] + 1) % 2},"
                    f" v: {prot_to_num(results['probing']['companies'][i])}")

        get_a_v = lambda rob, v: rob['region_num'] if v in [rob['region_name'], 'idk'] else (rob['region_num'] + 1) % 2
        answers_df = pd.DataFrame([
            {
                'A': get_a_v(r, results['probing']['companies'][i]),
                'B': r['feature_values'][r['proxy']],
                **{k: v for k, v in r['feature_values'].items() if k not in [r['proxy'], 'A']},
                'Y': r['outcome'],
                'Yhat': r['prediction'],
                'u': rel_to_num(results['probing']['reliability'][i]),
                'v': prot_to_num(results['probing']['companies'][i]),
                'probing': True
            }
            for i, r in enumerate(results['probing']['first_robots'])
        ])

        answers_df.drop_duplicates(subset=[c for c in answers_df.columns if c not in ['A', 'Y', 'Yhat', 'u', 'probing', 'v']], inplace=True)
        init_answers_df = answers_df.copy()
        # add rows with A = 1-A from init_asnwers_df, rest of the columsn the same
        all_unique_df = pd.concat([init_answers_df, init_answers_df.copy().assign(A=1 - init_answers_df['A'], probing=False)], ignore_index=True)

        jud_to_num = lambda string: 0 if string == "FAIR" else 1 if string == "UNFAIR" else 100
        judgments_df = pd.DataFrame([
            {
                **{'B' if k == r['proxy'] else k: v for k, v in r['feature_values'].items() if k != 'A'},
                'c': jud_to_num(results['auditing']['judgements'][i]),
                'Yhat': r['prediction']
            }
            for i, r in enumerate(results['auditing']['robots'])])
        print(judgments_df.to_string())
        judgments_df.drop_duplicates(subset=[c for c in judgments_df.columns if c not in ['c']], inplace=True)
        print(judgments_df.to_string())

        # add column "c" to answers_df joining by the other columns
        answers_df = answers_df.merge(judgments_df, how='right', on=[c for c in judgments_df.columns if c not in ['c', 'A']])
        print(answers_df.to_string())
        # rename columns that are neither A B Y, c nor u to X%d
        i = 1
        for col in answers_df.columns:
            if col not in ['A', 'B', 'Y', 'Yhat', 'c', 'u', 'probing', 'v']:
                answers_df.rename(columns={col: f'X{i}'}, inplace=True)
                init_answers_df.rename(columns={col: f'X{i}'}, inplace=True)
                all_unique_df.rename(columns={col: f'X{i}'}, inplace=True)
                i += 1
        print(answers_df.to_string())

        # save to an excel file
        with pd.ExcelWriter(paths.reports_dir / f'mip_{participant_id}.xlsx') as writer:
            answers_df.to_excel(writer, sheet_name='mip')
            init_answers_df.to_excel(writer, sheet_name='probing')
            all_unique_df.to_excel(writer, sheet_name='unique')

        mid_to_reliability_belief = {r['robot_uid']: results['probing']['reliability'][i] for i, r in enumerate(results['probing']['first_robots'])}

        round_df = {'robot_id': [r['robot_id'] for r in results['auditing']['robots']],
                    'robot_protected_attribute': [r['region_name'] for r in results['auditing']['robots']],
                    'proxy_reliability': [mid_to_reliability_belief[r['robot_uid']] for r in results['auditing']['robots']],
                    'cf_num': [len(r['counterfactuals']) for r in results['auditing']['robots']],
                    'cf_with_proxy': [any(r['proxy'] not in cf for cf in r['unchanged_cf_features']) for r in results['auditing']['robots']],
                    'cf_exclusive_proxy': [r['cf_exclusive_proxy'] for r in results['auditing']['robots']],
                    'robot_observed_fairness': [r['observed_fairness'] for r in results['auditing']['robots']],
                    'robot_gt_fairness': [r['ground_truth_fairness'] for r in results['auditing']['robots']],
                    'robot_user_fairness': [robot_user_fairness[r['robot_id']] for r in results['auditing']['robots']],
                    'robot_user_delta': [robot_user_delta[r['robot_id']] for r in results['auditing']['robots']],
                    'robot_belief_delta': [robot_belief_delta[r['robot_id']] for r in results['auditing']['robots']],
                    'robot_delta': [r['delta'] for r in results['auditing']['robots']],
                    'judgement': results['auditing']['judgements'],
                    'prediction_time': results['auditing']['times']}


        quest = results['questionnaire']

        questionnaire_df = pd.DataFrame({'question_id': [i+1 for i in range(len(quest.keys()))],
                                         'question_text': list(quest.keys()),
                                         'question_type': ['structured' if (quest[k] is not None and len(quest[k]) > 1)
                                                            else 'free' for k in quest.keys()],
                                         'response': list(quest.values())})

        output = {'participant_id': participant_id,
                  'completion_check': finished,
                  'attention_check': attention_check,
                  'proxy': proxy,
                  'proxy_strength': proxy_strength,
                  'fairness_threshold': fairness_threshold,
                  'CE_method': CE_method,
                  'rejected': False,
                  'date': date,
                  'experimental_parameters': params,
                  'time': results['total_time'] if "total_time" in results.keys() else "NA",
                  'anchoring_df': anchoring_df,
                  'probing_df': probing_df,
                  'round_df': round_df,
                  'questionnaire_df': questionnaire_df,
                  'bonus': bonus}
    return output, participant_id


def append_participant_data(processed_data):
    """
     participant_df

     anchoring_df
     - robot_id
     - robot_features
     - robot_protected_attribute

     probing_df:
     - robot_id
     - robot_features
     - robot_protected_attribute
     - robot_has_proxy
     - decision
     - company
     - reliability
     - prediction_time

     round_df
     - robot_id
     - robot_features
     - robot_protected_attribute
     - cf_num
     - cf_with_proxy
     - cf_exclusive_proxy
     - robot_observed_fairness
     - robot_gt_fairness
     - judgement
    """
    participant_df = pd.DataFrame()
    anchoring_df = pd.DataFrame()
    probing_df = pd.DataFrame()
    round_df = pd.DataFrame()
    questionnaire_df = pd.DataFrame()

    for f in processed_data:
        with open(f, 'rb') as r:
            part_results = dill.load(r)
        pid = part_results['participant_id']

        participant_df = pd.concat([participant_df,
                                    pd.DataFrame({'pid': [pid],
                                                  'completion_check': [part_results['completion_check']],
                                                  'attention_check': [part_results['attention_check']],
                                                  'rejected': [part_results['rejected']],
                                                  #'causality_belief': [part_results['causality_belief']],
                                                  'experimental_params': [part_results['experimental_parameters']],
                                                  'proxy': [part_results['proxy']],
                                                  'proxy_strength': [part_results['proxy_strength']],
                                                  'fairness_threshold': [part_results['fairness_threshold']],
                                                  'CE_method': [part_results['CE_method']]})],
                                   axis=0)

        if not part_results['rejected']:
            anchoring_df = pd.concat([anchoring_df,
                                             pd.DataFrame({'pid': [pid]*len(part_results['anchoring_df']['robot_id']),
                                                           'robot_id': part_results['anchoring_df']['robot_id'],
                                                           'robot_protected_attribute': part_results['anchoring_df']['robot_protected_attribute']})],
                                            axis=0)

            probing_df = pd.concat([probing_df,
                                             pd.DataFrame({'pid': [pid]*len(part_results['probing_df']['robot_id']),
                                                           'robot_id': part_results['probing_df']['robot_id'],
                                                           'robot_protected_attribute': part_results['probing_df']['robot_protected_attribute'],
                                                           'robot_has_proxy': part_results['probing_df']['robot_has_proxy'],
                                                           'company': part_results['probing_df']['company'],
                                                           'reliability': part_results['probing_df']['reliability'],
                                                           'prediction_time': part_results['probing_df']['prediction_time']})],
                                            axis=0)


            round_df = pd.concat([round_df,
                                  pd.DataFrame({'pid': [pid]*len(part_results['round_df']['robot_id']),
                                                'robot_id': part_results['round_df']['robot_id'],
                                                'robot_protected_attribute': part_results['round_df']['robot_protected_attribute'],
                                                'proxy_reliability': part_results['round_df']['proxy_reliability'],
                                                'cf_num': part_results['round_df']['cf_num'],
                                                'cf_with_proxy': part_results['round_df']['cf_with_proxy'],
                                                'cf_exclusive_proxy': part_results['round_df']['cf_exclusive_proxy'],
                                                'robot_observed_fairness': part_results['round_df']['robot_observed_fairness'],
                                                'robot_gt_fairness': part_results['round_df']['robot_gt_fairness'],
                                                'robot_user_delta': part_results['round_df']['robot_user_delta'],
                                                'robot_belief_delta': part_results['round_df']['robot_belief_delta'],
                                                'robot_delta': part_results['round_df']['robot_delta'],
                                                'judgement': part_results['round_df']['judgement'],
                                                'prediction_time': part_results['round_df']['prediction_time']})],
                                    axis=0)

            questionnaire_df = pd.concat([questionnaire_df,
                                          pd.DataFrame({'pid': [pid]*len(part_results['questionnaire_df']['question_id']),
                                                        'question_id': part_results['questionnaire_df']['question_id'],
                                                        'question_text': part_results['questionnaire_df']['question_text'],
                                                        'question_type': part_results['questionnaire_df']['question_type'],
                                                        'response': part_results['questionnaire_df']['response']})],
                                         axis=0)

    output = {'participants_df': participant_df,
              'anchoring_df': anchoring_df,
              'probing_df': probing_df,
              'round_df': round_df,
              'questionnaire_df': questionnaire_df
              }

    return output