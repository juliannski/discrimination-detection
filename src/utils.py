import math
import decimal


def power_set(numbers):
    """
    Output a list of subsets of a list

    :param numbers: list of <Any>
    :return: list of all subsets within the input
    """
    if numbers == []:
        return [[]]
    x = power_set(numbers[1:])
    return x + [[numbers[0]] + y for y in x]


def compute_model_info(model_string, type):
    if type in ['score_function', 'explanations']:
        prediction_score, _, score_body = model_string.split('\n')
        prediction_body, score = score_body.split('>=')
        elements = prediction_body.split(' + ')
        conditions = {}
        for i, el in enumerate(elements):
            els = el.split('*')
            if len(els) == 1:
                feature_value = els[0]
                score_num = '1'
            else:
                feature_value = els[1]
                score_num = els[0]

            feature, value = feature_value.split('=')

            if score_num[0] == ' ':
                score_num = score_num[1:]

            conditions[feature.strip()] =  (value.strip(), score_num)
        user_model_info = {'model_type': type,
                           'conditions': conditions,
                           'threshold': score.strip()}
    elif type == 'boolean':
        user_model_info = {'model_type': type, 'model_string': model_string}
    else:
        user_model_info = {}
    return user_model_info


def linear_model_to_shap(model_info, features, random_state):
    """
    Converts a linear model to a SHAP explanation
    :param model_info: the model information
    :param features: the features
    :return: list of triples (feature, value, shap value), mean value
    """
    conditions=model_info['conditions']
    shap_feature_importance = []
    mean_val = 0
    max_neg = 0
    random_noises = list(random_state.uniform(0, 0.5, size=len(conditions)).round(3))
    for feature in features:
        value, coeff = conditions[feature] if feature in conditions else (None, None)
        is_present = features.get(feature, None) == value
        rval = features.get(feature, None)

        if value is None:
            shap_feature_importance.append((feature, rval, 0))
            continue

        # 0.5 is the mean value of every condition on a feature in the roboto catalog, since all features are binary
        if is_present:
            shap_val = int(coeff) * (1 - 0.5)
        else:
            shap_val = int(coeff) * (0 - 0.5)
        random_noise = random_noises.pop()
        max_neg -= int(coeff)
        print(f"{feature}: {shap_val}")
        # we can add to positive shap values to even strengthen the prediction
        # or subtract from negative shap values to weaken the prediction
        if shap_val > 0:
            shap_val += random_noise
        else:
            shap_val -= random_noise
        shap_feature_importance.append((feature, rval, shap_val))
        mean_val += 0.5 * int(coeff)

    mean_val -= int(model_info['threshold'])

    return {'feature_importance': shap_feature_importance, "mean_val": mean_val, "max_negative_shap_sum": max_neg}


def binary_to_number(point):
    """
    Turn a len(point)-bit representations of a number into the number

    :param point: len(point)-bit name of number n; list of dict
    :return: n
    """
    if isinstance(point, dict):
        point = list(point.values())
    binary_rep = sum([2 ** ((e + 1) * i) for e, i in enumerate(point)])
    return binary_rep


def find_exp(number) -> int:
    """
    To get the power of 10 in the scientific notation for 'number'

    :param number: any integer
    :return: k, where a * 10^k = number
    """
    base10 = math.log10(abs(number))
    return abs(math.floor(base10))


def abs_list(list_):
    """
    Absolute value for each elem in a list

    :param list_:
    :return:
    """
    return [abs(el) for el in list_]


def frange(start, stop, step=0.1):
    while start < stop:
        yield float(start)
        start += decimal.Decimal(str(step))


def cond_probs(cf_fairness, precision=0.1):
    """
    Generate all possible pairs of distributions that result in a given CF-fairness including the ordering of
    distributions.

    :param cf_fairness: delta for prob distributions
    :param precision: how granular should different tested prob values be
    :return:
    """
    distributions = []
    big = 0
    round_to = abs(find_exp(precision))
    while big < 1.0:
        small = big - cf_fairness
        small = math.copysign(round(abs(big-cf_fairness), round_to), small)
        if small <= 0.0:
            big += precision
            big = round(big, round_to)
            continue
        d1 = [[big, round(1-big, round_to)], [small, round(1-small, round_to)]]
        d2 = [[round(1-big, round_to), big], [round(1-small, round_to), small]]
        distributions.extend([d1, d2])
        big += precision
        big = round(big, round_to)
    return distributions
