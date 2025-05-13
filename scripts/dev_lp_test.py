import numpy as np
from src.classifier import INTERCEPT_NAME, LinearClassificationModel, DistributionClassificationModel
from src.dag import Variable, FinalBooleanVariable, FairnessDAG
from src.robot_catalog import RobotCatalog

def transform_parameters_to_arrays(dict_of_p_yxab, dict_of_q_yxab, dict_of_r_yxab):
    """
    Transform p_yxab variables that stand for joint distribution P(Y=y, X=x, A=a, B=b)
    into an array with as many axes as there are variables so that the array is indexed
    by [y, x, a, b].

    Do the same for q_yxab and r_yxab.

    :param dict_of_p_yxab: {'yxab': p_yxab (float)}
    :param dict_of_q_yxab:
    :param dict_of_r_yxab:
    :return: np.array, np.array, np.array
    """
    #find the largest possible value of y
    yxab_codes = dict_of_p_yxab.keys()
    ymax = max([int(code[0]) for code in yxab_codes])
    xmax = max([int(code[1]) for code in yxab_codes])
    amax = max([int(code[2]) for code in yxab_codes])
    bmax = max([int(code[3]) for code in yxab_codes])

    # create empty array
    p_yxab = np.zeros((ymax+1, xmax+1, amax+1, bmax+1))
    q_yxab = np.zeros((ymax+1, xmax+1, amax+1, bmax+1))
    r_yxab = np.zeros((ymax+1, xmax+1, amax+1, bmax+1))
    for code, pyxab in dict_of_p_yxab.items():
        y,x,a,b = code
        y,x,a,b = int(y), int(x), int(a), int(b)
        p_yxab[y,x,a,b] = pyxab
    for code, qyxab in dict_of_q_yxab.items():
        y,x,a,b = code
        y,x,a,b = int(y), int(x), int(a), int(b)
        q_yxab[y,x,a,b] = qyxab
    for code, ryxab in dict_of_r_yxab.items():
        y,x,a,b = code
        y,x,a,b = int(y), int(x), int(a), int(b)
        r_yxab[y,x,a,b] = ryxab

    return p_yxab, q_yxab, r_yxab


def check_solution(p_yxab, q_yxab, r_yxab):
    # assuming all variables are binary

    # test P(Y)
    p_y = np.sum(p_yxab, axis=(1, 2, 3))
    q_y = np.sum(q_yxab, axis=(1, 2, 3))
    r_y = np.sum(r_yxab, axis=(1, 2, 3))
    assert np.allclose(np.sum(p_y), 1.0), "P(Y) is not 1; p variables"
    assert np.allclose(np.sum(q_y), 1.0), "P(Y) is not 1; q variables"
    assert np.allclose(np.sum(r_y), 1.0), "P(Y) is not 1; r variables"

    # test P(X)
    p_x = np.sum(p_yxab, axis=(0, 2, 3))
    q_x = np.sum(q_yxab, axis=(0, 2, 3))
    r_x = np.sum(r_yxab, axis=(0, 2, 3))
    assert np.allclose(np.sum(p_x), 1.0), "P(X) is not 1; p variables"
    assert np.allclose(np.sum(q_x), 1.0), "P(X) is not 1; q variables"
    assert np.allclose(np.sum(r_x), 1.0), "P(X) is not 1; r variables"

    # test P(B)
    p_b = np.sum(p_yxab, axis=(0, 1, 2))
    q_b = np.sum(q_yxab, axis=(0, 1, 2))
    r_b = np.sum(r_yxab, axis=(0, 1, 2))
    assert np.allclose(np.sum(p_b), 1.0), "P(B) is not 1; p variables"
    assert np.allclose(np.sum(q_b), 1.0), "P(B) is not 1; q variables"
    assert np.allclose(np.sum(r_b), 1.0), "P(B) is not 1; r variables"

    # test P(A)
    p_a = np.sum(p_yxab, axis=(0, 1, 3))
    q_a = np.sum(q_yxab, axis=(0, 1, 3))
    r_a = np.sum(r_yxab, axis=(0, 1, 3))
    assert np.allclose(np.sum(p_a), 1.0), "P(A) is not 1; p variables"
    assert np.allclose(np.sum(q_a), 1.0), "P(A) is not 1; q variables"
    assert np.allclose(np.sum(r_a), 1.0), "P(A) is not 1; r variables"

    # test independence B and X
    p_b_cond_x = np.sum(p_yxab, axis=(1, 2)) / p_x
    p_x_cond_b = np.sum(p_yxab, axis=(0, 2)) / p_b
    assert np.allclose(p_b_cond_x, p_b*p_x), "B and X are not independent"
    assert np.allclose(p_x_cond_b, p_b*p_x), "B and X are not independent"

    # test independence A and X
    p_a_cond_x = np.sum(p_yxab, axis=(1, 3)) / p_x
    p_x_cond_a = np.sum(p_yxab, axis=(0, 3)) / p_a
    assert np.allclose(p_a_cond_x, p_a * p_x), "B and X are not independent"
    assert np.allclose(p_x_cond_a, p_a * p_x), "B and X are not independent"


    # test if data is correct to create a DAG
    p_b_cond_a = np.sum(p_yxab, axis=(1, 2)) / p_a
    assert np.allclose(np.sum(p_b_cond_a[0]), 1.0), "P(B|A) is not 1"
    assert np.allclose(np.sum(p_b_cond_a[1]), 1.0), "P(B|A) is not 1"

    p_yxb = np.sum(p_yxab, axis=2)
    p_xb = np.sum(p_yxb, axis=0)

    p_y_cond_bx = p_yxb / p_xb
    assert np.allclose(np.sum(p_y_cond_bx[0][0]), 1.0), "P(Y|B,X) is not 1"
    assert np.allclose(np.sum(p_y_cond_bx[0][1]), 1.0), "P(Y|B,X) is not 1"
    assert np.allclose(np.sum(p_y_cond_bx[1][0]), 1.0), "P(Y|B,X) is not 1"
    assert np.allclose(np.sum(p_y_cond_bx[1][1]), 1.0), "P(Y|B,X) is not 1"

    DELTA = 0.05
    NROBOTS = 64
    ACC  = 1.0
    yhat_func = {INTERCEPT_NAME: -0.24,  # linear model parameters for the predicted Y_hat(x) = logit(yhat_func(x))
                 'antenna': 0.36,
                 'head': 0.24}

    y_func = DistributionClassificationModel(p_y_cond_bx)

    random_state = np.random.RandomState(seed=123)
    A = Variable(name='A', values=[0, 1], probabilities=[p_a[0], p_a[1]])
    B = Variable(name='proxy', values=[0, 1],
                 probabilities=dict({'A_0': [p_b_cond_a[0][0], p_b_cond_a[0][1]],
                                     'A_1': [p_b_cond_a[1][0], p_b_cond_a[1][1]]}))
    X = {0: Variable(name=0, values=[0, 1], probabilities=[p_x[0], p_x[1]])}
    Y = FinalBooleanVariable(name='Y', vars_ordering=[B.name] + [x.name for x in X.values()],
                             model_parameters=y_func)
    dag = FairnessDAG(features=X, outcome=Y, protected=A, proxy=B)
    dag.add_edges_from_coefficients(coefficients=yhat_func)

    # Define the predictor
    feature_ordering = dag.feature_names
    yhat_func = {k: yhat_func[k] for k in feature_ordering + [INTERCEPT_NAME]}
    model = LinearClassificationModel.from_dict(**yhat_func)

    # Create the robot catalog
    # Generate a giant data frame of attributes, robots, likelihoods, true labels
    catalog = RobotCatalog(dag=dag, model=model, random_state=random_state)
    catalog.generate(n=dag.n_distinct, with_replacement=False)
    catalog.adapt(to_size=NROBOTS)
    catalog.draw_fairness(threshold=DELTA)

    assert catalog.df['delta'] <= DELTA, f"Delta is too high, look: {catalog.df['delta']}"

    print(catalog.df.to_string())
    print(f"Accuracy: {1 - (catalog.df['Y'] - catalog.df['yhat']).mean()}")
    acc = 1 - (catalog.df['Y'] - catalog.df['yhat']).mean()
    assert acc >= ACC, "Model's accuracy too low"