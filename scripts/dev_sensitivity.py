import numpy as np
import pandas as pd
import src.paths as paths
from itertools import product
from typing import Dict, List, Optional
from dataclasses import dataclass
from src.sensitivity import check_sensitivity_mip_solution, build_sensitivity_mip_scipopt


@dataclass
class DiscreteCausalDistribution:
    """A dataclass representing discrete causal distributions with variables Y, X, A, B"""

    _states_df: pd.DataFrame
    n_features: int
    n_proxy: int
    n_protected: int
    n_labels: int

    def __post_init__(self):
        """Convert dictionary data into a DataFrame for easier manipulation"""
        self._states_df.rename(columns={'p': 'p_yxab', 'q': 'q_yxab', 'r': 'r_yxab'}, inplace=True)
        self.n_features = len([k for k in self._states_df.keys() if 'X' in k])
        self.state_columns = ['Y', 'A', 'B'] + [f'X{i}' for i in range(1, self.n_features + 1)]
        self.__check_rep__()

    def __check_rep__(self):
        """Check if the distribution is valid"""
        # For each fixed Y,X,B combination, should have exactly 2 values of A
        yxb_groups = self._states_df.groupby(['Y', 'B'] + [f'X{i}' for i in range(1, self.n_features + 1)])
        assert all(len(group) == 2 for _, group in yxb_groups), "Each Y,X,B combination should have exactly 2 values of A"

        # For each fixed Y,X,A combination, should have exactly 2 values of B
        yxa_groups = self._states_df.groupby(['Y', 'A'] + [f'X{i}' for i in range(1, self.n_features + 1)])
        assert all(len(group) == 2 for _, group in yxa_groups), "Each Y,X,A combination should have exactly 2 values of B"

        # For each fixed X,A,B combination, should have exactly 2 values of Y
        xab_groups = self._states_df.groupby([f'X{i}' for i in range(1, self.n_features + 1)] + ['A', 'B'])
        assert all(len(group) == 2 for _, group in xab_groups), "Each X,A,B combination should have exactly 2 values of Y"

        # Validate probabilities sum to 1
        for dist in ['p_yxab', 'q_yxab', 'r_yxab']:
            total = self._states_df[dist].sum()
            if not np.isclose(total, 1.0, atol=1e-10):
                raise ValueError(f"{dist} does not sum to 1 (sum = {total})")

        return True

    def get_index_of(self, states: np.ndarray) -> List[int]:
        """Returns the index of the row in the DataFrame that matches the given state"""
        return [self._states_df[self.state_columns].eq(state).all(axis=1).idxmax()
                for state in states]

    def index_proxy(self, b: int) -> List[int]:
        """Returns indices of rows where proxy B equals given value"""
        return self._states_df[self._states_df['B'] == b].index.tolist()

    def change_protected(self, a: int) -> np.ndarray:
        """Returns distribution vector when we intervene on protected attribute A"""
        if a not in self._states_df['A'].unique():
            raise ValueError(f"Invalid value for A: {a}")

        intervention_dist = self._states_df.copy()
        intervention_dist = intervention_dist.loc[intervention_dist['A'] != a, 'r_yxab']

        return intervention_dist['r_yxab'].values

    def change_proxy(self, b: int) -> np.ndarray:
        """Returns distribution vector when we intervene on protected attribute A"""
        if b not in self._states_df['B'].unique():
            raise ValueError(f"Invalid value for B: {b}")

        intervention_dist = self._states_df.copy()
        intervention_dist = intervention_dist.loc[intervention_dist['B'] != b, 'q_yxab']

        return intervention_dist['q_yxab'].values

    def sample_with(self, dist='joint'):
        """Samples a vector of y from either of the available distributions based on the threshold of 0.5"""
        assert dist in ['joint', 'intervene_protected', 'intervene_proxy'], "Invalid distribution type"

        groups = self._states_df.groupby([f'X{i}' for i in range(1, self.n_features + 1)] + ['A', 'B'])
        if dist == 'joint':
            dist_name = 'p_yxab'
        elif dist == 'intervene_protected':
            dist_name = 'r_yxab'
        else:
            dist_name = 'q_yxab'

        y_values = []
        for _, group in groups:
            y0_prob = group[group['Y'] == 0][dist_name].iloc[0]
            y1_prob = group[group['Y'] == 1][dist_name].iloc[0]
            # Choose Y with higher probability
            y = 1 if y1_prob > y0_prob else 0
            y_values.append(y)

        return np.array(y_values)

    def __repr__(self):
        return self._states_df.to_string()

    @property
    def states(self) -> pd.DataFrame:
        """Returns DataFrame with values of X, Y, A, B"""
        return self._states_df[['Y', 'A', 'B'] + [f'X{i}' for i in range(1, self.n_features + 1)]]

    @property
    def joint(self) -> np.ndarray:
        """Returns vector for p_yxab"""
        return self._states_df['p_yxab'].values

    @property
    def n(self) -> int:
        """Returns number of states"""
        return len(self._states_df)


def solution_to_distribution(solution):
    """
    Transform the solution of the sensitivity MIP into a distribution array.

    :param solution: np.array
    :return: np.array
    """
    # todo: return dataclass
    out = {
        'p_yxab': None,
        'q_yxab': None,
        'r_yxab': None,
        }
    return out


# Create the distribution object
def empty_distribution_df(n_features: int = 3, n_proxy: int = 2, n_protected: int = 2, n_labels: int = 2):
    X = np.array(list(product([0, 1], repeat = n_features)))
    X_names = [f"X{i}" for i in range(1, n_features + 1)]
    dfs = {
        'X': pd.DataFrame(X, columns = X_names),
        'B': pd.DataFrame(np.arange(0, n_proxy), columns = ['B']),
        'A': pd.DataFrame(np.arange(0, n_protected), columns = ['A']),
        'Y': pd.DataFrame(np.arange(0, n_labels), columns = ['Y']),
        }
    df = dfs['X'].merge(dfs['A'], how = 'cross').merge(dfs['B'], how = 'cross').merge(dfs['Y'], how = 'cross')
    df[['p', 'q', 'r']] = np.nan
    return df


def check_distribution_df(df):
    dist_column_names = ['p', 'q', 'r']
    for col in dist_column_names:
        vals = df[col].values
        assert np.greater_equal(vals, 0.0).all(), f"Column {col} is not a distribution"
        assert np.less_equal(vals, 1.0).all(), f"Column {col} is not a distribution"
        assert np.allclose(df[col].sum(), 1.0), f"Column {col} is not a distribution"
    return True


def logit(x):
    return 1 / (1 + np.exp(-x))


def calculate_probabilities():
    # Constants
    P_A = 0.5
    P_X = 0.125
    P_B0_A0, P_B1_A0 = 0.95, 0.05
    P_B0_A1, P_B1_A1 = 0.55, 0.45

    df = empty_distribution_df(n_features=3, n_proxy=2, n_protected=2, n_labels=2)
    df[['p', 'q', 'r']] = 1.0 / len(df)

    df['p'] = df.apply(lambda row: (logit(row.B + row.X1 + row.X2 + row.X3 - 2) if row.Y == 1 else
                                    1 - logit(row.B + row.X1 + row.X2 + row.X3 - 2)) *
                                   (P_B0_A0 if (row.B == 0 and row.A == 0) else
                                    P_B1_A0 if (row.B == 1 and row.A == 0) else
                                    P_B0_A1 if (row.B == 0 and row.A == 1) else
                                    P_B1_A1) * P_X * P_A, axis=1)

    df['q'] = df.apply(lambda row: (logit(row.B + row.X1 + row.X2 + row.X3 - 2) if row.Y == 1 else
                                    1 - logit(row.B + row.X1 + row.X2 + row.X3 - 2)) * P_X * P_A, axis=1)

    df['r'] = df.apply(lambda row: (logit(row.B + row.X1 + row.X2 + row.X3 - 2) if row.Y == 1 else
                                    1 - logit(row.B + row.X1 + row.X2 + row.X3 - 2)) *
                                   (P_B0_A0 if (row.B == 0 and row.A == 0) else
                                    P_B1_A0 if (row.B == 1 and row.A == 0) else
                                    P_B0_A1 if (row.B == 0 and row.A == 1) else
                                    P_B1_A1) * P_X, axis=1)

    return df

c = calculate_probabilities()


def create_perfect_proxy_test():
    """
    Creates a test case where:
    - B is a perfect proxy for A (B = A always)
    - Model has zero error (E_min = E_max = 0)
    - Single feature X
    - All user beliefs are correct

    Expected Solution:
    For p_yxab (base probabilities):
    - When X=0:
        - If A=0,B=0: p_0000=0.25, p_1000=0
        - If A=0,B=1: p_0001=0 (impossible since B=A)
        - If A=1,B=0: p_0010=0 (impossible since B=A)
        - If A=1,B=1: p_0011=0.25, p_1011=0
    - When X=1:
        - If A=0,B=0: p_0100=0, p_1100=0.25
        - If A=0,B=1: p_0101=0 (impossible since B=A)
        - If A=1,B=0: p_0110=0 (impossible since B=A)
        - If A=1,B=1: p_0111=0, p_1111=0.25

    The q_yxab and r_yxab should match p_yxab since intervention
    doesn't change anything in this perfect case.
    """
    # Create the unique states dataframe
    # Format: X1, A, B, Y, where Y is also equal to Yhat
    states = [
        # X=0 cases
        #X  A  B  Y
        [0, 0, 0, 0],
        [0, 1, 0, 0], # prob = 0
        [0, 1, 1, 0],
        [0, 0, 1, 0], # prob = 0
        # X=1 cases
        [1, 0, 0, 1],
        [1, 1, 0, 1], # prob = 0
        [1, 0, 1, 1], # prob = 0
        [1, 1, 1, 1],
    ]

    unique_df = pd.DataFrame(states, columns=['X1', 'A', 'B', 'Y'])
    unique_df['Yhat'] = unique_df['Y'] # Y = Yhat
    unique_df['u'] = 0  # No reliability impact
    unique_df['probing'] = [1,0,1,0,1,0,0,1]
    unique_df['v'] = unique_df['B']  # Believes A=B (correct)

    # Create user decisions dataframe
    # Each row represents a user's assessment of fairness
    user_decisions = []
    for x, a, b, yhat in states:
        if a == b:
            user_decisions.append({
                'X1': x,
                'A': a,
                'B': b,
                'Y': yhat,
                'Yhat': yhat,
                'c': 0,  # Claims fair (correct since B=A always)
                'u': 0,  # No reliability impact (correct Y = X)
                'v': b  # Believes A=B (correct)
            })

    user_df = pd.DataFrame(user_decisions)

    # Expected solution
    # left column is Y=0, right Y=1
    expected_solution = {
        'p': np.array([
            0.25, 0.0,  # X=0,A=0,B=0
            0.0, 0.0,  # X=0,A=0,B=1 (impossible)
            0.0, 0.0,  # X=0,A=1,B=0 (impossible)
            0.25, 0.0,  # X=0,A=1,B=1
            0.0, 0.25,  # X=1,A=0,B=0
            0.0, 0.0,  # X=1,A=0,B=1 (impossible)
            0.0, 0.0,  # X=1,A=1,B=0 (impossible)
            0.0, 0.25  # X=1,A=1,B=1
        ]),
        'q': None,  # Should match p
        'r': None,  # Should match p
        'e': np.zeros(8),  # No errors
        'd': np.zeros(4),  # All fair
        'B': np.zeros(4),  # No absolute value indicators needed
    }
    expected_solution['q'] = expected_solution['p']
    expected_solution['r'] = expected_solution['p']

    return unique_df, user_df, expected_solution


def run_case(unique_df, user_df, n_feats=1, expected_solution=None):
    """
    Runs the sensitivity MIP with the given data.

    Args:
        unique_df: DataFrame with unique states
        user_df: DataFrame with user decisions
        settings_override: Optional dict to override default settings
    """
    # Create empty distribution
    df = empty_distribution_df(n_features=n_feats, n_proxy=2, n_protected=2, n_labels=2)
    df[['p', 'q', 'r']] = 1.0 / len(df)
    assert check_distribution_df(df)

    dist_true = DiscreteCausalDistribution(df, n_features=n_feats, n_proxy=2, n_protected=2, n_labels=2)

    # update user_df with the indices that you will use in the MIP
    cols = dist_true.state_columns  # column names
    # sub Y with Yhat
    cols = [c if c != 'Y' else 'Yhat' for c in cols]

    states = user_df[cols]  # values of ['X','B','A','Y' = 'Yhat']
    states_a = states.copy()
    states_a['A'] = 1 - states['A'] # flip the A value

    indices = dist_true.get_index_of(states.values)  # returns the indices of the states in the distribution
    indices_aprim = dist_true.get_index_of(states_a.values)
    user_df['id'] = indices
    user_df['id_aprim'] = indices_aprim

    cols = dist_true.state_columns  # column names

    states = unique_df[cols]  # values of ['X','B','A','Y']
    states2 = states.copy()
    states2['Y'] = 1 - states['Y']  # flip the Y value

    states_a = states.copy()
    states_a['A'] = 1 - states['A']  # flip the A value

    states_b = states.copy()
    states_b['B'] = 1 - states['B']  # flip the B value

    indices = dist_true.get_index_of(states.values)  # returns the indices of the states in the distribution
    indices_complement = dist_true.get_index_of(states2.values)  # returns the indices of the states in the distribution
    indices_aprim = [unique_df[cols].eq(state).all(axis=1).idxmax() for state in states_a.values]
    indices_bprim = [unique_df[cols].eq(state).all(axis=1).idxmax() for state in states_b.values]
    unique_df['id'] = indices
    unique_df['id_complement'] = indices_complement
    unique_df['id_aprim'] = indices_aprim
    unique_df['id_bprim'] = indices_bprim

    states_user = user_df[cols].values  # values of ['X','B','A','Y']
    indices_user = [unique_df[cols].eq(state).all(axis=1).idxmax() for state in states_user]
    user_df['id_unique'] = indices_user

    # Settings for perfect prediction case
    settings = {
        'dist_true': dist_true,
        'user_df': user_df,
        'unique_df': unique_df,
        'delta': 0.5,
        'E_min': 0,  # Zero error case
        'E_max': 3,
        'epsilon': 0.0001,
        'gamma': 0.0001,
        'M': 2,
    }

    print(unique_df)
    print(user_df)
    print(dist_true)

    # Create SCIP model
    model, info = build_sensitivity_mip_scipopt(**settings)  # This function needs to be rewritten for SCIP

    # Enable conflict analysis
    model.setBoolParam('conflict/enable', True)
    model.setIntParam("presolving/maxrounds", 0)

    # Solve
    model.optimize()

    # If infeasible, analyze conflict
    if model.getStatus() == 'infeasible':
        print("Model is infeasible")

        model.writeProblem("model.lp")

    # Check solution (this function needs to be modified for SCIP)
    is_valid = check_sensitivity_mip_solution(model, info)
    print(f"\nPerfect Proxy Test Results:")
    print(f"Solution valid: {is_valid}")
    print(f"Objective value: {model.getObjVal()}")  # Changed from CPLEX syntax

    # Get solution values
    names = info['names']
    vars_dict = info['vars_dict']
    actual_solution = {
        'p': np.array([round(model.getVal(vars_dict[name]), 5) for name in names['p']]),
        'q': np.array([round(model.getVal(vars_dict[name]), 5) for name in names['q']]),
        'r': np.array([round(model.getVal(vars_dict[name]), 5) for name in names['r']]),
        'e': np.array([model.getVal(vars_dict[name]) for name in names['e']]),
        'd': np.array([model.getVal(vars_dict[name]) for name in names['d']]),
        'B': np.array([model.getVal(vars_dict[name]) for name in names['B']]),
    }

    # Rest of the comparison code stays the same
    if expected_solution:
        for key in expected_solution:
            if expected_solution[key] is not None:
                matches = np.allclose(actual_solution[key], expected_solution[key])
                print(f"\n{key} matches expected: {matches}")
                if not matches:
                    print("Expected:")
                    print(expected_solution[key])
                    print("Actual:")
                    print(actual_solution[key])

    dist_true._states_df['p_yxab'] = actual_solution['p']
    dist_true._states_df['q_yxab'] = actual_solution['q']
    dist_true._states_df['r_yxab'] = actual_solution['r']
    # add new column "seen" that will add True to whatever is in user_df (based on the id)
    dist_true._states_df['seen'] = dist_true._states_df.index.isin(user_df['id'].values)

    print(dist_true)

    return model, info, expected_solution


# unique_df, user_df, expected_solution = create_perfect_proxy_test()
# nfeats = len([k for k in unique_df.columns if 'X' in k])
# model, info, expected_solution = run_case(unique_df, user_df, n_feats=nfeats, expected_solution=expected_solution)

def read_participant_case(pid):
    """
    Read participant's answers
    """
    user_df = pd.read_excel(paths.reports_dir / f'mip_{pid}.xlsx', sheet_name='mip', dtype=int, index_col=0)
    unique_df = pd.read_excel(paths.reports_dir / f'mip_{pid}.xlsx', sheet_name='unique', dtype=int, index_col=0)

    return unique_df, user_df

unique_df, user_df = read_participant_case('5e3eb0bcd3d69e10901f0cb3')
nfeats = len([k for k in unique_df.columns if 'X' in k])
model, info, _ = run_case(unique_df, user_df, n_feats=nfeats)


# # check the solution (minimal internal consistency checks)
# assert check_sensitivity_mip_solution(cpx, info)
#
#
#
#
# # transform solution back to distribution object
# solution = cpx.solution.get_values()
# user_dist = solution_to_distribution(solution)
# # check that your solution matches the ground truth
# assert check_distribution(user_dist)