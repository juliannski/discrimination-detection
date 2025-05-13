import numpy as np
from cplex import Cplex, SparsePair, SparseTriple
from src.cpx.utils import add_variable, set_mip_time_limit, StatsCallback, get_mip_stats, set_mip_max_gap
from pyscipopt import Model, quicksum


def build_sensitivity_mip_scipopt(**settings):
    """
        :param settings:
        :return:
        --
        variable vector = [
        p_{yxab},
        q_{yxab},
        r_{yxab},
        e_{xab},
        B_{xb},
        d_{xb},
        ]
        --
        ----------------------------------------------------------------------------------------------------------------
        name                 length                         type        description
        ----------------------------------------------------------------------------------------------------------------
        p_{yxab}:            |Y| x |X| x |A| x |B|          real        P(Y=y, X=x, A=a, B=b) - base joint probability
        q_{yxab}:            |Y| x |X| x |A| x |B|          real        P(Y=y, X=x, A=a, do(B=b)) - intervened probability on B
        r_{yxab}:            |Y| x |X| x |A| x |B|          real        P(Y=y, X=x, do(A=a), B=b) - intervened probability on A
        e_{xab}:             |X| x |A| x |B|                binary      error indicator: e_{xab} = 1 if p_{h(b,x)xab} > p_{h(b,x)'xab}
        B_{xb}:              |X| x |B|                      binary      absolute value indicator |p|xab - r|xab| > delta: if B_{xb} = 1
                                                                        then p|xab - r|xab > delta, otherwise p|xab - r|xab < - delta, for
                                                                        a = v_{xb}, y = h(x,b)
        d_{xb}:              |X| x |B|                      binary      discrimination indicator: if d_{xb} = 1 then |p|xab-r|xab| > delta

        Key dimensions:
        |Y| = 2 (binary prediction)
        |X| = 8 (number of unique features)
        |A| = 2 (number of companies)
        |B| = 2 (binary proxy variable)

        Vector parameters
        u_{xb} ∈ {-1,0,1} (user beliefs about proxy's effect on reliability)
        v_{xb} ∈ {0,1} (user beliefs about robot's protected attribute)
        c_{xb} ∈ {0,1} (user beliefs about prediction's discrimination)

        Parameters:
        E^{max} = 2 (model error)
        M = 2 (cap on difference in distributions)
        ε ∈ (0,1] (reliability change slack variable)
        γ ∈ (0,1] (difference in distributions slack variable)
        δ (counterfactual fairness bound)
        """
    # Parameter checks remain the same
    assert settings['E_min'] <= settings['E_max'] <= 16
    assert 0 < settings['epsilon'] <= 1
    assert 0 < settings['gamma'] <= 1
    assert 0 < settings['delta'] < 1
    assert settings['M'] >= 2

    dist_true = settings['dist_true']
    user_df = settings['user_df']
    unique_df = settings['unique_df']

    # Same dimensions
    n_yxab = dist_true._states_df.shape[0]
    n_xab = unique_df.shape[0]
    n_responses = len(user_df)

    # Create model
    model = Model()

    # Variable naming remains the same
    print_vnames = lambda vfmt, vcnt: list(map(lambda v: vfmt % v, range(vcnt)))
    var_name_fmt = {
        'p': 'p_%d',
        'q': 'q_%d',
        'r': 'r_%d',
        'ry': 'ry_%d',
        'py': 'py_%d',
        'D': 'D_%d',
        'd': 'd_%d',
        'B': 'B_%d',
        'e': 'e_%d',
        'pi': 'pi_%d',
        'phi': 'phi_%d',
    }

    names = {
        'p': print_vnames(var_name_fmt['p'], n_yxab),
        'q': print_vnames(var_name_fmt['q'], n_yxab),
        'r': print_vnames(var_name_fmt['r'], n_yxab),
        'D': print_vnames(var_name_fmt['D'], n_responses),
        'd': print_vnames(var_name_fmt['d'], n_responses),
        'B': print_vnames(var_name_fmt['B'], n_responses),
        'e': print_vnames(var_name_fmt['e'], n_xab),
        'pi': print_vnames(var_name_fmt['pi'], n_xab),
        'phi': print_vnames(var_name_fmt['phi'], n_xab),
        'ry': print_vnames(var_name_fmt['ry'], n_xab),
        'py': print_vnames(var_name_fmt['py'], n_xab),
        'n_fp': 'n_fp',
        'n_fn': 'n_fn',
    }

    # Create variables (SCIP style)
    vars_dict = {}
    # Continuous variables
    for name_list in [names['p'], names['q'], names['r'], names['D'],
                      names['pi'], names['phi'], names['ry'], names['py']]:
        for name in name_list:
            vars_dict[name] = model.addVar(name=name, vtype="C", lb=0 if 'D' in name else 0.00001, ub=1.0)

    # Binary variables
    for name_list in [names['d'], names['B'], names['e']]:
        for name in name_list:
            vars_dict[name] = model.addVar(name=name, vtype="B", lb=0.0, ub=1.0)

    vars_dict[names['n_fp']] = model.addVar(name=names['n_fp'], vtype="I", lb=0.0, ub=None, obj=1.0)
    vars_dict[names['n_fn']] = model.addVar(name=names['n_fn'], vtype="I", lb=0.0, ub=None, obj=1.0)

    # User False Alarms and Missed Cases constraints
    d_c1_vars = [vars_dict[n] for i, n in enumerate(names['d']) if user_df.iloc[i]['c'] == 1]
    d_c0_vars = [vars_dict[n] for i, n in enumerate(names['d']) if user_df.iloc[i]['c'] == 0]

    model.addCons(vars_dict[names['n_fp']] == quicksum(1 - d for d in d_c1_vars))
    model.addCons(vars_dict[names['n_fn']] == quicksum(d_c0_vars))

    # Ground Truth Fairness constraints
    for ry in names['ry']:
        ry_ind = int(ry.split('_')[1])
        r_id = unique_df.loc[ry_ind]['id']
        py = names['py'][ry_ind]
        r = names['r'][r_id]
        p = names['p'][r_id]
        phi = names['phi'][ry_ind]
        pi = names['pi'][ry_ind]

        # Quadratic constraint r_{yaxb} = ry_{axb} * phi_{xab}
        model.addCons(vars_dict[r] == vars_dict[ry] * vars_dict[phi])

        # p[yxab_idx] == ry * pival for ratio p[yxab_idx]/pi and r[yxab_idx]/phi == ry consistency
        model.addCons(vars_dict[p] == vars_dict[py] * vars_dict[pi])

        # ry = py
        model.addCons(vars_dict[ry] == vars_dict[py])


    for d, D, B in zip(names['d'], names['D'], names['B']):
        D_ind = int(D.split('_')[1])
        ry_id = user_df.loc[D_ind]['id_unique']
        ryprim_id = unique_df.loc[ry_id]['id_aprim']
        ry = names['ry'][ry_id]
        ryprim = names['ry'][ryprim_id]

        # D = ry_axb - ry_a'xb
        model.addCons(vars_dict[D] == vars_dict[ry] - vars_dict[ryprim])

        # For indicator constraints in SCIP:
        # if x = 0 then ax <= b becomes:
        # model.addConsIndicator(cons=ax <= b, binvar=x, val=False)
        # if x = 1 then ax <= b becomes:
        # model.addConsIndicator(cons=ax <= b, binvar=x, val=True)

        # complemented=1 means when d=0
        # complemented=0 means when d=1

        # d_fair1: if d=0 then D <= delta
        model.addConsIndicator(
            cons=vars_dict[D] <= settings['delta'],
            binvar=vars_dict[d],
            activeone=False
        )

        # d_fair2: if d=0 then D >= -delta
        model.addConsIndicator(
            cons=vars_dict[D] >= -settings['delta'],
            binvar=vars_dict[d],
            activeone=False
        )

        # d_unfair1: if d=1 then D + M*B >= delta + gamma
        model.addConsIndicator(
            cons=vars_dict[D] + settings['M'] * vars_dict[B] >= settings['delta'] + settings['gamma'],
            binvar=vars_dict[d],
            activeone=True
        )

        # d_unfair2: if d=1 then -D + M*B >= delta + gamma - M
        model.addConsIndicator(
            cons=-vars_dict[D] - settings['M'] * vars_dict[B] >= settings['delta'] + settings['gamma'] - settings['M'],
            binvar=vars_dict[d],
            activeone=True
        )

    # Model Error constraints
    e_vars = [vars_dict[e] for e in names['e']]
    model.addCons(quicksum(e_vars) <= 2 * settings['E_max'])
    model.addCons(quicksum(e_vars) >= 2 * settings['E_min'])

    for e in names['e']:
        e_ind = int(e.split('_')[1])
        yxab_id = unique_df.loc[e_ind]['id']
        yxaprimb_id = unique_df.loc[unique_df.loc[e_ind]['id_aprim']]['id']
        yprimxab_id = unique_df.loc[e_ind]['id_complement']
        yprimxaprimb_id = unique_df.loc[unique_df.loc[e_ind]['id_aprim']]['id_complement']

        yhat = unique_df.loc[e_ind]['Yhat']
        y = unique_df.loc[e_ind]['Y']

        if y == yhat:
            p = names['p'][yxab_id]
            paprim = names['p'][yxaprimb_id]
            pprim = names['p'][yprimxab_id]
            pprimaprim = names['p'][yprimxaprimb_id]
        else:
            p = names['p'][yprimxab_id]
            paprim = names['p'][yprimxaprimb_id]
            pprim = names['p'][yxab_id]
            pprimaprim = names['p'][yxaprimb_id]

        # if e=0 then p - pprim >= gamma
        model.addConsIndicator(
            cons=vars_dict[p] - vars_dict[pprim] >= settings['gamma'],
            binvar=vars_dict[e],
            activeone=False
        )

        # if e=0 then paprim - pprimaprim >= gamma
        model.addConsIndicator(
            cons=vars_dict[paprim] - vars_dict[pprimaprim] >= settings['gamma'],
            binvar=vars_dict[e],
            activeone=False
        )

        # if e=1 then pprim - p >= gamma
        model.addConsIndicator(
            cons=vars_dict[pprim] - vars_dict[p] >= settings['gamma'],
            binvar=vars_dict[e],
            activeone=True
        )

        # if e=1 then pprimaprim - paprim >= gamma
        model.addConsIndicator(
            cons=vars_dict[pprimaprim] - vars_dict[paprim] >= settings['gamma'],
            binvar=vars_dict[e],
            activeone=True
        )

    # Protected Attribute Beliefs constraints
    for pi in names['pi']:
        pi_ind = int(pi.split('_')[1])
        xab_id = unique_df.loc[pi_ind]['id']
        xab_complement_id = unique_df.loc[pi_ind]['id_complement']
        pi_xaprimb_id = unique_df.loc[pi_ind]['id_aprim']
        p = names['p'][xab_id]
        pi_complement = names['pi'][pi_xaprimb_id]
        pprim = names['p'][xab_complement_id]

        # pi = p + pprim
        model.addCons(
            vars_dict[pi] == vars_dict[p] + vars_dict[pprim],
            name=f'robot_probability_constraint_{pi_ind}'
        )

        if unique_df.loc[pi_ind]['probing'] and unique_df.loc[pi_ind]['v'] in [0,1]:

            # pi >= pi_complement
            model.addCons(
                vars_dict[pi] >= vars_dict[pi_complement],
                name=f'protected_attribute_constraint_{pi_ind}'
            )

    # Protected Attribute Intervention Beliefs auxiliary constraints
    for phi in names['phi']:
        phi_ind = int(phi.split('_')[1])
        xab_id = unique_df.loc[phi_ind]['id']
        xab_complement_id = unique_df.loc[phi_ind]['id_complement']
        r = names['r'][xab_id]
        rprim = names['r'][xab_complement_id]

        # phi = r + rprim
        model.addCons(
            vars_dict[phi] == vars_dict[r] + vars_dict[rprim],
            name=f'robot_probability_constraint_{phi_ind}'
        )

    # Reliability constraints: u == 1, y = 1
    q_improvement_indices = [row['id'] if row['Y'] == 1 else row['id_complement']
                             for i, row in unique_df.iterrows() if row['u'] == 1 and row['B'] == 1]
    p_improvement_indices = [row['id'] if row['Y'] == 1 else row['id_complement']
                             for i, row in unique_df.iterrows() if row['u'] == 1 and row['B'] == 0]
    for i, (qind, pind) in enumerate(zip(q_improvement_indices, p_improvement_indices)):
        q = names['q'][qind]
        p = names['p'][pind]

        # q - p >= epsilon
        model.addCons(
            vars_dict[q] - vars_dict[p] >= settings['epsilon'],
            name=f"reliability_improvement_{i}"
        )

    # Decrease in Reliability constraints: u == -1, y = 1
    q_decrease_indices = [row['id'] if row['Y'] == 1 else row['id_complement']
                          for i, row in unique_df.iterrows() if row['u'] == -1 and row['B'] == 0]
    p_decrease_indices = [row['id'] if row['Y'] == 1 else row['id_complement']
                          for i, row in unique_df.iterrows() if row['u'] == -1 and row['B'] == 1]
    for i, (qind, pind) in enumerate(zip(q_decrease_indices, p_decrease_indices)):
        q = names['q'][qind]
        p = names['p'][pind]

        # p - q <= epsilon
        model.addCons(
            vars_dict[p] - vars_dict[q] <= settings['epsilon'],
            name=f"reliability_decrease_{i}"
        )

    # Const Reliability constraints: u == 0
    const_b0_indices = [row['id'] if row['Y'] == 1 else row['id_complement']
                        for i, row in unique_df.iterrows() if row['u'] == 0 and row['B'] == 0]
    const_b1_indices = [row['id'] if row['Y'] == 1 else row['id_complement']
                        for i, row in unique_df.iterrows() if row['u'] == 0 and row['B'] == 1]
    for i, (bzeroind, boneind) in enumerate(zip(const_b0_indices, const_b1_indices)):
        pzero = names['p'][bzeroind]
        qzero = names['q'][bzeroind]
        pone = names['p'][boneind]
        qone = names['q'][boneind]

        # pzero = qzero
        model.addCons(
            vars_dict[pzero] == vars_dict[qzero],
            name=f"const_reliability1_{i}"
        )

        # pone = qone
        model.addCons(
            vars_dict[pone] == vars_dict[qone],
            name=f"const_reliability2_{i}"
        )

    # Causal Structure constraints
    seen_ax_combinations, seen_bx_combinations = set(), set()
    seen_ax, seen_bx = 0, 0
    for p, r in zip(names['p'], names['r']):
        p_ind = int(p.split('_')[1])
        r_ind = int(r.split('_')[1])
        # Find corresponding row logic
        row = unique_df[unique_df['id'] == p_ind]
        if row.empty:
            row = unique_df[unique_df['id_complement'] == p_ind]
        pi_phi_ind = row.index[0]
        pi = names['pi'][pi_phi_ind]
        phi = names['phi'][pi_phi_ind]

        # P(Y, X, A, B) <= P(Y, X, do(A), B)
        model.addCons(vars_dict[p] <= vars_dict[r])

        # Quadratic constraint p*phi = r*pi
        model.addCons(
            vars_dict[p] * vars_dict[phi] == vars_dict[r] * vars_dict[pi],
            name=f"causal_structure_{p_ind}"
        )

        aval = int(dist_true._states_df.iloc[p_ind]['A'])
        bval = int(dist_true._states_df.iloc[p_ind]['B'])
        xval = [int(dist_true._states_df.iloc[p_ind][k]) for k in dist_true._states_df.columns if 'X' in k]
        ax_vals = [aval] + xval
        bx_vals = [bval] + xval

        if tuple(ax_vals) not in seen_ax_combinations:
            # add the constraint \sum_y,b p_yxab = sum_y,b r_yxab
            # find ps and rs that we should sum over where A=a, X=x
            Xcols = [k for k in dist_true._states_df.columns if 'X' in k]
            ps = [names['p'][i] for i, row in dist_true._states_df.iterrows()
                  if all(row[k] == v for k, v in zip(Xcols, xval))]
            rs = [names['r'][i] for i, row in dist_true._states_df.iterrows()
                  if row['A'] == aval and all(row[k] == v for k, v in zip(Xcols, xval))]

            # sum(ps) = sum(rs)
            model.addCons(
                quicksum(vars_dict[p] for p in ps) == quicksum(vars_dict[r] for r in rs),
                name=f"independence_ax_structure_{seen_ax}"
            )
            seen_ax_combinations.add(tuple(ax_vals))
            seen_ax += 1

        if tuple(bx_vals) not in seen_bx_combinations:
            Xcols = [k for k in dist_true._states_df.columns if 'X' in k]
            ps = [names['p'][i] for i, row in dist_true._states_df.iterrows()
                  if all(row[k] == v for k, v in zip(Xcols, xval))]
            qs = [names['q'][i] for i, row in dist_true._states_df.iterrows()
                  if row['B'] == bval and all(row[k] == v for k, v in zip(Xcols, xval))]

            # sum(ps) = sum(qs)
            model.addCons(
                quicksum(vars_dict[p] for p in ps) == quicksum(vars_dict[q] for q in qs),
                name=f"independence_bx_structure_{seen_bx}"
            )
            seen_bx_combinations.add(tuple(bx_vals))
            seen_bx += 1

    # Probability constraints
    model.addCons(
        quicksum(vars_dict[p] for p in names['p']) == 1.0,
        name='p_prob'
    )

    b0_indices = [i for i, row in dist_true._states_df.iterrows() if row['B'] == 0]
    b1_indices = [i for i, row in dist_true._states_df.iterrows() if row['B'] == 1]

    model.addCons(
        quicksum(vars_dict[q] for i, q in enumerate(names['q']) if i in b0_indices) == 1.0,
        name='q_prob_b0'
    )

    model.addCons(
        quicksum(vars_dict[q] for i, q in enumerate(names['q']) if i in b1_indices) == 1.0,
        name='q_prob_b1'
    )

    a0_indices = [i for i, row in dist_true._states_df.iterrows() if row['A'] == 0]
    a1_indices = [i for i, row in dist_true._states_df.iterrows() if row['A'] == 1]

    model.addCons(
        quicksum(vars_dict[r] for i, r in enumerate(names['r']) if i in a0_indices) == 1.0,
        name='r_prob_a0'
    )

    model.addCons(
        quicksum(vars_dict[r] for i, r in enumerate(names['r']) if i in a1_indices) == 1.0,
        name='r_prob_a1'
    )

    # enforce all pi and phi variables to be at least 1e-5
    for pi in names['pi']:
        model.addCons(vars_dict[pi] >= 1e-5)
    for phi in names['phi']:
        model.addCons(vars_dict[phi] >= 1e-5)


    mip_info = {
        'settings': dict(settings),
        'names': names,
        'upper_bounds': {k: np.array([vars_dict[n].getUbOriginal() for n in n_list]) if isinstance(n_list, list) else
                            np.array([vars_dict[n_list].getUbOriginal()]) for k, n_list in names.items()},
        'lower_bounds': {k: np.array([vars_dict[n].getLbOriginal() for n in n_list]) if isinstance(n_list, list) else
                            np.array([vars_dict[n_list].getLbOriginal()]) for k, n_list in names.items()},
        'vars_dict': vars_dict
    }

    return model, mip_info

def check_sensitivity_mip_solution(mip, mip_info, debug_flag=False):
    """
    Check if MIP solution satisfies all constraints and properties.

    Args:
    mip: CPLEX object with solution
    mip_info: Dictionary with MIP metadata including variable names and settings
    debug_flag: Whether to enter debugger on error
    """
    debugger = lambda: True
    if debug_flag:
        from src.debug import ipsh as debugger

    names = mip_info['names']
    settings = mip_info['settings']
    # objval = sol.get_objective_value()
    #
    # # Get all variable values
    # p = np.array(sol.get_values(names['p']))
    # q = np.array(sol.get_values(names['q']))
    # r = np.array(sol.get_values(names['r']))
    # D = np.array(sol.get_values(names['D']))
    # d = np.array(sol.get_values(names['d']))
    # B = np.array(sol.get_values(names['B']))
    # e = np.array(sol.get_values(names['e']))
    # pi = np.array(sol.get_values(names['pi']))
    # phi = np.array(sol.get_values(names['phi']))
    # ry = np.array(sol.get_values(names['ry']))
    # n_fp = sol.get_values(names['n_fp'])
    # n_fn = sol.get_values(names['n_fn'])

    model = mip
    vars_dict = mip_info['vars_dict']
    objval = model.getObjVal()

    # Get all variable values
    p = np.array([model.getVal(vars_dict[name]) for name in names['p']])
    q = np.array([model.getVal(vars_dict[name]) for name in names['q']])
    r = np.array([model.getVal(vars_dict[name]) for name in names['r']])
    D = np.array([model.getVal(vars_dict[name]) for name in names['D']])
    d = np.array([model.getVal(vars_dict[name]) for name in names['d']])
    B = np.array([model.getVal(vars_dict[name]) for name in names['B']])
    e = np.array([model.getVal(vars_dict[name]) for name in names['e']])
    pi = np.array([model.getVal(vars_dict[name]) for name in names['pi']])
    phi = np.array([model.getVal(vars_dict[name]) for name in names['phi']])
    ry = np.array([model.getVal(vars_dict[name]) for name in names['ry']])
    n_fp = model.getVal(vars_dict[names['n_fp']])
    n_fn = model.getVal(vars_dict[names['n_fn']])

    try:
        # Check probability constraints
        # round up all variables to 1e-5
        p = np.round(p, 8)
        q = np.round(q, 8)
        r = np.round(r, 8)
        pi = np.round(pi, 8)
        phi = np.round(phi, 8)
        ry = np.round(ry, 8)
        D = np.round(D, 8)

        assert np.allclose(np.sum(p), 1.0), "p does not sum to 1"
        # choose A, sum over r to see if it gives 1
        a0_indices = [i for i, row in settings['dist_true']._states_df.iterrows() if row['A'] == 0]
        a1_indices = [i for i, row in settings['dist_true']._states_df.iterrows() if row['A'] == 1]
        b0_indices = [i for i, row in settings['dist_true']._states_df.iterrows() if row['B'] == 0]
        b1_indices = [i for i, row in settings['dist_true']._states_df.iterrows() if row['B'] == 1]
        assert np.allclose(np.sum(r[a0_indices]), 1.0), "r does not sum to 1 for A=0"
        assert np.allclose(np.sum(r[a1_indices]), 1.0), "r does not sum to 1 for A=1"
        assert np.allclose(np.sum(q[b0_indices]), 1.0), "q does not sum to 1 for B=0"
        assert np.allclose(np.sum(q[b1_indices]), 1.0), "q does not sum to 1 for B=1"

        assert np.all(p >= 0) and np.all(p <= 1), "p not in [0,1]"
        assert np.all(q >= 0) and np.all(q <= 1), "q not in [0,1]"
        assert np.all(r >= 0) and np.all(r <= 1), "r not in [0,1]"

        # Check marginal distributions pi and phi
        dist_true = settings['dist_true']
        unique_df = settings['unique_df']
        user_df = settings['user_df']

        # For each x,a,b combination
        for xab_idx in range(len(pi)):
            # Get corresponding y indices for this x,a,b
            y0_idx = unique_df.loc[xab_idx]['id']
            y1_idx = unique_df.loc[xab_idx]['id_complement']

            # Check pi = sum_y p
            assert np.allclose(pi[xab_idx], p[y0_idx] + p[y1_idx], atol=1e-5), f"pi != sum_y p for xab={xab_idx} with pi = {pi[xab_idx]}"

            # Check phi = sum_y r
            assert np.allclose(phi[xab_idx], r[y0_idx] + r[y1_idx], atol=1e-5), f"phi != sum_y r for xab={xab_idx} with phi = {phi[xab_idx]}"

            # Check if matches user beliefs on the protected attribute
            if unique_df.loc[xab_idx]['probing'] and unique_df.loc[xab_idx]['v'] in [0,1]:
                xaprim_idx = unique_df.loc[xab_idx]['id_aprim']
                # this suffices because on probing examples with v in [0,1] A is set to v in unique_df

                assert pi[xab_idx] >= pi[xaprim_idx], f"Protected attribute belief violated for xab={xab_idx}"


        # Check causal structure constraints
        for yxab_idx in range(len(p)):
            row = unique_df[unique_df['id'] == yxab_idx]
            if row.empty:
                row = unique_df[unique_df['id_complement'] == yxab_idx]
            xab_idx = row.index[0]
            # Check p*phi = r*pi
            assert np.allclose(p[yxab_idx] * phi[xab_idx], r[yxab_idx] * pi[xab_idx], atol=1e-5), \
                f"Causal structure violated for yxab={yxab_idx}, got p*phi={p[yxab_idx] * phi[xab_idx]}, r*pi={r[yxab_idx] * pi[xab_idx]}"

            # Check p <= r
            assert r[yxab_idx] - p[yxab_idx] >= -1e-05, f"p not less than r for yxab={yxab_idx} with {p[yxab_idx]} !!<=!! {r[yxab_idx]}"


        # Check fairness constraints
        all_fair, all_unfair = 0, 0
        observed_fp, observed_fn = 0, 0
        for i, (d_val, D_val, B_val) in enumerate(zip(d, D, B)):
            if d_val == 0:  # Fair case
                assert abs(D_val) <= settings['delta'], f"Unfair D for fair case {i}"
                all_fair += 1
                if user_df.loc[i]['c'] == 1:
                    observed_fp += 1
            else:  # Unfair case
                if B_val == 1:
                    assert D_val <= -settings['delta'] - settings['gamma'], \
                        f"D not negative enough for unfair case {i}: {D_val} !!<=!! {-settings['delta'] - settings['gamma']}"
                else:
                    assert D_val >= settings['delta'] - settings['gamma'], \
                        f"D not large enough for unfair case {i}: {D_val} !!>=!! {settings['delta'] + settings['gamma']}"
                all_unfair += 1
                if user_df.loc[i]['c'] == 0:
                    observed_fn += 1

            # D = ry_axb - ry_a'xb
            ry_id = user_df.loc[i]['id_unique']
            expected_D = ry[ry_id] - ry[unique_df.loc[ry_id]['id_aprim']]
            assert np.allclose(D_val, expected_D, atol=1e-5), f"Expected D value {expected_D} does not match {D_val} for user decision {i}"

        assert all_fair + all_unfair == len(user_df), "Not all user decisions accounted for in fairness constraints"

        # Check false alarms, missed cases indicators
        assert np.allclose(n_fp, observed_fp), f"False alarms count {n_fp} does not match expected {observed_fp}"
        assert np.allclose(n_fn, observed_fn), f"Missed cases count {n_fn} does not match expected {observed_fn}"

        # Check P(X) = P(X,do(A))
        # Check P(X) = P(X,do(B))
        for yxab_idx in range(len(p)):
            state = dist_true._states_df.iloc[yxab_idx][dist_true.state_columns]

            # Create states with flipped Y, A, B and all combinations
            state_yprim = state.copy()
            state_yprim['Y'] = 1 - state['Y']

            state_aprim = state.copy()
            state_aprim['A'] = 1 - state['A']

            state_bprim = state.copy()
            state_bprim['B'] = 1 - state['B']

            state_yaprim = state_aprim.copy()
            state_yaprim['Y'] = 1 - state['Y']

            state_ybprim = state_bprim.copy()
            state_ybprim['Y'] = 1 - state['Y']

            state_abprim = state_aprim.copy()
            state_abprim['B'] = 1 - state['B']

            state_yabprim = state_yaprim.copy()
            state_yabprim['B'] = 1 - state['B']

            indices = dist_true.get_index_of([state_yprim.values, state_aprim.values, state_bprim.values,
                                              state_yaprim.values, state_ybprim.values,
                                              state_abprim.values, state_yabprim.values])
            yprimxab_idx, yxaprimb_idx, yxabprim_idx, yprimxaprimb_idx, yprimxabprim_idx, yxaprimbprim_idx, yprimxaprimbrpim_idx = indices

            p_x = 0
            for ind in [yxab_idx] + indices:
                p_x += p[ind]

            r_xa = r[yxab_idx] + r[yprimxab_idx] + r[yxabprim_idx] + r[yprimxabprim_idx]
            q_xb = q[yxab_idx] + q[yprimxab_idx] + q[yxaprimb_idx] + q[yprimxaprimb_idx]

            assert np.allclose(p_x, r_xa, atol=1e-5), f"P(X) ≠ P(X,do(A)) for index {yxab_idx}: got p_x: {p_x}, r_xa: {r_xa}"
            assert np.allclose(p_x, q_xb, atol=1e-5), f"P(X) ≠ P(X,do(B)) for index {yxab_idx}: got p_x: {p_x}, q_xb: {q_xb}"

        # Check Pr((Y| A=a,X,B)) = Pr((Y | do(A=a),X,B)
        for yxab_idx in range(len(p)):
            state = dist_true._states_df.iloc[yxab_idx][dist_true.state_columns]

            state_yprim = state.copy()
            state_yprim['Y'] = 1 - state['Y']
            yprimxab_idx = dist_true.get_index_of([state_yprim.values])[0]

            # p_{yaxb} / pi_{xab} = p_{yaxb} / (p_{yaxb} + p_{y'axb})
            cond_p = 0 if p[yxab_idx] == 0 else p[yxab_idx] / (p[yxab_idx] + p[yprimxab_idx])

            # r_{yaxb} / phi_{xab} = r_{yaxb} / (r_{yaxb} + r_{y'axb})
            cond_r = 0 if r[yxab_idx] == 0 else r[yxab_idx] / (r[yxab_idx] + r[yprimxab_idx])

            row = unique_df[unique_df['id'] == yxab_idx]
            if row.empty:
                row = unique_df[unique_df['id_complement'] == yxab_idx]
            xab_idx = row.index[0]
            phival = phi[xab_idx]
            pival = pi[xab_idx]

            assert abs(phival - r[yxab_idx] - r[yprimxab_idx]) <= 1e-5, f"phi {phival} !!=!! {r[yxab_idx] + r[yprimxab_idx]} for xab={xab_idx}"
            assert abs(pival - p[yxab_idx] - p[yprimxab_idx]) <= 1e-5, f"pi {pival} !!=!! {p[yxab_idx] + p[yprimxab_idx]} for xab={xab_idx}"
            assert np.allclose(cond_p, p[yxab_idx] / pival, atol=1e-5), f"Conditional probabilities don't match for index {yxab_idx}: got {cond_p}, {p[yxab_idx]}, {pival}"
            assert np.allclose(cond_r, r[yxab_idx] / phival, atol=1e-5), f"Conditional probabilities don't match for index {yxab_idx}: got {cond_r}, {r[yxab_idx]}, {phival}"

            assert np.allclose(p[yxab_idx] * phi[xab_idx], r[yxab_idx] * pi[xab_idx], atol=1e-5), \
                f"Causal structure violated for yxab={yxab_idx}, got p*phi={p[yxab_idx] * phi[xab_idx]}, r*pi={r[yxab_idx] * pi[xab_idx]}"

            print(p[yxab_idx], r[yxab_idx])

            # assert np.allclose(p[yxab_idx] * phi[xab_idx] / pi[xab_idx], r[yxab_idx], atol=1e-5), \
            #     f"Causal structure violated for yxab={yxab_idx}, got p*phi={p[yxab_idx] * phi[xab_idx]}, r*pi={r[yxab_idx] * pi[xab_idx]}"

            # assert np.allclose(p[yxab_idx] / pi[xab_idx], r[yxab_idx] / phi[xab_idx], atol=1e-5), \
            #     f"Causal structure violated for yxab={yxab_idx}, got p*phi={p[yxab_idx]} * {phi[xab_idx]}, r*pi={r[yxab_idx]} * {pi[xab_idx]}"
            #
            assert np.allclose(cond_p, cond_r, atol=1e-5), f"Conditional probabilities don't match for index {yxab_idx}: got {cond_p}, {cond_r}; p: {p[yxab_idx]}, phi: {phival}, r: {r[yxab_idx]}, pi: {pival}"

        # Check reliability constraints
        for xb_idx in range(len(user_df)):
            u = user_df.loc[xb_idx]['u']
            cols = dist_true.state_columns  # column names
            state_b0 = user_df.loc[xb_idx][cols].values  # values of ['X','B','A','Y' = 'Yhat']
            state_b1 = state_b0.copy()
            for i, col in enumerate(cols):
                if col == 'Y':
                    state_b0[i] = 1
                    state_b1[i] = 1
                elif col == 'B':
                    state_b1[i] = 1
                    state_b0[i] = 0
            p_ind_b0 = dist_true.get_index_of(state_b0)[0]
            p_ind_b1 = dist_true.get_index_of(state_b1)[0]
            if u == 1:
                assert q[p_ind_b1] - p[p_ind_b0] >= settings['epsilon'], \
                    f"Reliability constraint violated for user MORE RELIABLE decision {xb_idx}"
            elif u == -1:
                assert p[p_ind_b1] - q[p_ind_b0] <= settings['epsilon'], \
                    f"Reliability constraint violated for user LESS RELIABLE decision {xb_idx}"
            elif u == 0:
                assert np.allclose(p[p_ind_b0], q[p_ind_b1], atol=1e-5), \
                    f"Reliability constraint violated for user NO CHANGE decision {xb_idx}"
                assert np.allclose(p[p_ind_b1], q[p_ind_b0], atol=1e-5), \
                    f"Reliability constraint violated for user NO CHANGE decision {xb_idx}"

        # Check error constraints
        total_error = np.sum(e)
        assert settings['E_min'] <= total_error <= settings['E_max'], \
            f"Total error {total_error} outside bounds [{settings['E_min']}, {settings['E_max']}]"
        for i, e_val in enumerate(e):
            xab_idx = unique_df.loc[i]['id']
            xaprimb_idx = unique_df.loc[unique_df.loc[i]['id_aprim']]['id']
            primxab_idx = unique_df.loc[i]['id_complement']
            primxaprimb_idx = unique_df.loc[unique_df.loc[i]['id_aprim']]['id_complement']

            yhat = unique_df.loc[i]['Yhat']
            y = unique_df.loc[i]['Y']

            if y == yhat:
                pval = p[xab_idx]
                paprimval = p[xaprimb_idx]
                pprimval = p[primxab_idx]
                pprimaprimval = p[primxaprimb_idx]
            else:
                pval = p[primxab_idx]
                paprimval = p[primxaprimb_idx]
                pprimval = p[xab_idx]
                pprimaprimval = p[xaprimb_idx]

            if e_val == 1:
                assert pval < pprimval, f"Error constraint violated for error at xab={i}"
                assert paprimval < pprimaprimval, f"Error constraint violated for error at xab={i}"
            else:
                assert pval >= pprimval, f"Error constraint violated for correct at xab={i}: {pval} <= {pprimval}"
                assert paprimval >= pprimaprimval, f"Error constraint violated for correct at xab={i}"


    except AssertionError as e:
        print(f'Solution validation failed: {str(e)}')
        if debug_flag:
            debugger()
        return False

    return True


class SensitivityMIP:
    """
    Convenience class to create, solve, and check the integrity of the ERM
    to fit a logistic regression classifier with accessibility constraints
    """

    PRINT_FLAG = True
    PARALLEL_FLAT = True

    def __init__(self, X, y, random_seed = 2338, **kwargs):
        """
        :param X:
        :param y:
        :param print_flag:
        :param parallel_flag:
        """
        raise NotImplementedError()
        default_params = SensitivityMIP.DEFAULT_VALUES
        default_params.update(kwargs)

        # set flags
        self._print_flag = SensitivityMIP.PRINT_FLAG
        self._parallel_flag = SensitivityMIP.PARALLEL_FLAT

        # initialize mip
        cpx, indices = build_sensitivity_mip(X = X, y = y, **default_params)

        # attach loss callback
        loss_cb = cpx.register_callback(LossCallback)
        loss_cb.initialize(X = X, y = y, loss_idx = indices['loss'], coef_idx = indices['coefs'])

        # set parameters
        cpx = self._set_mip_parameters(cpx, random_seed)

        # attach CPLEX object
        self.mip = cpx
        self.vars = self.mip.variables
        self.cons = self.mip.linear_constraints
        self.parameters = self.mip.parameters
        self.indices = indices

        # attach loss
        self.loss_cb = loss_cb

        # random seed
        self.random_seed = random_seed


    def solve(self, time_limit = 60.0, max_gap = 0.01, return_stats = False, return_incumbents = False):
        """
        solves MIP
        #
        :param time_limit: max # of seconds to run before stopping the B & B.
        :param return_stats: set to True to record basic profiling information as the B&B algorithm runs (warning: this may slow down the B&B)
        :param return_incumbents: set to True to record all imcumbent solution processed during the B&B search (warning: this may slow down the B&B)
        :return:
        """

        if (return_stats or return_incumbents):
            self._add_stats_callback(store_solutions = return_incumbents)

        if max_gap is not None:
            self.mip = set_mip_max_gap(self.mip, max_gap)

        if time_limit is not None:
            self.mip = set_mip_time_limit(self.mip, time_limit)

        self.mip.solve()
        # cpx = self.mip
        # cpx.solution.get_status_string()
        # cpx.solution.get_status()
        # cpx.solve()
        # cpx.solution.get_status_string()
        # cpx.solution.get_status()
        # cpx.conflict.get()
        # print(e)
        # cpx = self.mip
        # cpx.conflict.refine()

        info = self.solution_info
        if (return_stats or return_incumbents):
            info['progress_info'], info['progress_incumbents'] = self._stats_callback.get_stats()

        return info

    #### properties #todo: don't touch this
    @property
    def solution(self):
        """
        :return: handle to CPLEX solution
        """
        # todo add wrapper if solution does not exist
        return self.mip.solution

    @property
    def solution_info(self):
        """returns information associated with the current best solution for the mip"""
        return get_mip_stats(self.mip)

    @property
    def print_flag(self):
        """
        set as True in order to print output information of the MIP
        :return:
        """
        return self._print_flag

    @print_flag.setter
    def print_flag(self, flag):
        if flag is None:
            self._print_flag = SensitivityMIP.PRINT_FLAG
        elif isinstance(flag, bool):
            self._print_flag = bool(flag)
        else:
            raise ValueError('print_flag must be boolean or None')

        # toggle flag
        if self._print_flag:
            self.parameters.mip.display.set(self.parameters.mip.display.default())
            self.parameters.simplex.display.set(self._print_flag)
        else:
            self.parameters.mip.display.set(False)
            self.parameters.simplex.display.set(False)

    @property
    def parallel_flag(self):
        """
        set as True in order to print output information of the MIP
        :return:
        """
        return self._parallel_flag

    @parallel_flag.setter
    def parallel_flag(self, flag):

        if flag is None:
            self._parallel_flag = SensitivityMIP.PARALLEL_FLAG
        elif isinstance(flag, bool):
            self._parallel_flag = bool(flag)
        else:
            raise ValueError('parallel_flag must be boolean or None')

        # toggle parallel
        p = self.mip.parameters
        if self._parallel_flag:
            p.threads.set(0)
            p.parallel.set(0)
        else:
            p.parallel.set(1)
            p.threads.set(1)

    #### methods ####
    def check_solution(self):
        """
        runs basic tests to make sure that the MIP contains a suitable solution
        :return:
        """
        s = self.solution

    #### generic MIP methods ####
    def add_initial_solution(self, solution):
        """
        adds initial solutions to MIP
        :param coefs:
        :return:
        """
        raise NotImplementedError()
        sol = coefs.tolist()
        idx = self.indices['coefs']
        self.mip.MIP_starts.add(SparsePair(val = sol, ind = idx), self.mip.MIP_starts.effort_level.solve_MIP)

    def _set_mip_parameters(self, cpx, random_seed):
        """
        sets CPLEX parameters
        :param cpx:
        :return:
        """
        p = cpx.parameters
        p.randomseed.set(random_seed)

        # annoyances
        p.paramdisplay.set(False)
        p.output.clonelog.set(0)
        p.mip.tolerances.mipgap.set(0.0)
        return cpx

    def _add_stats_callback(self, store_solutions = False):
        if not hasattr(self, '_stats_callback'):
            sol_idx = self.indices['coefs']
            min_idx, max_idx = min(sol_idx), max(sol_idx)
            assert np.array_equal(np.array(sol_idx), np.arange(min_idx, max_idx + 1))
            cb = self.mip.register_callback(StatsCallback)
            cb.initialize(store_solutions, solution_start_idx = min_idx, solution_end_idx = max_idx)
            self._stats_callback = cb