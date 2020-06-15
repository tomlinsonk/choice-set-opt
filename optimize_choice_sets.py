import time
from itertools import chain, combinations

import numpy as np
import gurobipy as gp

from choice_models import DiscreteChoiceSetting, MNL, CDM, NL


def power_set(collection):
    """
    An iterator over the power set of a collection.
    :param collection: an iterable
    :return: an iterator
    """
    as_list = list(collection)
    return chain.from_iterable(combinations(as_list, r) for r in range(len(as_list) + 1))


def greedy_agreement_step(model, maximize=False):
    """
    A single step in the greedy algorithm for (dis)agreement. Find the single item whose inclusion decreases D(Z)
    by as much as possible (or increases, if maximize is True)
    :param model: a DiscreteChoiceModel
    :param maximize: boolean indicating whether to minimize or maximize agreement
    :return: the tuple (item to include, D after it's included)
    """
    opt_item = None
    opt_D = model.disagreement()

    for i in model.dcs.not_C:
        model.update_Z(i)
        after_D = model.disagreement()

        if (after_D > opt_D) if maximize else (after_D < opt_D):
            opt_item = i
            opt_D = after_D
        model.update_Z(i)

    return opt_item, opt_D


def greedy_agreement(model, maximize=False):
    """
    Optimize the agreement by repeatedly finding the alternative that makes the most progress. Updates model.dcs.Z
    to the found set.
    :param model: a DiscreteChoiceModel
    :param maximize: boolean indicating whether to minimize or maximize agreement
    :return: the model
    """
    model.set_Z([])

    before_D = -np.inf if maximize else np.inf
    after_D = model.disagreement()

    while (before_D < after_D) if maximize else (before_D > after_D):
        before_D = after_D
        opt_item, after_D = greedy_agreement_step(model, maximize=maximize)
        if opt_item is None:
            break
        model.update_Z(opt_item)

    return model


def opt_agreement(model, maximize=False):
    """
    Use brute force to find the set of alternatives optimizing agreement. By default, minimize D(Z).
    Updates model.dcs.Z to the optimal set.
    :param model: a DiscreteChoiceModel
    :param maximize: boolean indicating whether to minimize or maximize agreement
    :return: the model
    """
    opt_Z = None
    opt_D = -np.inf if maximize else np.inf

    for Z in power_set(model.dcs.not_C):
        Z = list(Z)
        model.set_Z(Z)
        this_D = model.disagreement()
        if (this_D > opt_D) if maximize else (this_D < opt_D):
            opt_Z = Z
            opt_D = this_D

    model.set_Z(opt_Z)
    return model


def approx_agreement(model, epsilon=0.01, maximize=False):
    """
    Wrapper to run the appropriate approximation algorithm on model depending on its class.
    :param model: a DiscreteChoiceModel
    :param epsilon: error tolerance
    :param maximize: boolean indicating whether to minimize or maximize agreement
    :return: the model
    """
    if isinstance(model, MNL):
        return mnl_approx_agreement(model, epsilon, maximize)
    elif isinstance(model, CDM):
        return cdm_approx_agreement(model, epsilon, maximize)
    elif isinstance(model, NL):
        return nl_approx_agreement(model, epsilon, maximize)

    raise TypeError('model must be one of: MNL, CDM, NL')


def process_L(model, L, maximize):
    """
    Do the final step of Algorithm 1: find best sets.
    :param model: a DiscreteChoiceModel
    :param L: output of Algorithm 1
    :param maximize: if True, set Z to max_Z, otherwise set it to min_Z
    :return:
    """
    min_Z = None
    max_Z = None
    min_D = np.inf
    max_D = -np.inf

    for key, (Z, _) in L.items():
        Z = list(Z)
        model.set_Z(Z)
        this_D = model.disagreement()
        if this_D < min_D:
            min_Z = Z
            min_D = this_D
        if this_D > max_D:
            max_Z = Z
            max_D = this_D

    model.set_Z(max_Z if maximize else min_Z)

    return min_Z, min_D, max_Z, max_D, len(L)


def mnl_approx_agreement(mnl, epsilon=0.01, maximize=False):
    """
    Use approximation algorithm to find the set of alternatives optimizing agreement within error epsilon.
    By default, minimize D(Z). Updates model.dcs.Z to the optimal set. Must be run with a MultinomialLogit
    :param mnl: a MultinomialLogit
    :param epsilon: error tolerance
    :param maximize: boolean indicating whether to minimize or maximize agreement
    :return: the model
    """

    if not isinstance(mnl, MNL):
        raise TypeError('This method must be run with an MNL.')

    n = len(mnl.dcs.A)
    k = len(mnl.dcs.C)
    m = len(mnl.dcs.not_C)

    exp_utils = np.exp(mnl.u - np.min(mnl.u) + 0.01)  # make sure utilities are positive
    delta = epsilon / (k * m * n * (n - 1))

    base_change = np.log(1 + delta)

    # an entry of L stores Z and t_a for each a in A (total exp-utility on alts)
    L = {tuple([0] * n): (frozenset(), np.zeros(n, dtype=int))}
    for item in mnl.dcs.not_C:
        updates = dict()
        for key, (Z, exp_util_sums) in L.items():
            new_key = tuple(np.floor(np.log(exp_utils[:, item] + exp_util_sums) / base_change).astype(int))
            if new_key not in L:
                updates[new_key] = Z.union({item}), exp_util_sums + exp_utils[:, item]

        L.update(updates)

    return process_L(mnl, L, maximize)


def cdm_approx_agreement(cdm, epsilon=0.01, maximize=False):
    """
    Use approximation algorithm to find the set of alternatives optimizing agreement within error epsilon (only
    guaranteed if alternatives have no pulls on each other).
    By default, minimize D(Z). Updates model.dcs.Z to the optimal set. Must be run with a CDM.
    :param cdm: a CDM
    :param epsilon: error tolerance
    :param maximize:  boolean indicating whether to minimize or maximize agreement
    :return: the model
    """

    if not isinstance(cdm, CDM):
        raise TypeError('This method must be run with a CDM.')

    n = len(cdm.dcs.A)
    k = len(cdm.dcs.C)
    m = len(cdm.dcs.not_C)

    # Make everything positive, keep diagonal 0
    shifted_u_p = (cdm.u_p - np.min(cdm.u_p) + 0.01)
    shifted_u_p[:, np.arange(n), np.arange(n)] = 0

    delta = epsilon / (4 * k * m * n * (n - 1))
    base_change = np.log(1 + delta)

    # each entry of L stores Z and sum_{z in C\cup Z} q_a{z, x} for each a in A and item x in C (numerator of Pr)
    L = {tuple([0] * (k * n)): (frozenset(), np.zeros(n * k))}
    for item in cdm.dcs.not_C:
        updates = dict()
        for key, (Z, pull_sums) in L.items():
            new_pull_sums = pull_sums + shifted_u_p[:, item][:, cdm.dcs.C].flatten()

            # don't need to take log, since we want to approximate exp(pull sums)
            new_key = tuple(np.floor(new_pull_sums / base_change).astype(int))

            if new_key not in L:
                updates[new_key] = Z.union({item}), new_pull_sums

        L.update(updates)

    return process_L(cdm, L, maximize)


def nl_approx_agreement(nl, epsilon=0.01, maximize=False):
    """
    Use approximation algorithm to find the set of alternatives optimizing agreement within error epsilon.
    By default, minimize D(Z). Updates model.dcs.Z to the optimal set. Must be run with a NL.
    :param nl: a NL
    :param epsilon: error tolerance
    :param maximize:  boolean indicating whether to minimize or maximize agreement
    :return: the model
    """

    if not isinstance(nl, NL):
        raise TypeError('This method must be run with a NL.')

    n = len(nl.dcs.A)
    k = len(nl.dcs.C)
    m = len(nl.dcs.not_C)

    # for fast membership checking
    set_not_C = frozenset(nl.dcs.not_C)

    tracked_nodes = [node for root in nl.roots for node in root.traverse() if
                     not {child.item for child in node.children}.isdisjoint(set_not_C)]

    exp_utility_effects = np.zeros((k + m, len(tracked_nodes)))
    for i, node in enumerate(tracked_nodes):
        for child in node.children:
            if child.item in set_not_C:
                exp_utility_effects[child.item, i] = child.utility

    exp_utility_effects -= np.min(exp_utility_effects) - 0.01  # make utilities positive
    exp_utility_effects[nl.dcs.C, :] = np.nan  # make sure we don't accidentally use placeholder rows
    exp_utility_effects = np.exp(exp_utility_effects)

    max_height = max(root.height for root in nl.roots) - 1
    if epsilon / (k * n * (n - 1)) > 1:
        delta = ((epsilon / (k * n * (n - 1)) + 1) ** (1 / max_height) - 1) / m
    else:
        delta = min((epsilon / (k * n * (n - 1)) + 1) ** (1 / max_height) - 1,
                    1 - (1 - epsilon / (k * n * (n - 1))) ** (1 / max_height)) / m
    base_change = np.log(1 + delta)

    # L stores the exp utility sum at each node in tracked_nodes, and the set Z inducing that sum
    L = {tuple([0] * len(tracked_nodes)): (frozenset(), np.zeros(len(tracked_nodes)))}
    for item in nl.dcs.not_C:
        updates = dict()
        for key, (Z, exp_utility_sums) in L.items():
            new_exp_utility_sums = exp_utility_sums + exp_utility_effects[item]
            new_key = tuple(np.floor(np.log(new_exp_utility_sums) / base_change).astype(int))

            if new_key not in L:
                updates[new_key] = Z.union({item}), new_exp_utility_sums

        L.update(updates)

    return process_L(nl, L, maximize)


def mnl_integer_program_agreement(mnl, maximize=False):
    mnl.set_Z([])

    k = len(mnl.dcs.C)
    m = len(mnl.dcs.not_C)
    n = len(mnl.dcs.A)

    e_C = [sum(np.exp(mnl.u[a, y]) for y in mnl.dcs.C) for a in range(n)]
    e = np.exp(mnl.u)

    model = gp.Model()
    model.setParam('LogToConsole', False)
    delta = model.addVars(k, n, n, name='delta')
    z = model.addVars(n, name='z')
    x = model.addVars(m, vtype=gp.GRB.BINARY, name='x')

    if maximize:
        g = model.addVars(k, n, n, vtype=gp.GRB.BINARY, name='g')
        model.addConstrs(2 * g[y, a, b] + z[a] * e[a, mnl.dcs.C[y]] - z[b] * e[b, mnl.dcs.C[y]] >= delta[y, a, b]
                         for y in range(k) for a in range(n) for b in range(a + 1, n))
        model.addConstrs(2 * (1 - g[y, a, b]) + z[b] * e[b, mnl.dcs.C[y]] - z[a] * e[a, mnl.dcs.C[y]] >= delta[y, a, b]
                         for y in range(k) for a in range(n) for b in range(a + 1, n))

    model.addConstrs(z[a] * e[a, mnl.dcs.C[y]] - z[b] * e[b, mnl.dcs.C[y]] <= delta[y, a, b]
                     for y in range(k) for a in range(n) for b in range(a+1, n))
    model.addConstrs(z[b] * e[b, mnl.dcs.C[y]] - z[a] * e[a, mnl.dcs.C[y]] <= delta[y, a, b]
                     for y in range(k) for a in range(n) for b in range(a+1, n))
    model.addConstrs(z[a] * e_C[a] + z[a] * gp.quicksum(x[i] * e[a, mnl.dcs.not_C[i]] for i in range(m)) == 1
                     for a in range(n))

    obj = gp.quicksum(delta[y, a, b] for y in range(k) for a in range(n) for b in range(a+1, n))
    model.setObjective(obj, gp.GRB.MAXIMIZE if maximize else gp.GRB.MINIMIZE)
    model.optimize()

    for i in range(m):
        if x[i].x > 0.5:
            mnl.update_Z(mnl.dcs.not_C[i])


def greedy_promotion_step(model, target):
    """
    A single step in the greedy algorithm for promotion. Find the single item whose inclusion increases the number of
    agent whose favorite item is target, or if no item increases the number of agents, pick the item that decreases
    the total gap between the target's choice probability and  the favorites by as much as possible
    :param model: a DiscreteChoiceModel
    :param target: an index into C indicating which item to promote
    :return: the tuple (item to include, num promoted agents after it's included, promotion gap after it's included)
    """
    opt_item = None
    opt_agents = model.promotion_count(target)
    opt_gap = model.promotion_gap(target)

    for i in model.dcs.not_C:
        model.update_Z(i)
        after_agents = model.promotion_count(target)
        after_gap = model.promotion_gap(target)

        if after_agents > opt_agents or (after_agents == opt_agents and after_gap < opt_gap):
            opt_item = i
            opt_agents = after_agents
            opt_gap = after_gap

        model.update_Z(i)

    return opt_item, opt_agents, opt_gap


def greedy_promotion(model, target):
    """
    Optimize the number of agents whose favorite item is target greedily, see greedy_promotion_step for details.
    Updates model.dcs.Z to the found set.
    :param model: a DiscreteChoiceModel
    :param target: an index into model.dcs.C
    :return: the model
    """
    model.set_Z([])

    before_agents = -np.inf
    before_gap = -np.inf
    after_agents = model.promotion_count(target)
    after_gap = model.promotion_gap(target)

    while before_agents < after_agents or before_gap < after_gap:
        before_agents = after_agents
        before_gap = after_gap

        opt_item, after_agents, after_gap = greedy_promotion_step(model, target)
        if opt_item is None:
            break
        model.update_Z(opt_item)

    return model


def opt_promotion(model, target):
    """
    Use brute force to find the set of alternatives optimizing promotion of target.
    Updates model.dcs.Z to the optimal set.
    :param model: a DiscreteChoiceModel
    :param target: index into model.dcs.C
    :return: the model
    """
    opt_Z = None
    opt_agents = -np.inf

    for Z in power_set(model.dcs.not_C):
        Z = list(Z)
        model.set_Z(Z)
        this_agents = model.promotion_count(target)
        if this_agents > opt_agents:
            opt_Z = Z
            opt_agents = this_agents

    model.set_Z(opt_Z)
    return model


def cdm_approx_promotion(cdm, target, epsilon=0.01):
    """
    Use approximation algorithm to find the set of alternatives optimizing promotion within error epsilon
    (only guaranteed if alternatives have no pulls on each other).
    Updates model.dcs.Z to the found set. Must be run with a CDM.
    :param cdm: a CDM
    :param epsilon: error tolerance
    :return: the model
    """

    if not isinstance(cdm, CDM):
        raise TypeError('This method must be run with a CDM.')

    n = len(cdm.dcs.A)
    k = len(cdm.dcs.C)
    m = len(cdm.dcs.not_C)

    # Make everything positive, keep diagonal 0
    shifted_u_p = (cdm.u_p - np.min(cdm.u_p) + 1)
    shifted_u_p[:, np.arange(n), np.arange(n)] = 0

    delta = epsilon / (20 * m)
    base_change = np.log(1 + delta)

    # each entry of L stores Z and sum_{z in C\cup Z} q_a{z, x} for each a in A and item x in C (numerator of Pr)
    L = {tuple([0] * (k * n)): (frozenset(), np.zeros(n * k))}
    for item in cdm.dcs.not_C:
        updates = dict()
        for key, (Z, pull_sums) in L.items():
            new_pull_sums = pull_sums + shifted_u_p[:, item][:, cdm.dcs.C].flatten()

            # don't need to take log, since we want to approximate exp(pull sums)
            new_key = tuple(np.floor(new_pull_sums / base_change).astype(int))

            if new_key not in L:
                updates[new_key] = Z.union({item}), new_pull_sums

        L.update(updates)

    opt_Z = None
    opt_agents = -np.inf

    for key, (Z, _) in L.items():
        Z = list(Z)
        cdm.set_Z(Z)
        this_agents = cdm.promotion_count(target)
        if this_agents > opt_agents:
            opt_Z = Z
            opt_agents = this_agents

    cdm.set_Z(opt_Z)
    return cdm, len(L), 2 ** m


if __name__ == '__main__':
    # Example usage
    cdm = CDM.random_instance(2, 2, 41)
    mnl = MNL.random_instance(2, 2, 41)

    # Example usage
    dcs = DiscreteChoiceSetting(
        list(range(2)),
        list(range(4)),
        list(range(2))
    )

    u = np.array([
        [8, 2, 15, 0],
        [8, 8, 0, 15]
    ])

    mnl = MNL(dcs, u)

    print('Items:', dcs.U)
    print('Agents:', dcs.A)
    print('Choice set:', dcs.C)
    print('Utilities:', u)

    print('Initial disagreement:', mnl.disagreement())

    start = time.time()
    greedy_agreement(mnl, maximize=False)
    print('Greedy')
    print('\tSolution:', mnl.dcs.Z)
    print('\tD(Z):', mnl.disagreement())
    print('\tRuntime:', time.time() - start)

    start = time.time()
    mnl_approx_agreement(mnl, 0.1, maximize=False)
    print('Approximation alg')
    print('\tSolution:', mnl.dcs.Z)
    print('\tD(Z):', mnl.disagreement())
    print('\tRuntime:', time.time() - start)

    start = time.time()
    opt_agreement(mnl, maximize=False)
    print('Brute force')
    print('\tSolution:', mnl.dcs.Z)
    print('\tD(Z):', mnl.disagreement())
    print('\tRuntime:', time.time() - start)

    start = time.time()
    mnl_integer_program_agreement(mnl, maximize=False)
    print('MIBLP')
    print('\tSolution:', mnl.dcs.Z)
    print('\tD(Z):', mnl.disagreement())
    print('\tRuntime:', time.time() - start)
