from abc import ABC, abstractmethod
from itertools import combinations

import numpy as np


class DiscreteChoiceSetting:
    """
    This class stores the agents A , items U, initial choice set C, and alternatives we have chosen to include Z.
    IMPORTANT: C_or_Z must have
    """

    def __init__(self, A, U, C):
        """
        :param A: list of agents
        :param U: list of all items
        :param C: choice set indices into U
        """
        self.A = A
        self.U = U
        self.C = C
        self.Z = []
        self.C_or_Z = C[:]

        # Indices of C in C_or_Z
        self.C_indices = np.arange(len(C))

        self.not_C = [i for i in range(len(U)) if i not in C]

    def update_C_indices(self):
        """
        Update the indices of C in C_or_Z
        """
        _, self.C_indices, _ = np.intersect1d(self.C_or_Z, self.C, return_indices=True)


class DiscreteChoiceModel(ABC):
    """
    Abstract base class for discrete choice models (e.g. MultinomialLogit and CDM).
    Defines methods for computing disagreement, promotion, and choice probabilities.
    """

    def __init__(self, dcs):
        self.dcs = dcs

    def disagreement(self):
        """
        Compute the total disagreement of agents about items in dcs.C
        :return: the sum of disagreement of every pair of agents
        """

        total = 0
        P = self.all_probabilities()[:, self.dcs.C_indices]

        for i, j in combinations(range(len(P)), 2):
            total += np.abs(P[i] - P[j]).sum()

        return total

    def promotion_count(self, target):
        """
        Compute the number of agents for whom target is their favorite item among dcs.C
        :param target: index into self.dcs.C
        :return: the number of such agents
        """

        P = self.all_probabilities()
        return np.count_nonzero(P[:, self.dcs.C_indices].max(axis=1) == P[:, self.dcs.C_indices[target]])

    def promotion_gap(self, target):
        """
        Compute the total probability gap between target and each agent's favorite item
        :return: the number of such agents
        """

        P = self.all_probabilities()
        return np.sum(P[:, self.dcs.C_indices].max(axis=1) - P[:, self.dcs.C_indices[target]])

    def favorite(self, agent):
        """
        Return the index into C of the favorite item among C of agent
        :param agent: an index into A
        :return: the index into C
        """

        P = self.all_agent_probabilities(agent)
        return np.argmax(P[self.dcs.C_indices])

    def set_Z(self, Z):
        """
        Set the included alternatives to Z.
        :param Z: a list of indices into U
        """
        self.dcs.Z = Z
        self.dcs.C_or_Z = self.dcs.C + Z

    def update_Z(self, i):
        """
        Add i to the included alternatives if it isn't one already, otherwise remove it
        :param i: an index into U
        """
        if i in self.dcs.Z:
            self.dcs.C_or_Z.remove(i)
            self.dcs.Z.remove(i)
        else:
            self.dcs.C_or_Z.append(i)
            self.dcs.Z.append(i)

    def set_C(self, C):
        """
        Set the choice set to C, setting the included alternatives Z to [] in the process.
        :param C: a list of indices into U
        """
        self.dcs.C = list(C)
        self.dcs.not_C = [i for i in range(len(self.dcs.U)) if i not in C]
        self.set_Z([])
        self.dcs.update_C_indices()

    @abstractmethod
    def probability(self, agent, item):
        """
        Compute the probability that agent picks item
        :param agent: index into dcs.A
        :param item: index into dcs.U
        :return: Pr[agent <- ]
        """
        pass

    @abstractmethod
    def all_agent_probabilities(self, agent):
        """
        Compute all choice probabilities for agent
        :param agent: an index into dcs.A
        :return: an array of choice probabilities for C_or_Z
        """
        pass

    @abstractmethod
    def all_probabilities(self):
        """
        Compute all choice probabilities for every agent over items in C_or_Z
        :return: a matrix whose i, j entry is agent's i probability of picking item j
        """
        pass

    @classmethod
    @abstractmethod
    def random_instance(cls, n, k, m):
        """
        Construct a random instance with n agents, k items, m alternatives, and random preferences.
        :param n: num agents
        :param k: num items
        :param m: num alternatives
        :return: an instance of this DiscreteChoiceModel
        """
        pass


class MNL(DiscreteChoiceModel):
    """
    MNL model, introduced in:

    [1] McFadden, D. Conditional logit analysis of qualitative choice behavior. In Zarembka, P. (ed.),
            Frontiers in Econometrics, pp. 105–142. Academic Press, 1974.
    """

    short_name = 'mnl'

    def __init__(self, dcs, u):
        """
        :param dcs: a DiscreteChoiceSetting
        :param u: a utility matrix whose [i, j] entry is agent i's utility for item j
        """
        super().__init__(dcs)
        self.u = u

    def probability(self, agent, item):
        """
        Compute the probability that agent picks item
        :param agent: index into dcs.A
        :param item: index into dcs.U
        :return: Pr(agent <- item | dcs.C)
        """
        return np.exp(self.u[agent, item]) / np.exp(self.u[agent, self.dcs.C_or_Z]).sum()

    def all_agent_probabilities(self, agent):
        """
        Compute all choice probabilities for agent
        :param agent: an index into dcs.A
        :return: an array of choice probabilities for C_or_Z
        """
        x = np.exp(self.u[agent, self.dcs.C_or_Z])
        return x / x.sum()

    def all_probabilities(self):
        """
        Compute all choice probabilities for every agent over items in C_or_Z
        :return: a matrix whose i, j entry is agent's i probability of picking item j
        """
        x = np.exp(self.u[:, self.dcs.C_or_Z])
        return x / x.sum(axis=1, keepdims=True)

    @classmethod
    def random_instance(cls, n, k, m):
        """
        Construct a random MNL instance with n agent, k items, m alternatives, and utilities uniform over [0, 1].
        :param n: num agents
        :param k: num items
        :param m: num alternatives
        :return: a MultinomialLogit instance
        """
        dcs = DiscreteChoiceSetting(
            list(range(n)),
            list(range(k + m)),
            list(range(k))
        )

        u = np.array([
            np.random.uniform(0, 1, k + m),
            np.random.uniform(0, 1, k + m)
        ])

        return MNL(dcs, u)


class CDM(DiscreteChoiceModel):
    """
    Context-dependent random utility model, introduced in:

    [2] Seshadri, A., Peysakhovich, A., and Ugander, J. Discovering context effects from raw choice data.
        In International Conference on Machine Learning, pp. 5660–5669, 2019.

    Note: for this code, we use the simplified parametrization in [2], Eq. (1)
    """

    short_name = 'cdm'

    def __init__(self, dcs, u_p):
        """
        :param dcs: a DiscreteChoiceSetting
        :param u_p: an array of utility-adjusted context effect matrices for each agent.
                    The (i, j)th entry of a context effect matrix stores the push of item i on item j (u_{ij} in [1])
        """
        super().__init__(dcs)
        self.u_p = u_p

    def probability(self, agent, item):
        """
        Compute the probability that agent picks item
        :param agent: index into dcs.A
        :param item: index into dcs.U
        :return: Pr(agent <- item | dcs.C)
        """
        effective_u = self.u_p[agent][self.dcs.C_or_Z].sum(axis=0)

        return np.exp(effective_u[item]) / np.exp(effective_u[self.dcs.C_or_Z]).sum()

    def all_agent_probabilities(self, agent):
        """
        Compute all choice probabilities for agent
        :param agent: an index into dcs.A
        :return: an array of choice probabilities for C_or_Z
        """
        exp_effective_u = np.exp(self.u_p[agent, self.dcs.C_or_Z, :][:, self.dcs.C_or_Z].sum(axis=0))

        return exp_effective_u / exp_effective_u.sum()

    def all_probabilities(self):
        """
        Compute all choice probabilities for every agent over items in C_or_Z
        :return: a matrix whose i, j entry is agent's i probability of picking item j
        """

        exp_effective_u = np.exp(self.u_p[:, self.dcs.C_or_Z][:, :, self.dcs.C_or_Z].sum(axis=1))
        return exp_effective_u / exp_effective_u.sum(axis=1, keepdims=True)

    @classmethod
    def random_instance(cls, n, k, m):
        """
        Construct a random CDM instance with n agent, k items, m alternatives, and utilities/contexts uniform over [0, 1].
        :param n: num agents
        :param k: num items
        :param m: num alternatives
        :return: a CDM instance
        """
        dcs = DiscreteChoiceSetting(
            list(range(n)),
            list(range(k + m)),
            list(range(k))
        )

        u_p = np.array([np.random.uniform(0, 1, (k + m, k + m)) for _ in range(n)])
        for matrix in u_p:
            np.fill_diagonal(matrix, 0)

        return CDM(dcs, u_p)


class NLNode:
    """
    Node for a nested logit (NL) tree
    """

    def __init__(self, *children, item=None, utility=-np.inf):
        """
        Build a new node with specified children and utility
        :param children: zero or more NLNodes
        :param item: an index into dcs.U
        :param utility:
        """
        self.parent = None
        self.active = False
        self.item = item
        self.children = list(children)
        self.utility = utility
        for child in self.children:
            child.parent = self

    def traverse(self):
        for child in self.children:
            yield from child.traverse()
        yield self

    def update_active(self, active_items):
        if self.is_leaf:
            self.active = self.item in active_items
        else:
            self.active = any([child.update_active(active_items) for child in self.children])

        return self.active

    def add_children(self, *children):
        for child in children:
            self.children.append(child)
            child.parent = self

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def height(self):
        return 1 + (0 if self.is_leaf else max(child.height for child in self.children))

    def __repr__(self):
        return f'Node(item={self.item}, utility={self.utility})'


class NL(DiscreteChoiceModel):
    """
    Nested logit model, introduced in:

    [3] McFadden, D. Modeling the choice of residential location. Transportation Research Record, (673), 1978.

    This is the generalized tree-based version, as used in:

    [4] Benson, A. R., Kumar, R., and Tomkins, A. On the relevance of irrelevant alternatives.
        In Proceedings of the 25th International Conference on World Wide Web, pp. 963– 973.
        International World Wide Web Conferences Steering Committee, 2016.
    """
    short_name = 'nl'

    def __init__(self, dcs, roots):
        super().__init__(dcs)

        self.roots = roots
        self._leaf_maps = [{node.item: node for node in self.roots[agent].traverse() if node.is_leaf}
                           for agent in range(len(self.dcs.A))]
        for agent in range(len(self.dcs.A)):
            self.roots[agent].update_active(self.dcs.C_or_Z)
        self._update_probabilities()

    def set_Z(self, Z):
        """
        Set the included alternatives to Z.
        :param Z: a list of indices into U
        """
        for item in self.dcs.Z:
            for agent in range(len(self.dcs.A)):
                self._leaf_maps[agent][item].active = False

        super().set_Z(Z)

        for item in self.dcs.Z:
            for agent in range(len(self.dcs.A)):
                self._leaf_maps[agent][item].active = True

        for agent in range(len(self.dcs.A)):
            self.roots[agent].update_active(self.dcs.C_or_Z)
        self._update_probabilities()

    def update_Z(self, i):
        """
        Add i to the included alternatives if it isn't one already, otherwise remove it
        :param i: an index into U
        """
        super().update_Z(i)

        for agent in range(len(self.dcs.A)):
            self._leaf_maps[agent][i].active ^= True  # flip active
            self.roots[agent].update_active(self.dcs.C_or_Z)  # could be optimized with selective updates
        self._update_probabilities()  # could also be optimized with selective updates

    def set_C(self, C):
        """
        Set the choice set to C, setting the included alternatives Z to [] in the process.
        :param C: a list of indices into U
        """
        super().set_C(C)
        for item in C:
            for agent in range(len(self.dcs.A)):
                self._leaf_maps[agent][item].active = True

        for agent in range(len(self.dcs.A)):
            self.roots[agent].update_active(self.dcs.C_or_Z)
        self._update_probabilities()

    def probability(self, agent, item):
        """
        Compute the probability that agent picks item
        :param agent: index into dcs.A
        :param item: index into dcs.U
        :return: Pr[agent <- ]
        """
        return self._probabilities[agent, item]

    def all_agent_probabilities(self, agent):
        """
        Compute all choice probabilities for agent
        :param agent: an index into dcs.A
        :return: an array of choice probabilities for C_or_Z
        """
        return self._probabilities[agent]

    def all_probabilities(self):
        """
        Compute all choice probabilities for every agent over items in C_or_Z
        :return: a matrix whose i, j entry is agent's i probability of picking item j
        """
        return self._probabilities

    @classmethod
    def random_instance(cls, n, k, m):
        """
        Construct a random instance with n agents, k items, m alternatives, and random preferences.
        :param n: num agents
        :param k: num items
        :param m: num alternatives
        :return: an instance of this DiscreteChoiceModel
        """
        pass

    def _update_probabilities(self):
        self._probabilities = np.zeros((len(self.dcs.A), len(self.dcs.C_or_Z)))
        C_or_Z_indices = {self.dcs.C_or_Z[i]: i for i in range(len(self.dcs.C_or_Z))}

        for agent in range(len(self.dcs.A)):
            stack = [(self.roots[agent], 1)]
            while len(stack) > 0:
                node, probability = stack.pop()
                if node.is_leaf:
                    self._probabilities[agent, C_or_Z_indices[node.item]] = probability
                else:
                    denominator = np.exp([child.utility for child in node.children if child.active]).sum()
                    for child in node.children:
                        if child.active:
                            stack.append((child, probability * np.exp(child.utility) / denominator))


if __name__ == '__main__':
    # Do not call optimize_choice_sets.approx methods from this file: isinstance() performs unexpectedly

    np.set_printoptions(precision=4, suppress=True)

    dcs = DiscreteChoiceSetting(
        ['a', 'b'],  # agents
        ['x', 'y', 'z'],  # items
        [0, 1]  # choice set is {'x', 'y'}
    )

    print('NL')
    root_a = NLNode(
        NLNode(item=0, utility=1),
        NLNode(
            NLNode(item=1, utility=2),
            NLNode(item=2, utility=1),
            utility=3)
    )

    root_b = NLNode(
        NLNode(
            NLNode(item=0, utility=1),
            NLNode(item=1, utility=2),
            utility=1),
        NLNode(item=2, utility=2),
    )

    nl = NL(dcs, roots=[root_a, root_b])  # NOTE: beware when sharing a DCS with another model

    print('Choice probabilities over C = {x, y}:')
    print(nl.all_probabilities())
    print(f'Initial disagreement: {nl.disagreement():.4f}')
    nl.update_Z(2)
    print(f'Disagreement if we include item z: {nl.disagreement():.4f}')

    print('\nCDM')
    u_p = np.array([
        np.array([  # utility-adjusted context effects for agent 'a'
            [0, 2, 0],
            [1, 0, 3],
            [1, 0, 0]
        ]),
        np.array([  # utility-adjusted context effects for agent 'b'
            [0, 1, 3],
            [3, 0, 0],
            [1, 8, 0]
        ])
    ])

    cdm = CDM(dcs, u_p)
    cdm.set_Z([])

    print('Choice probabilities over C = {x, y}:')
    print(cdm.all_probabilities())

    print(f'Initial disagreement: {cdm.disagreement():.4f}')

    cdm.update_Z(2)
    print(f'Disagreement if we include item z: {cdm.disagreement():.4f}')
