"""
PyTorch inference code was adapted from Arjun Seshadri's code (https://github.com/arjunsesh/cdm-icml),
accompanying the paper:

[1] Seshadri, A., Peysakhovich, A., and Ugander, J. Discovering context effects from raw choice data.
        In International Conference on Machine Learning, pp. 5660–5669, 2019.
"""
from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
from tqdm import tqdm

from choice_models import NLNode


class DataLoader:
    """
    Simplified, faster DataLoader.
    From https://github.com/arjunsesh/cdm-icml with minor tweaks.
    """
    def __init__(self, data, batch_size=None, shuffle=False):
        self.data = data
        self.data_size = data[0].shape[0]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.counter = 0
        self.stop_iteration = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop_iteration:
            self.stop_iteration = False
            raise StopIteration()

        if self.batch_size is None or self.batch_size == self.data_size:
            self.stop_iteration = True
            return self.data
        else:
            i = self.counter
            bs = self.batch_size
            self.counter += 1
            batch = [item[i * bs:(i + 1) * bs] for item in self.data]
            if self.counter * bs >= self.data_size:
                self.counter = 0
                self.stop_iteration = True
                if self.shuffle:
                    random_idx = np.arange(self.data_size)
                    np.random.shuffle(random_idx)
                    self.data = [item[random_idx] for item in self.data]
            return batch


class Embedding(nn.Module):
    """
    Add zero-ed out dimension to Embedding for the padding index.
    From https://github.com/arjunsesh/cdm-icml with minor tweaks.
    """

    def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx

        self.weight = nn.Parameter(
            torch.randn([self.num_embeddings, self.embedding_dim]) / np.sqrt(self.num_embeddings)
        )

        with torch.no_grad():
            self.weight[self.padding_idx].fill_(0)

    def forward(self, x):
        with torch.no_grad():
            self.weight[self.padding_idx].fill_(0)

        return self.weight[x]


class TorchChoiceModel(nn.Module, ABC):
    """
    Base class for TorchLowRankCDM and TorchMNL.
    """

    @abstractmethod
    def forward(self, x, x_lengths):
        """
        Compute log(choice probabilities) of items in choice sets
        :param x: the choice sets
        :param x_lengths: the number of items in each choice set
        :return: log(choice probabilities) over every choice set
        """
        pass

    def loss(self, y_hat, y):
        """
        The error in inferred log-probabilities given observations
        :param y_hat: log(choice probabilities)
        :param y: observed choices
        :return: the loss
        """
        return nnf.nll_loss(y_hat, y[:, None])

    def accuracy(self, y_hat, y):
        """
        Compute accuracy (fraction of choice set correctly predicted)
        :param y_hat: log(choice probabilities)
        :param y: observed choices
        :return: the accuracy
        """
        return (y_hat.argmax(1).int() == y[:, None].int()).float().mean()


class TorchLowRankCDM(TorchChoiceModel):
    """
    Implementation of low-rank CDM [1].
    From https://github.com/arjunsesh/cdm-icml with minor tweaks.
    """

    def __init__(self, num_items, rank):
        """
        Initialize a low rank CDM model for inference
        :param num_items: size of U
        :param rank: the rank of the CDM
        """
        super().__init__()
        self.num_items = num_items
        self.embedding_dim = rank

        self.target_embedding = Embedding(
            num_embeddings=self.num_items + 1,
            embedding_dim=self.embedding_dim,
            padding_idx=self.num_items
        )

        self.context_embedding = Embedding(
            num_embeddings=self.num_items + 1,
            embedding_dim=self.embedding_dim,
            padding_idx=self.num_items
        )

    def forward(self, x, x_lengths):
        """
        Compute log(choice probabilities) of items in choice sets
        :param x: the choice sets
        :param x_lengths: the number of items in each choice set
        :return: log(choice probabilities) over every choice set
        """
        batch_size, seq_len = x.size()
        context_vecs = self.context_embedding(x)

        context_sums = context_vecs.sum(-2, keepdim=True) - context_vecs
        utilities = (self.target_embedding(x) * context_sums).sum(-1, keepdim=True)
        utilities[torch.arange(seq_len)[None, :] >= x_lengths[:, None]] = -np.inf

        return nnf.log_softmax(utilities, 1)

    def get_u_p(self):
        """
        Return the utility-adjust context effect matrix defined by the current target and context embeddings
        :return: the u_p matrix
        """

        context_vecs = self.context_embedding(torch.arange(self.num_items))
        target_vecs = self.target_embedding(torch.arange(self.num_items))

        u_p = np.zeros((self.num_items, self.num_items))

        for i in range(self.num_items):
            for j in range(self.num_items):
                if i != j:
                    u_p[i, j] = (context_vecs[i] * target_vecs[j]).sum(-1)  # u_p is the pull of i on j

        return u_p


class TorchMNL(TorchChoiceModel):
    """
    Modification of above class for MNL.
    """

    def __init__(self, num_items):
        """
        Initialize an MNL model for inference
        :param num_items: size of U
        """
        super().__init__()
        self.num_items = num_items

        self.utilities = Embedding(
            num_embeddings=self.num_items + 1,
            embedding_dim=1,
            padding_idx=self.num_items
        )

    def forward(self, x, x_lengths):
        """
        Compute log(choice probabilities) of items in choice sets
        :param x: the choice sets
        :param x_lengths: the number of items in each choice set
        :return: log(choice probabilities) over every choice set
        """
        batch_size, seq_len = x.size()

        utilities = self.utilities(x)
        utilities[torch.arange(seq_len)[None, :] >= x_lengths[:, None]] = -np.inf

        return nnf.log_softmax(utilities, 1)

    def get_utilities(self):
        """
        Return the inferred utilies
        :return: a numpy array of item utilities
        """
        return torch.flatten(self.utilities(torch.arange(self.num_items))).detach().numpy()


class TorchNL(TorchChoiceModel):
    """
    Modification of above classes for NL.
    """

    def __init__(self, num_items, tree):
        """
        Initialize an NL model for inference
        :param num_items: size of U
        """
        super().__init__()
        self.num_items = num_items
        self.tree = tree

        tree_traversal = list(tree.traverse())
        self.index_to_node_map = {i: node for i, node in enumerate(tree_traversal)}
        self.node_to_index_map = {node: i for i, node in enumerate(tree_traversal)}
        self.leaf_map = {node.item: node for node in tree_traversal if node.is_leaf}
        self.num_nodes = len(self.node_to_index_map)

        self.leaf_ancestor_matrix = torch.full((num_items + 1, self.num_nodes), -np.inf)
        for node in self.leaf_map.values():
            current = node
            while current.parent is not None:
                self.leaf_ancestor_matrix[node.item, self.node_to_index_map[current]] = 1
                current = current.parent

        self.inf_adjacency_matrix = torch.full((self.num_nodes, self.num_nodes), -np.inf)
        for node, index in self.node_to_index_map.items():
            for child in node.children:
                self.inf_adjacency_matrix[index, self.node_to_index_map[child]] = np.inf

        self.utilities = Embedding(
            num_embeddings=self.num_nodes + 1,
            embedding_dim=1,
            padding_idx=self.num_items
        )

    def forward(self, x, x_lengths):
        """
        Compute log(choice probabilities) of items in choice sets
        :param x: the choice sets
        :param x_lengths: the number of items in each choice set
        :return: log(choice probabilities) over every choice set
        """
        num_obervations, seq_len = x.size()

        utilities = self.utilities(torch.arange(self.num_nodes)[None, :].repeat(num_obervations, 1))

        # Give inactive nodes -inf utility
        activations = self.leaf_ancestor_matrix[x].max(1)[0][:, :, None]
        utilities[activations == -np.inf] = -np.inf

        # Compute log probability that each node is chosen from among its siblings
        sibling_uts = torch.min(utilities, self.inf_adjacency_matrix.t())
        node_probs = sibling_uts.log_softmax(1)
        nan_to_inf_node_probs = torch.where(~torch.isnan(node_probs), node_probs, torch.tensor(-np.inf))
        repeated_node_probs = nan_to_inf_node_probs.max(2)[0][:, None, :].repeat(1, self.num_items + 1, 1)

        # Compute log probability that each item is chosen by summing node probs up the tree
        item_probs = torch.min(repeated_node_probs, self.leaf_ancestor_matrix)
        item_probs[item_probs == -np.inf] = 0
        item_probs = item_probs.sum(2)

        # Fill in probabilities for items in the observed choice sets
        obs_probs = item_probs.gather(1, x)[:, :, None]
        obs_probs[torch.arange(seq_len)[None, :] >= x_lengths[:, None]] = -np.inf

        return obs_probs

    def populate_tree_utilities(self):
        """
        Put the inferred utilities into the tree
        """
        for index in range(self.num_nodes):
            self.index_to_node_map[index].utility = self.utilities(index).item()


def fit(model, data, epochs=500, learning_rate=5e-2, weight_decay=2.5e-4, show_live_loss=False):
    """
    Fit a choice model to data using the given optimizer.

    :param model: a nn.Module
    :param data:
    :param epochs: number of optimization epochs
    :param learning_rate: step size hyperparameter for Adam
    :param weight_decay: regularization hyperparameter for Adam
    :param show_live_loss: if True, add loss/accuracy to progressbar. Adds ~50% overhead
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True, weight_decay=weight_decay)
    data_loader = DataLoader(data, batch_size=128, shuffle=True)
    progress_bar = tqdm(range(epochs), total=epochs)
    for epoch in progress_bar:
        model.train()
        accuracies = []
        losses = []
        for choice_sets, choice_set_sizes, choices in data_loader:
            loss = model.loss(model(choice_sets, choice_set_sizes), choices)
            loss.backward(retain_graph=None if epoch != epochs - 1 else True)
            optimizer.step()
            optimizer.zero_grad()

            if show_live_loss:
                model.eval()
                with torch.no_grad():
                    accuracy = model.accuracy(model(choice_sets, choice_set_sizes), choices)
                losses.append(loss.detach())
                accuracies.append(accuracy)
        if show_live_loss:
            progress_bar.set_description(f'Loss: {np.mean(losses):.4f}, Accuracy: {np.mean(accuracies):.4f}. Epochs')

    loss.backward()
    with torch.no_grad():
        gradient = torch.stack([(item.grad ** 2).sum() for item in model.parameters()]).sum()
    print('Done. Final gradient:', gradient.item())


if __name__ == '__main__':
    # Example usage
    torch_choice_sets = torch.tensor(  # items are 0, 1, 2, 3. 4 is a padding index
        [[0, 1, 2, 4],
         [0, 2, 4, 4],
         [1, 3, 4, 4]]
    )

    torch_choice_set_sizes = torch.tensor(  # size of each torch_choice_set
        [3, 2, 2]
    )

    torch_choices = torch.tensor(  # indices into each torch_choice_set
        [0, 1, 1]
    )

    tree = NLNode(
        NLNode(
            NLNode(item=0),
            NLNode(item=2)
        ),
        NLNode(
            NLNode(item=1),
            NLNode(item=3),
        )
    )

    torch.manual_seed(0)

    print('Fitting MNL...')
    torch_model = TorchMNL(4)
    fit(torch_model, (torch_choice_sets, torch_choice_set_sizes, torch_choices), show_live_loss=True)
    utilities = torch_model.get_utilities()
    print('Inferred MNL utilities:\n', utilities, '\n')

    print('Fitting rank 2 CDM...')
    torch_model = TorchLowRankCDM(4, 2)
    fit(torch_model, (torch_choice_sets, torch_choice_set_sizes, torch_choices), show_live_loss=True)
    u_p = torch_model.get_u_p()
    print('Inferred rank 2 CDM parameter matrix:\n', u_p, '\n')

    print('Fitting rank 10 CDM...')
    torch_model = TorchLowRankCDM(4, 10)
    fit(torch_model, (torch_choice_sets, torch_choice_set_sizes, torch_choices), show_live_loss=True)
    u_p = torch_model.get_u_p()
    print('Inferred rank 10 CDM parameter matrix:\n', u_p, '\n')

    print('Fitting NL...')
    torch_model = TorchNL(4, tree)
    fit(torch_model, (torch_choice_sets, torch_choice_set_sizes, torch_choices), show_live_loss=True)
    torch_model.populate_tree_utilities()
    print('Inferred NL utilities:\n', list(tree.traverse()), '\n')
