import os
import pickle as pkl
import networkx as nx
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from data_utils import rand_train_test_idx, split_data

DATAPATH = "../../data/"


class NCDataset(object):
    def __init__(self, name, root=f'{DATAPATH}'):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None,
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25):
        """
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}

        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))


def load_nc_dataset(args):
    """ Loader for NCDataset
        Returns NCDataset
    """
    global DATAPATH

    DATAPATH = args.data_dir
    dataname = args.dataset
    print('>> Loading dataset: {}'.format(dataname))
    if dataname in ('cora', 'citeseer', 'pubmed'):
        dataset = load_planetoid_dataset(dataname, args.no_feat_norm)
    elif dataname == 'airport':
        dataset = load_airport_dataset()
    else:
        raise ValueError('PHDM node-classification datasets are airport, cora, citeseer and pubmed.')
    return dataset


def load_airport_dataset():
    filename = 'airport'
    dataset = NCDataset(filename)
    graph = pkl.load(open(os.path.join(DATAPATH, 'hgcn_data', 'airport', 'airport.p'), 'rb'))
    adj = nx.adjacency_matrix(graph)
    features = np.array([graph._node[u]['feat'] for u in graph.nodes()])
    label_idx = 4
    labels = features[:, label_idx]
    features = features[:, :label_idx]
    labels = bin_feat(labels, bins=[7.0 / 7, 8.0 / 7, 9.0 / 7])
    num_nodes = adj.shape[0]
    features = torch.tensor(features, dtype=torch.float)
    val_prop, test_prop = 0.15, 0.15
    idx_val, idx_test, idx_train = split_data(labels, val_prop, test_prop)
    deg = np.squeeze(np.sum(adj, axis=0).astype(int))
    deg[deg > 5] = 5
    deg_onehot = torch.tensor(np.eye(6)[deg], dtype=torch.float).squeeze()
    const_f = torch.ones(features.size(0), 1)
    features = torch.cat((features, deg_onehot, const_f), dim=1)
    edge_index = torch.tensor(adj.nonzero(), dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)
    dataset.graph = {'edge_index': edge_index,
                     'edge_feat': None,
                     'node_feat': features,
                     'num_nodes': num_nodes}
    dataset.label = labels
    dataset.train_idx = idx_train
    dataset.valid_idx = idx_val
    dataset.test_idx = idx_test
    return dataset


def bin_feat(feat, bins):
    digitized = np.digitize(feat, bins)
    return digitized - digitized.min()


def load_planetoid_dataset(name, no_feat_norm=False):
    # import pdb
    # pdb.set_trace()
    if not no_feat_norm:
        transform = T.NormalizeFeatures()
        torch_dataset = Planetoid(root=f'{DATAPATH}/Planetoid',
                                  name=name, transform=transform)
    else:
        torch_dataset = Planetoid(root=f'{DATAPATH}/Planetoid', name=name)
    data = torch_dataset[0]

    edge_index = data.edge_index
    node_feat = data.x
    label = data.y
    num_nodes = data.num_nodes
    print(f"Num nodes: {num_nodes}")

    dataset = NCDataset(name)

    dataset.train_idx = torch.where(data.train_mask)[0]
    dataset.valid_idx = torch.where(data.val_mask)[0]
    dataset.test_idx = torch.where(data.test_mask)[0]

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset
