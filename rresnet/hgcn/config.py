import argparse

from utils.train_utils import add_flags_from_config

config_args = {
    'training_config': {
        'lr': (0.01, 'learning rate'),
        'dropout': (0.0, 'dropout probability'),
        'cuda': (-1, 'which cuda device to use (-1 for cpu training)'),
        'epochs': (5000, 'maximum number of epochs to train for'),
        'weight-decay': (0., 'l2 regularization strength'),
        'optimizer': ('Adam', 'Euclidean optimizer for the neural vector-field parameters'),
        'momentum': (0.999, 'momentum in optimizer'),
        'patience': (100, 'patience for early stopping'),
        'seed': (1234, 'seed for training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (1, 'how often to compute val metrics (in epochs)'),
        'save': (0, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'sweep-c': (0, ''),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.5, 'gamma for lr scheduler'),
        'print-epoch': (True, ''),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'min-epochs': (100, 'do not early stop before min-epochs'),
        'metrics-out': (None, 'optional path for an atomic structured result JSON'),
        'checkpoint-out': (None, 'path for the validation-selected model checkpoint'),
        'evaluate-test': (1, 'evaluate test once after restoring the selected checkpoint'),
        'val-tie-break': (1, 'prefer lower validation loss when validation scores tie'),
        'run-id': (None, 'optional stable experiment run identifier'),
        'fail-on-nonfinite': (1, 'abort when non-finite tensors, losses, or gradients are detected')
    },
    'model_config': {
        'task': ('lp', 'Cora link prediction'),
        'model': ('RRNetHyperbolicLorentz', 'PHDM RResNet encoder selected by scripts/run_table6.sh'),
        'dim': (128, 'embedding dimension'),
        'manifold': ('Hyperboloid', 'Hyperboloid, PoincareBall, Sphere or ProjectedSphere'),
        'c': (1.0, 'hyperbolic radius, set to None for trainable curvature'),
        'r': (2., 'fermi-dirac decoder parameter for lp'),
        't': (1., 'fermi-dirac decoder parameter for lp'),
        'pretrained-embeddings': (None, 'path to pretrained embeddings (.npy file) for Shallow node classification'),
        'pos-weight': (0, 'whether to upweight positive class in node classification tasks'),
        'num-layers': (2, 'number of hidden layers in encoder'),
        'bias': (1, 'whether to use bias (1) or not (0)'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'n-heads': (4, 'number of attention heads for graph attention networks, must be a divisor dim'),
        'alpha': (0.2, 'alpha for leakyrelu in graph attention networks'),
        'double-precision': ('0', 'whether to use double precision'),
        'use-att': (0, 'whether to use hyperbolic attention or not'),
        'local-agg': (0, 'whether to local tangent space aggregation or not'),
        # added for RResNet
        'hdim': (128, 'hidden dimension within RResNet blocks'),
        'num-blocks': (5, 'number of RResNet blocks'),
        'num-horospheres': (250, 'number of horosphere features, if relevant for a hyperbolic riemannian resnet'),
        'proj-type': ('lorentz_fc', 'PHDM input projection: poincare_fc, lorentz_fc, psphere_fc or sphere_exact'),
        'phdm-scale': (1.0, 'fixed scale applied to Euclidean logits before the PHDM input map'),
        'phdm-init-scale': (1.0, 'fixed multiplier applied to PHDM input-layer weights at initialization'),
        'adj-power': (2, 'power of adjacency multiplication used in graph-based RResNet encoders'),
        'graph-mix-mode': ('sum', 'how graph-smoothed features are mixed with node features in graph RResNet encoders: sum, concat, or gate'),
        'graph-aggregation-mode': ('ambient', 'graph aggregation mode for graph RResNet encoders: ambient, self_only, or sphere_local'),
        'graph-neighbor-scale': (1.0, 'scale of the graph neighbor correction term for sphere_local graph aggregation'),
    },
    'data_config': {
        'dataset': ('cora', 'which dataset to use'),
        'val-prop': (0.05, 'proportion of validation edges for link prediction'),
        'test-prop': (0.1, 'proportion of test edges for link prediction'),
        'use-feats': (1, 'whether to use node features or not'),
        'normalize-feats': (1, 'whether to normalize input node features'),
        'normalize-adj': (1, 'whether to row-normalize the adjacency matrix'),
        'split-seed': (1234, 'seed for data splits (train/test/val)'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
