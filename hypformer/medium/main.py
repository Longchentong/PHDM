import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils import class_rand_splits, eval_acc, evaluate, load_fixed_splits
from dataset import load_nc_dataset
from logger import Logger
from parse import parse_method, parser_add_main_args
from sklearn.neighbors import kneighbors_graph

from manifolds.hyp_layer import Optimizer
from training import make_schedulers, step_schedulers

warnings.filterwarnings('ignore')


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def tensor_digest(items):
    digest = hashlib.sha256()
    for name, value in items:
        value = value.detach().cpu().contiguous()
        digest.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split(split_idx, num_nodes, require_disjoint=True):
    indices = {name: value.detach().cpu().tolist() for name, value in split_idx.items()}
    sets = {name: set(value) for name, value in indices.items()}
    for name, value in indices.items():
        if not value:
            raise ValueError(f'Empty {name} split')
        if require_disjoint and len(value) != len(sets[name]):
            raise ValueError(f'Duplicate {name} split')
        if min(value) < 0 or max(value) >= num_nodes:
            raise ValueError(f'Invalid node index in {name} split')
    for left, right in [('train', 'valid'), ('train', 'test'), ('valid', 'test')]:
        if require_disjoint and sets[left] & sets[right]:
            raise ValueError(f'Overlapping {left}/{right} splits')


### Parse args ###
parser = argparse.ArgumentParser(description='Medium Data Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
if args.epochs < 1 or args.runs < 1:
    parser.error('epochs and runs must be positive')
if not 0 <= args.selected_run <= args.runs:
    parser.error('selected_run must be between 0 and runs')
if not 0 < args.lr_min_ratio <= 1 or not 0 < args.lr_decay_factor < 1:
    parser.error('invalid learning rate decay or minimum ratio')
if args.lr_decay_patience < 0 or args.lr_cosine_epochs < 1 or args.grad_clip < 0:
    parser.error('invalid scheduler budget or gradient clipping threshold')
if not 0 <= args.label_smoothing < 1:
    parser.error('label smoothing must be in [0, 1)')
os.environ['HYPFORMER_INPUT_MAP'] = args.input_map
selection_policy = 'validation_accuracy_then_loss' if args.val_tie_break else 'validation_accuracy'
print('====' * 20)
print(args)
fix_seed(args.seed)

if args.cpu:
    device = torch.device("cpu")
    print('>> Using CPU')
else:
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; pass --cpu 1 for an explicit CPU run')
    device = torch.device("cuda:" + str(args.device))
    print('>> Using GPU: ' + torch.cuda.get_device_name(device))

### Load and preprocess data ###
dataset = load_nc_dataset(args)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

dataset_name = args.dataset

if args.rand_split:
    print('>> loading random splits ...')
    split_idx_lst = [dataset.get_idx_split(train_prop=args.train_prop, valid_prop=args.valid_prop)
                     for _ in range(args.runs)]
elif args.rand_split_class:
    print('>> loading random class splits ...')
    split_idx_lst = [class_rand_splits(
        dataset.label, args.label_num_per_class, args.valid_num, args.test_num)]
else:
    print('>> loading fixed splits ...')
    split_idx_lst = load_fixed_splits(
        dataset, name=args.dataset, protocol=args.protocol)

if args.dataset in ('mini', '20news'):
    adj_knn = kneighbors_graph(dataset.graph['node_feat'], n_neighbors=args.knn_num, include_self=True)
    edge_index = torch.tensor(adj_knn.nonzero(), dtype=torch.long)
    dataset.graph['edge_index'] = edge_index

n = dataset.graph['num_nodes']
num_class = max(dataset.label.max().item() + 1, dataset.label.shape[1])
num_class = (int)(num_class)
node_feat_dim = dataset.graph['node_feat'].shape[1]
args.in_channels = node_feat_dim
args.out_channels = num_class

data_sha256 = tensor_digest([
    ('features', dataset.graph['node_feat']),
    ('edges', dataset.graph['edge_index']),
    ('labels', dataset.label),
])

dataset.graph['edge_index'] = dataset.graph['edge_index'].to(device),
dataset.graph['node_feat'] = dataset.graph['node_feat'].to(device)

print(f">> num nodes {n} | num classes {num_class} | num node feats {node_feat_dim}")

if args.dataset in ('deezer-europe'):
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.NLLLoss()
train_criterion = (nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
                   if args.label_smoothing else criterion)

eval_func = eval_acc

# ===============================================================================
logger = Logger(args.runs, args)
run_records = []
results = None
checkpoint_dir = Path(args.checkpoint_dir).resolve()
checkpoint_dir.mkdir(parents=True, exist_ok=True)
result_file = Path(args.result_file).resolve()
result_file.parent.mkdir(parents=True, exist_ok=True)
for run in range(args.runs):
    print(f'Run {run + 1}/{args.runs}')
    report_run = not args.selected_run or run + 1 == args.selected_run
    evaluate_test = bool(args.evaluate_test) and report_run
    if args.dataset in ['cora', 'citeseer', 'pubmed', 'airport', 'disease'] and args.protocol == 'semi':
        split_idx = split_idx_lst[0]
    else:
        split_idx = split_idx_lst[run]
    # Preserve the official Airport indices, including their repetitions.
    official_airport_split = (args.dataset == 'airport' and args.protocol == 'semi'
                              and not args.rand_split and not args.rand_split_class)
    validate_split(split_idx, n, require_disjoint=not official_airport_split)
    train_idx = split_idx['train'].to(device)  # get train split
    model = parse_method(args, device)  # load model
    optimizer = Optimizer(model, args)  # load optimizer
    schedulers = make_schedulers(optimizer, args)
    training_history = []

    best_val = float('-inf')
    best_val_loss = float('inf')
    best_state = None
    best_epoch = None
    started = time.monotonic()
    patience = 0
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        emb = None
        out = model(dataset)
        out = F.log_softmax(out, dim=1)
        loss = train_criterion(
            out[train_idx], dataset.label.squeeze(1)[train_idx])
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f'Non-finite loss in run {run + 1}, epoch {epoch + 1}')
        loss.backward()
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
        optimizer.step()

        result = evaluate(model, dataset, split_idx, eval_func, criterion, args,
                          evaluate_test=False)
        if not math.isfinite(float(result[3])):
            raise FloatingPointError(f'Non-finite validation loss in run {run + 1}')
        logger.add_result(run, (*result[:3], float(result[3])))
        training_history.append({'epoch': epoch + 1, 'loss': float(loss.item()),
                                 'learning_rates': [group['lr'] for item in optimizer.optimizer
                                                    for group in item.param_groups]})
        step_schedulers(schedulers, args, float(result[3]))

        higher_accuracy = result[1] > best_val
        lower_loss_tie = (args.val_tie_break and result[1] == best_val
                          and float(result[3]) < best_val_loss)
        if higher_accuracy or lower_loss_tie:
            best_val = result[1]
            best_val_loss = float(result[3])
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone()
                          for name, value in model.state_dict().items()}
        # Keep the upstream accuracy-based stopping budget unchanged.
        if higher_accuracy:
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

        if epoch % args.display_step == 0:
            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%')

    if best_state is None:
        raise RuntimeError('No finite validation-selected checkpoint was produced')
    split_sha256 = tensor_digest(sorted(split_idx.items()))
    checkpoint_path = checkpoint_dir / f'run{run + 1}_val_best.pt'
    torch.save({'model_state_dict': best_state, 'epoch': best_epoch + 1,
                'selection_policy': selection_policy, 'args': vars(args),
                'validation_history': [
                    {'epoch': index + 1, 'accuracy': 100 * row[1], 'loss': float(row[3])}
                    for index, row in enumerate(logger.results[run])],
                'training_history': training_history,
                'data_sha256': data_sha256, 'split_sha256': split_sha256}, checkpoint_path)
    selected = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(selected['model_state_dict'])
    result = evaluate(model, dataset, split_idx, eval_func, criterion, args,
                      evaluate_test=evaluate_test)
    checked_metrics = (*result[:2], result[3])
    if evaluate_test:
        checked_metrics = (*checked_metrics, result[2])
    if not all(math.isfinite(float(value)) for value in checked_metrics):
        raise FloatingPointError('Non-finite selected-checkpoint evaluation')
    logger.results[run][best_epoch] = (*result[:3], float(result[3]))
    logger.selected_epochs[run] = best_epoch
    record = {
        'run': run + 1, 'base_seed': args.seed,
        'rng_policy': 'continuous_from_base_seed',
        'selection_policy': selection_policy, 'input_map': args.input_map,
        'lr_schedule': args.lr_schedule, 'grad_clip': args.grad_clip,
        'label_smoothing': args.label_smoothing, 'stable_focusing': args.stable_focusing,
        'best_epoch': best_epoch + 1, 'epochs_trained': epoch + 1,
        'train_accuracy': 100 * result[0], 'validation_accuracy': 100 * result[1],
        'validation_loss': float(result[3]),
        'test_accuracy': 100 * result[2] if evaluate_test else None,
        'test_evaluations': int(evaluate_test),
        'checkpoint': str(checkpoint_path), 'checkpoint_sha256': file_digest(checkpoint_path),
        'data_sha256': data_sha256, 'split_sha256': split_sha256,
        'split_sizes': {key: len(value) for key, value in split_idx.items()},
        'device': str(device), 'elapsed_seconds': time.monotonic() - started,
    }
    if report_run:
        run_records.append(record)
        temporary = result_file.with_suffix(result_file.suffix + '.tmp')
        temporary.write_text(json.dumps(run_records, indent=2) + '\n')
        temporary.replace(result_file)
        print('PHDM_RUN_RESULT: ' + json.dumps(record, sort_keys=True), flush=True)
    else:
        print('PHDM_PREPARATION_RUN_RESULT: ' + json.dumps(record, sort_keys=True), flush=True)
    if evaluate_test:
        logger.print_statistics(run)
    else:
        print(f'Validation-only run {run + 1}: epoch {best_epoch + 1}, '
              f'accuracy {100 * result[1]:.2f}', flush=True)

    if args.output_attention:
        attentions = model.get_attentions(dataset.graph['node_feat'].to(device))
        np.save(f'results/att/{args.dataset}_{args.method}_{args.hidden_channels}_attentions.npy', attentions.detach().cpu().numpy())
    # delete the model and optimizer and start a new run
    del model, optimizer

if args.runs > 1 and args.evaluate_test and not args.selected_run:
    results = logger.print_statistics()
    print('====================')
    print(results)
    print('====================')


out_folder = 'results'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

if args.save_result:
    mkdirs(f'results/{args.dataset}')
    csvfilename = f'results/{args.dataset}/{args.dataset}_{args.method}_{args.hidden_channels}.csv'
    logger.save(vars(args), results, csvfilename)
