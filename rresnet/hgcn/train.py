from __future__ import division
from __future__ import print_function

import datetime
import hashlib
import json
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np
import optimizers
import torch
from config import parser
from models.base_models import LPModel
from utils.data_utils import load_data
from utils.train_utils import get_dir_name, format_metrics


def metric_values(metrics):
    values = {}
    for key, value in metrics.items():
        if torch.is_tensor(value):
            value = value.detach().cpu().item()
        values[key] = float(value)
    return values


def collect_phdm_diagnostics(model):
    diagnostics = {}
    for name, module in model.encoder.named_modules():
        getter = getattr(module, 'get_diagnostics', None)
        if getter is None:
            continue
        values = getter()
        if values:
            diagnostics['module'] = name or module.__class__.__name__
            diagnostics.update({key: float(value) for key, value in values.items()})
    return diagnostics


def collect_geometry_diagnostics(model, embeddings):
    encoder = model.encoder
    manifold = getattr(encoder, 'manifold', getattr(encoder, 'rr_manifold', None))
    if manifold is None:
        return {}

    name = manifold.__class__.__name__
    result = {
        'manifold_class': name,
        'embedding_abs_max': float(embeddings.detach().abs().max().cpu()),
    }
    with torch.no_grad():
        if name == 'Poincare' and hasattr(manifold, '_curvature'):
            c = manifold._curvature(embeddings)
            scaled_norm_sq = c * embeddings.pow(2).sum(dim=-1)
            result['max_c_norm_sq'] = float(scaled_norm_sq.max().cpu())
            result['min_ball_margin'] = float((1.0 - scaled_norm_sq).min().cpu())
        elif name == 'Lorentz' and hasattr(manifold, 'minkowski_dot'):
            c = manifold._curvature(embeddings).clamp_min(1e-15)
            constraint = manifold.minkowski_dot(embeddings, embeddings) + c.reciprocal()
            result['max_constraint_residual'] = float(constraint.abs().max().cpu())
        elif name in {'Sphere', 'SphereManifold'} and hasattr(manifold, '_curvature'):
            c = manifold._curvature(embeddings).clamp_min(1e-15)
            constraint = embeddings.pow(2).sum(dim=-1) - c.reciprocal()
            result['max_constraint_residual'] = float(constraint.abs().max().cpu())
        elif name == 'ProjectedSphere':
            result['max_chart_norm'] = float(embeddings.norm(dim=-1).max().cpu())
    return result


def write_result(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, target)


def check_finite(name, tensor, enabled):
    if enabled and not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite values detected in {name}.")


def tensor_digest(tensors):
    result = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu()
        result.update(f'{name}:{value.dtype}:{tuple(value.shape)}'.encode())
        if value.is_sparse:
            value = value.coalesce()
            result.update(value.indices().contiguous().numpy().tobytes())
            result.update(value.values().contiguous().numpy().tobytes())
        else:
            result.update(value.contiguous().numpy().tobytes())
    return result.hexdigest()


def validation_score(args, metrics):
    if args.task == 'lp':
        return 0.5 * (float(metrics['roc']) + float(metrics['ap']))
    return float(metrics['f1'])


def train(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if int(args.double_precision):
        torch.set_default_dtype(torch.float64)
    if int(args.cuda) >= 0:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    if args.epochs < 1 or args.eval_freq < 1:
        raise ValueError('Epoch and evaluation counts must be positive.')
    if args.evaluate_test not in (0, 1) or args.val_tie_break not in (0, 1):
        raise ValueError('Evaluation and validation-tie flags must be 0 or 1.')
    args.device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    args.patience = args.epochs if not args.patience else  int(args.patience)
    logging.getLogger().setLevel(logging.INFO)
    if args.save:
        if not args.save_dir:
            dt = datetime.datetime.now()
            date = f"{dt.year}_{dt.month}_{dt.day}"
            models_dir = os.path.join(os.environ['LOG_DIR'], args.task, date)
            save_dir = get_dir_name(models_dir)
        else:
            save_dir = args.save_dir
            os.makedirs(save_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO,
                            handlers=[
                                logging.FileHandler(os.path.join(save_dir, 'log.txt')),
                                logging.StreamHandler()
                            ])

    logging.info(f'Using: {args.device}')
    logging.info("Using seed {}.".format(args.seed))
    logging.info("Using split seed {}.".format(args.split_seed))

    # Load data
    data = load_data(args, os.path.join(os.environ['DATAPATH'], args.dataset))
    data_sha256 = tensor_digest({key: data[key] for key in ('features', 'adj_train_norm')})
    split_keys = ('train_edges', 'train_edges_false', 'val_edges', 'val_edges_false',
                  'test_edges', 'test_edges_false') if args.task == 'lp' else ('idx_train', 'idx_val', 'idx_test')
    split_sha256 = tensor_digest({key: torch.as_tensor(data[key]) for key in split_keys})
    split_sizes = {key: len(data[key]) for key in split_keys}
    args.n_nodes, args.feat_dim = data['features'].shape
    args.nb_false_edges = len(data['train_edges_false'])
    args.nb_edges = len(data['train_edges'])
    Model = LPModel

    if not args.lr_reduce_freq:
        args.lr_reduce_freq = args.epochs

    # Model and optimizer
    model = Model(args)
    logging.info(str(model))
    optimizer = getattr(optimizers, args.optimizer)(params=model.parameters(), lr=args.lr,
                                                    weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(args.lr_reduce_freq),
        gamma=float(args.gamma)
    )
    tot_params = int(sum(p.numel() for p in model.parameters()))
    logging.info(f"Total number of parameters: {tot_params}")
    if args.cuda is not None and int(args.cuda) >= 0 :
        model = model.to(args.device)
        for x, val in data.items():
            if torch.is_tensor(data[x]):
                data[x] = data[x].to(args.device)
        torch.cuda.reset_peak_memory_stats(model.c.device)
    # Train model
    t_total = time.time()
    counter = 0
    best_val_metrics = model.init_metric_dict()
    best_test_metrics = None
    best_emb = None
    best_state = None
    best_epoch = None
    best_phdm_diagnostics = {}
    best_geometry_diagnostics = {}
    history = []
    last_grad_norm = None
    fail_on_nonfinite = bool(int(args.fail_on_nonfinite))
    epochs_ran = 0
    for epoch in range(args.epochs):
        epochs_ran = epoch + 1
        t = time.time()
        model.train()
        optimizer.zero_grad()
        embeddings = model.encode(data['features'], data['adj_train_norm'])
        check_finite('training embeddings', embeddings, fail_on_nonfinite)
        train_metrics = model.compute_metrics(embeddings, data, 'train')
        check_finite('training loss', train_metrics['loss'], fail_on_nonfinite)
        train_metrics['loss'].backward()
        all_params = [param for param in model.parameters() if param.grad is not None]
        if args.grad_clip is not None:
            max_norm = float(args.grad_clip)
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(all_params, float('inf'))
        last_grad_norm = float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm)
        if fail_on_nonfinite and not np.isfinite(last_grad_norm):
            raise FloatingPointError("Non-finite global gradient norm.")
        optimizer.step()
        lr_scheduler.step()
        if (epoch + 1) % args.log_freq == 0:
            logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1),
                                   'lr: {}'.format(lr_scheduler.get_last_lr()[0]),
                                   format_metrics(train_metrics, 'train'),
                                   'time: {:.4f}s'.format(time.time() - t)
                                   ]))
        if (epoch + 1) % args.eval_freq == 0 or epoch + 1 == args.epochs:
            model.eval()
            with torch.no_grad():
                embeddings = model.encode(data['features'], data['adj_train_norm'])
                check_finite('evaluation embeddings', embeddings, fail_on_nonfinite)
                val_metrics = model.compute_metrics(embeddings, data, 'val')
                if not all(np.isfinite(value) for value in metric_values(val_metrics).values()):
                    raise FloatingPointError('Non-finite validation metrics.')
            if (epoch + 1) % args.log_freq == 0:
                logging.info(" ".join(['Epoch: {:04d}'.format(epoch + 1), format_metrics(val_metrics, 'val')]))
            improved = model.has_improved(best_val_metrics, val_metrics)
            if args.val_tie_break and not improved:
                improved = (validation_score(args, val_metrics) == validation_score(args, best_val_metrics)
                            and float(val_metrics['loss']) < float(best_val_metrics.get('loss', float('inf'))))
            history_entry = {
                'epoch': epoch + 1,
                'val': metric_values(val_metrics),
                'val_composite': 0.5 * (float(val_metrics['roc']) + float(val_metrics['ap']))
                if args.task == 'lp' else None,
                'improved': bool(improved),
            }
            if improved:
                best_state = {name: value.detach().cpu().clone()
                              for name, value in model.state_dict().items()}
                best_val_metrics = val_metrics
                best_epoch = epoch + 1
                best_phdm_diagnostics = collect_phdm_diagnostics(model)
                best_geometry_diagnostics = collect_geometry_diagnostics(model, embeddings)
                counter = 0
            else:
                counter += 1
                if counter == args.patience and epoch > args.min_epochs:
                    logging.info("Early stopping")
                    history.append(history_entry)
                    break
            history.append(history_entry)

    logging.info("Optimization Finished!")
    logging.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    if best_state is None:
        raise RuntimeError('No validation-selected model checkpoint was produced.')
    if not all(torch.isfinite(value).all() for value in best_state.values()):
        raise FloatingPointError('Non-finite validation-selected model state.')
    selection_policy = ('validation_roc_ap' if args.task == 'lp' else 'validation_f1')
    if args.val_tie_break:
        selection_policy += '_then_loss'
    checkpoint_path = None
    if args.checkpoint_out:
        checkpoint_path = Path(args.checkpoint_out).resolve()
    elif args.metrics_out:
        checkpoint_path = Path(args.metrics_out).resolve().with_suffix('.pt')
    elif args.save:
        checkpoint_path = Path(save_dir) / 'model.pth'
    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(model_state_dict=best_state, epoch=best_epoch, args=vars(args),
                        selection_policy=selection_policy, validation_history=history,
                        data_sha256=data_sha256, split_sha256=split_sha256), checkpoint_path)
        selected = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(selected['model_state_dict'])
    else:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        best_emb = model.encode(data['features'], data['adj_train_norm'])
        check_finite('selected checkpoint embeddings', best_emb, True)
        replayed_val = model.compute_metrics(best_emb, data, 'val')
        for name, expected in metric_values(best_val_metrics).items():
            tolerance = 1e-6 if name == 'loss' else 1e-10
            if abs(float(replayed_val[name]) - expected) > tolerance:
                raise RuntimeError('Validation metrics changed after checkpoint reload.')
        best_phdm_diagnostics = collect_phdm_diagnostics(model)
        best_geometry_diagnostics = collect_geometry_diagnostics(model, best_emb)
        if args.evaluate_test:
            best_test_metrics = model.compute_metrics(best_emb, data, 'test')
            if not all(np.isfinite(value) for value in metric_values(best_test_metrics).values()):
                raise FloatingPointError('Non-finite selected-checkpoint test metrics.')
    logging.info(" ".join(["Val set results:", format_metrics(best_val_metrics, 'val')]))
    if best_test_metrics is not None:
        logging.info(" ".join(["Test set results:", format_metrics(best_test_metrics, 'test')]))
    predictions_path = None
    if checkpoint_path is not None and getattr(model, 'test_predictions', None) is not None:
        predictions_path = checkpoint_path.with_suffix('.predictions.npz')
        np.savez(predictions_path, **model.test_predictions)
    elapsed = time.time() - t_total
    peak_cuda_mem_mb = 0.0
    if int(args.cuda) >= 0:
        peak_cuda_mem_mb = torch.cuda.max_memory_allocated(model.c.device) / (1024 ** 2)
    result = {
        'status': 'complete',
        'run_id': args.run_id,
        'args': vars(args),
        'total_params': tot_params,
        'epochs_ran': epochs_ran,
        'best_epoch': best_epoch,
        'best_val': metric_values(best_val_metrics),
        'best_val_composite': 0.5 * (float(best_val_metrics['roc']) + float(best_val_metrics['ap']))
        if args.task == 'lp' else None,
        'test_at_best_val': metric_values(best_test_metrics) if best_test_metrics is not None else None,
        'selection_policy': selection_policy,
        'test_evaluations': args.evaluate_test,
        'checkpoint': str(checkpoint_path) if checkpoint_path is not None else None,
        'checkpoint_sha256': hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() if checkpoint_path is not None else None,
        'data_sha256': data_sha256,
        'split_sha256': split_sha256,
        'split_sizes': split_sizes,
        'model_state_sha256': tensor_digest(best_state),
        'test_predictions': str(predictions_path) if predictions_path is not None else None,
        'test_predictions_sha256': hashlib.sha256(predictions_path.read_bytes()).hexdigest() if predictions_path is not None else None,
        'last_global_grad_norm': last_grad_norm,
        'phdm_diagnostics': best_phdm_diagnostics,
        'geometry_diagnostics': best_geometry_diagnostics,
        'elapsed_sec': elapsed,
        'peak_cuda_mem_mb': peak_cuda_mem_mb,
        'history': history,
    }
    if args.metrics_out:
        write_result(args.metrics_out, result)
    if args.save:
        np.save(os.path.join(save_dir, 'embeddings.npy'), best_emb.cpu().detach().numpy())
        if hasattr(model.encoder, 'att_adj'):
            filename = os.path.join(save_dir, args.dataset + '_att_adj.p')
            pickle.dump(model.encoder.att_adj.cpu().to_dense(), open(filename, 'wb'))
            print('Dumped attention adj: ' + filename)

        json.dump(vars(args), open(os.path.join(save_dir, 'config.json'), 'w'))
        if checkpoint_path != Path(save_dir) / 'model.pth':
            torch.save(model.state_dict(), os.path.join(save_dir, 'model.pth'))
        logging.info(f"Saved model in {save_dir}")
    return result

if __name__ == '__main__':
    args = parser.parse_args()
    train(args)
