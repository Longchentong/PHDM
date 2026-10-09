import os
import random
import sys
from pathlib import Path

import configargparse
import numpy as np
import torch
from torch.nn import DataParallel

CODE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(CODE_DIR))

from utils.initialize import load_model_checkpoint, select_dataset, select_model


def get_arguments():
    parser = configargparse.ArgumentParser(description='Evaluate a trained PHDM image classifier.')
    parser.add_argument('-c', '--config_file', is_config_file=True)
    parser.add_argument('--mode', choices=['test_accuracy'], default='test_accuracy')
    parser.add_argument('--device', default='cuda:0', type=lambda value: value.replace(' ', '').split(','))
    parser.add_argument('--dtype', choices=['float32', 'float64'], default='float32')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--load_checkpoint')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=128)
    parser.add_argument('--num_layers', type=int, choices=[18], default=18)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--encoder_manifold', choices=['euclidean'], default='euclidean')
    parser.add_argument('--decoder_manifold', choices=['lorentz', 'poincare'], default='lorentz')
    parser.add_argument('--learn_k', action='store_true')
    parser.add_argument('--encoder_k', type=float, default=1.)
    parser.add_argument('--decoder_k', type=float, default=1.)
    parser.add_argument('--clip_features', type=float, default=2.)
    parser.add_argument('--dataset', choices=['CIFAR-10', 'CIFAR-100', 'Tiny-ImageNet'], default='CIFAR-100')
    args, _ = parser.parse_known_args()
    return args


def main():
    args = get_arguments()
    if not args.load_checkpoint:
        raise SystemExit('Pass --load_checkpoint with the trained PHDM checkpoint.')
    args.load_checkpoint = str(Path(args.load_checkpoint).resolve())
    os.chdir(CODE_DIR)
    from train import evaluate
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_default_dtype(getattr(torch, args.dtype))
    device = args.device[0]
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(args.seed)
    _, _, test_loader, img_dim, num_classes = select_dataset(args, validation_split=False)
    model = select_model(img_dim, num_classes, args).to(device)
    model = load_model_checkpoint(model, args.load_checkpoint)
    model = DataParallel(model, device_ids=args.device)
    model.eval()
    loss, accuracy, accuracy5 = evaluate(model, test_loader, torch.nn.CrossEntropyLoss(), device)
    print(f'Loss={loss:.4f}, Acc@1={accuracy:.4f}, Acc@5={accuracy5:.4f}')


if __name__ == '__main__':
    main()
