from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR, ImageNetZipped
import torch as ch

# We use cox (http://github.com/MadryLab/cox) to log, store and analyze
# results. Read more at https//cox.readthedocs.io.
from cox.utils import Parameters
import cox.store

import argparse

from IPython import embed

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--dataset', type=str, default='cifar')
parser.add_argument('--data-path', type=str, default='/tmp/')
parser.add_argument('--outdir', type=str, default='./outdir')
args = parser.parse_args()

# Hard-coded dataset, architecture, batch size, workers
if args.dataset == 'cifar':
    ds = CIFAR('/tmp/')
elif args.dataset == 'imagenet':
    ds = ImageNetZipped(args.data_path)
else:
    raise Exception("Unknown dataset")

m, _ = model_utils.make_and_restore_model(arch='resnet18', dataset=ds)

train_loader, val_loader = ds.make_loaders(batch_size=256, workers=16)

# Create a cox store for logging
out_store = cox.store.Store(args.outdir)

# Hard-coded base parameters
train_kwargs = {
    'out_dir': "train_out",
    'adv_train': 0,
    'constraint': '2',
    'eps': 0.5,
    'attack_lr': 1.5,
    'attack_steps': 20,
    'log_iters': 1,
    'mixed_precision': False,
}

# embed()
# train_kwargs = utils.Parameters({
#     'adv_train': False,
#     'constraint': '2',
#     'eps': 3.0,
#     'attack_lr': 2.0,
#     'attack_steps': 3,
#     'out_dir': args.outdir,
#     'custom_lr_multiplier': 'cyclic',
#     'lr_interpolation': 'step',
#     'lr': 1.2,
#     'epochs': 50,
#     'random_start': True,
#     'mixed_precision': True
# })
train_args = Parameters(train_kwargs)

# Fill whatever parameters are missing from the defaults
train_args = defaults.check_and_fill_args(train_args,
                        defaults.TRAINING_ARGS, CIFAR)
train_args = defaults.check_and_fill_args(train_args,
                        defaults.PGD_ARGS, CIFAR)

# Train a model
train.train_model(train_args, m, (train_loader, val_loader), store=out_store)