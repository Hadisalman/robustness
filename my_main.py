from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR, ImageNetZipped
import torch as ch
import torchvision
# We use cox (http://github.com/MadryLab/cox) to log, store and analyze
# results. Read more at https//cox.readthedocs.io.
from cox.utils import Parameters
import cox.store

import argparse
import os
# from IPython import embed

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', type=str, default='resnet18')
parser.add_argument('--dataset', type=str, default='cifar')
parser.add_argument('--data-path', type=str, default='/tmp/')
parser.add_argument('--outdir', type=str, default='./outdir')
parser.add_argument('--exp-id', type=str, default=None)
parser.add_argument('--mp', action='store_true', help='Flag for mixed precision')
parser.add_argument('--batch-size', type=int, default=256)
parser.add_argument('--AT', action='store_true', help='Adversarially train')
parser.add_argument('--resume', action='store_true', help='Whether to resume or not')
parser.add_argument('--num-steps', type=int, default=3)

args = parser.parse_args()

assert args.exp_id != None

model_path = os.path.join(args.outdir, args.exp_id, 'checkpoint.pt.latest')
if args.resume and os.path.isfile(model_path):
    args.resume = model_path

# Hard-coded dataset, architecture, batch size, workers
if args.dataset == 'cifar':
    ds = CIFAR('/tmp/')
elif args.dataset == 'imagenet':
    ds = ImageNetZipped(args.data_path)
else:
    raise Exception("Unknown dataset")

if args.arch in ['resnet18', 'resnet50', 'wide_resnet50', 'wide_resnet50_2']:
    model, checkpoint = model_utils.make_and_restore_model(arch=args.arch, dataset=ds, resume_path=args.resume)
elif args.arch == 'WRN':
    model = torchvision.models.wide_resnet50_2(pretrained=False)
    model, checkpoint = model_utils.make_and_restore_model(arch=model, dataset=ds, resume_path=args.resume)
else:
    raise Exception('Unknown architecture')

if 'module' in dir(model): model = model.module

train_loader, val_loader = ds.make_loaders(batch_size=args.batch_size, workers=16)

# Create a cox store for logging
out_store = cox.store.Store(args.outdir, args.exp_id)

# Hard-coded base parameters
train_args = Parameters({
    'adv_train': args.AT,
    'constraint': '2',
    'eps': 3.0,
    'attack_lr': 2.0,
    'attack_steps': args.num_steps,
    'out_dir': args.outdir,
    'custom_lr_multiplier': 'cyclic',
    'lr_interpolation': 'step',
    'lr': 1.2,
    'epochs': 50,
    'random_start': True,
    'mixed_precision': args.mp,
    'log_iters': 1,
})

# Fill whatever parameters are missing from the defaults
train_args = defaults.check_and_fill_args(train_args,
                        defaults.TRAINING_ARGS, CIFAR)
train_args = defaults.check_and_fill_args(train_args,
                        defaults.PGD_ARGS, CIFAR)

# Train a model
import time
start = time.time()
train.train_model(train_args, model, (train_loader, val_loader), store=out_store, checkpoint=checkpoint)
print('')
print('')
print('')
print('Total execution time:', time.time() - start)


