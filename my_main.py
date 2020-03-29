from robustness import model_utils, datasets, train, defaults
from robustness.datasets import CIFAR, ImageNetZipped, ImageNet
from robustness.loaders import TransformedLoader
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
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--AT', action='store_true', help='Adversarially train')
parser.add_argument('--resume', action='store_true', help='Whether to resume or not')
parser.add_argument('--num-steps', type=int, default=3)
parser.add_argument('--eps', type=float, default=3.0)
parser.add_argument('--attack-lr', type=float, default=2.0)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--epochs', type=int, default=90)
parser.add_argument('--step-lr', type=int, default=30)
parser.add_argument('--custom-lr-multiplier', type=str, default=None, help='Custon lr multiplier')
parser.add_argument('--frac-rand-labels', type=float, default=None, 
            help='Fraction of the training set which is random labelled (fixed during training)')
parser.add_argument('--subset', type=int, default=None, 
                    help='number of training data to use from the dataset')

args = parser.parse_args()

assert args.exp_id != None

if args.eps == 0:
    args.AT = False

model_path = os.path.join(args.outdir, args.exp_id, 'checkpoint.pt.latest')
if args.resume and os.path.isfile(model_path):
    args.resume = model_path
else:
    args.resume = None
    
# Hard-coded dataset, architecture, batch size, workers
if args.dataset == 'cifar':
    ds = CIFAR('/tmp/')
elif args.dataset == 'imagenet_local':
    ds = ImageNet(args.data_path)
elif args.dataset == 'imagenet':
    ds = ImageNetZipped(args.data_path)
else:
    raise Exception("Unknown dataset")

model, checkpoint = model_utils.make_and_restore_model(arch=args.arch, dataset=ds, resume_path=args.resume)

if 'module' in dir(model): model = model.module

if args.frac_rand_labels:
    def make_rand_labels(ims, targs):
        new_targs = (targs + ch.randint(low=1, high=10, size=targs.shape).long()) % 10
        return ims, new_targs

    train_loader, val_loader = ds.make_loaders(batch_size=args.batch_size, workers=16, data_aug=False)
    train_loader = TransformedLoader(train_loader,
                                    make_rand_labels,
                                    ds.transform_train,
                                    workers=train_loader.num_workers,
                                    batch_size=train_loader.batch_size,
                                    do_tqdm=True,
                                    fraction=args.frac_rand_labels)

else:
    train_loader, val_loader = ds.make_loaders(batch_size=args.batch_size, workers=16, subset=args.subset)

# Create a cox store for logging

# TODO: Check if store.h5 exists, and delete it before creating a new one.
#       This is a hack to avoid "End of HDF5 error back trace" on philly
store_file = os.path.join(args.outdir, args.exp_id, 'store.h5')
if os.path.isfile(store_file):
    os.remove(store_file)
# embed()
out_store = cox.store.Store(args.outdir, args.exp_id)
args_dict = args.__dict__
schema = cox.store.schema_from_dict(args_dict)
out_store.add_table('metadata', schema)
out_store['metadata'].append_row(args_dict)

# Hard-coded base parameters
train_args = Parameters({
    'adv_train': args.AT,
    'constraint': '2',
    'eps': args.eps,
    'attack_lr': args.attack_lr,
    'attack_steps': args.num_steps,
    'out_dir': args.outdir,
    'custom_lr_multiplier': args.custom_lr_multiplier,
    # 'lr_interpolation': 'step',
    'step_lr': args.step_lr,
    'lr': args.lr,
    'epochs': args.epochs,
    'batch_size': args.batch_size,
    'weight_decay': args.weight_decay,
    'random_start': False,
    'mixed_precision': args.mp,
    'log_iters': 1,
})

# Fill whatever parameters are missing from the defaults
train_args = defaults.check_and_fill_args(train_args,
                        defaults.TRAINING_ARGS, ImageNet)
train_args = defaults.check_and_fill_args(train_args,
                        defaults.PGD_ARGS, ImageNet)

# Train a model
import time
start = time.time()
train.train_model(train_args, model, (train_loader, val_loader), store=out_store, checkpoint=checkpoint)
print('')
print('')
print('')
print('Total execution time:', time.time() - start)


