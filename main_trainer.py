# Uncomment the following two lines if not running on azure
import sys
sys.path.append('robustness')

from robustness import model_utils, datasets, defaults, train
from torchvision import models
from cox import utils
import cox.store
import torch as ch
from torch import nn
import argparse
import os 
import numpy as np
import helper_split


parser = argparse.ArgumentParser(description='PyTorch finetuning for transfer learning', 
                                conflict_handler='resolve')
parser = defaults.add_args_to_parser(defaults.CONFIG_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.MODEL_LOADER_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.TRAINING_ARGS, parser)
parser = defaults.add_args_to_parser(defaults.PGD_ARGS, parser)

# Custom arguments
parser.add_argument('--dataset', type=str, default='cifar', help='Dataset (Overrides the one in robustness.defaults)')
parser.add_argument('--resume', action='store_true', help='Whether to resume or not (Overrides the one in robustness.defaults)')
parser.add_argument('--model-path', type=str, default=None, help='Path to a checkpoint to load (useful for evaluation). Ignored if `resume` is True')
parser.add_argument('--subset', type=int, default=None, help='number of training data to use from the dataset')
parser.add_argument('--frac-rand-labels', type=float, default=None, 
            help='Fraction of the training set which is random labelled (fixed during training)')
parser.add_argument('--no-tqdm', type=int, default=1, choices=[0, 1], help='Do not use tqdm.')

pytorch_models = {
    'alexnet': models.alexnet,
    'vgg16': models.vgg16,
    'vgg16_bn': models.vgg16_bn,
    'squeezenet': models.squeezenet1_0,
    'densenet': models.densenet161,
    # 'inception': models.inception_v3,
    # 'googlenet': models.googlenet,
    'shufflenet': models.shufflenet_v2_x1_0,
    'mobilenet': models.mobilenet_v2,
    'resnext50_32x4d': models.resnext50_32x4d,
    'mnasnet': models.mnasnet1_0,
}

def main(args, store):
    if args.dataset == 'cifar':
        ds = datasets.CIFAR('/tmp/')
    elif args.dataset in ['imagenet', 'stylized_imagenet']:
        ds = datasets.ImageNet(args.data)
        ds.custom_class = 'Zipped'

    if args.frac_rand_labels and not args.eval_only:
        def make_rand_labels(ims, targs):
            new_targs = (targs + ch.randint(low=1, high=10, size=targs.shape).long()) % 10
            return ims, new_targs

        train_loader, val_loader = ds.make_loaders(batch_size=args.batch_size, workers=args.workers, data_aug=False)
        train_loader = TransformedLoader(train_loader,
                                        make_rand_labels,
                                        ds.transform_train,
                                        workers=train_loader.num_workers,
                                        batch_size=train_loader.batch_size,
                                        do_tqdm=True,
                                        fraction=args.frac_rand_labels)

    else:
        train_loader, val_loader = ds.make_loaders(only_val=args.eval_only, batch_size=args.batch_size, workers=args.workers, subset=args.subset)

    # An option to resume finetuning from a checkpoint. Only for Imagenet-Imagenet transfer
    model_path = os.path.join(args.out_dir, args.exp_name, 'checkpoint.pt.latest')
    CONTINUE_FROM_CHECKPOINT = False
    if args.resume and os.path.isfile(model_path):
        print('[Resuming finetuning from a checkpoint...]')
        CONTINUE_FROM_CHECKPOINT = True
    else: 
        model_path = args.model_path

    model, checkpoint = \
        model_utils.make_and_restore_model(arch=pytorch_models[args.arch]() if args.arch in pytorch_models.keys() else args.arch, 
                                        dataset=ds, resume_path=model_path, add_custom_forward=args.arch in pytorch_models.keys())
    
    # don't pass checkpoint to train_model do avoid resuming for epoch, optimizers etc.
    if not CONTINUE_FROM_CHECKPOINT:
        checkpoint = None


    if 'module' in dir(model): model = model.module

    if args.eval_only:
        return train.eval_model(args, model, val_loader, store=store)

    print(f"Dataset: {args.dataset} | Model: {args.arch}")
    train.train_model(args, model, (train_loader, val_loader), store=store, checkpoint=checkpoint)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.adv_train and eval(args.eps) == 0:
        print('[Switching to standard training since eps = 0]')
        args.adv_train = 0


    # Important for automatic job retries on the cluster in case of premptions. Avoid uuids.
    assert args.exp_name != None

    # Preprocess args
    args = defaults.check_and_fill_args(args, defaults.CONFIG_ARGS, None)
    if not args.eval_only:
        args = defaults.check_and_fill_args(args, defaults.TRAINING_ARGS, None)
    if args.adv_train or args.adv_eval:
        args = defaults.check_and_fill_args(args, defaults.PGD_ARGS, None)
    args = defaults.check_and_fill_args(args, defaults.MODEL_LOADER_ARGS, None)

    # Create store and log the args
    # store_file = os.path.join(args.out_dir, args.exp_name, 'store.h5')
    # if os.path.isfile(store_file):
    #     os.remove(store_file)
    store = cox.store.Store(args.out_dir, args.exp_name)
    if 'metadata' not in store.keys:
        args_dict = args.__dict__
        schema = cox.store.schema_from_dict(args_dict)
        store.add_table('metadata', schema)
        store['metadata'].append_row(args_dict)
    else:
        print('[Found existing metadata in store. Skipping this part.]')
    main(args, store)
