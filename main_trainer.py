# Uncomment the following two lines if not running on azure
import sys
sys.path.append('robustness')

from robustness import model_utils, datasets, defaults, train
from robustness.tools.breeds_helpers import BreedsDatasetGenerator
from torchvision import models
from cox import utils
import cox.store
import torch as ch
from torch import nn
import argparse
import os 
import numpy as np

from custom_models.vision_transformer import *

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
vitmodeldict = {
    "vit_small_patch16_224": vit_small_patch16_224,
    "vit_base_patch16_224": vit_base_patch16_224,
    "vit_base_patch16_384": vit_base_patch16_384,
    "vit_base_patch32_384": vit_base_patch32_384,
    "vit_large_patch16_224": vit_large_patch16_224,
    "vit_large_patch16_384": vit_large_patch16_384,
    "vit_large_patch32_384": vit_large_patch32_384,
    "vit_huge_patch16_224": vit_huge_patch16_224,
    "vit_huge_patch32_384": vit_huge_patch32_384,
    'deit_tiny_patch16_224': deit_tiny_patch16_224,
    'deit_small_patch16_224': deit_small_patch16_224,
    'deit_base_patch16_224': deit_base_patch16_224,
    'deit_base_patch16_384': deit_base_patch16_384,
    ##CIFAR10
    'deit_tiny_patch4_32': deit_tiny_patch4_32,
    'deit_small_patch4_32': deit_small_patch4_32,
    'deit_base_patch4_32': deit_base_patch4_32,
}

def main(args, store):
    if args.dataset == 'cifar':
        ds = datasets.CIFAR('/tmp/')
    elif args.dataset in ['imagenet', 'stylized_imagenet']:
        ds = datasets.ImageNet(args.data)
        # Comment out if using a standard imagenet dataset
        ds.custom_class = 'Zipped'
    elif args.dataset in ['Mixed-13', 'Living-8', 'Living-11', 'Dogs-8', 'NonLiving-9']:
        # ds = get_breeds_dataset(args.dataset, args.data)
        INFO_DIR = os.path.join(args.data,'imagenet_info/modified')
        ds, _, _ = get_breeds_datasets(args.dataset, args.data, INFO_DIR)        # Comment out if using a standard imagenet dataset
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
        train_loader, val_loader = ds.make_loaders(only_val=args.eval_only, batch_size=args.batch_size, 
                                            val_batch_size=args.batch_size//2, workers=args.workers, subset=args.subset)

    # An option to resume finetuning from a checkpoint. Only for Imagenet-Imagenet transfer
    model_path = os.path.join(args.out_dir, args.exp_name, 'checkpoint.pt.latest')
    CONTINUE_FROM_CHECKPOINT = False
    if args.resume and os.path.isfile(model_path):
        print('[Resuming finetuning from a checkpoint...]')
        CONTINUE_FROM_CHECKPOINT = True
    else: 
        model_path = args.model_path

    add_custom_forward = True
    if args.arch in pytorch_models.keys():
        arch = pytorch_models[args.arch]()
    elif args.arch in vitmodeldict:
        arch = vitmodeldict[args.arch](num_classes=ds.num_classes,
                                        drop_rate=0.,
                                        drop_path_rate=0.1,
                                        norm_embed=True)
    else:
        arch = args.arch
        add_custom_forward = False

    model, checkpoint = \
        model_utils.make_and_restore_model(arch=arch, dataset=ds, 
                        resume_path=model_path, add_custom_forward=add_custom_forward)
    
    # don't pass checkpoint to train_model do avoid resuming for epoch, optimizers etc.
    if not CONTINUE_FROM_CHECKPOINT:
        checkpoint = None


    if 'module' in dir(model): model = model.module

    if args.eval_only:
        return train.eval_model(args, model, val_loader, store=store)

    print(f"Dataset: {args.dataset} | Model: {args.arch}")
    train.train_model(args, model, (train_loader, val_loader), store=store, checkpoint=checkpoint)

def get_breeds_dataset(ds_name, ds_path):
    # INFO_DIR = "/data/theory/robustopt/datasets/imagenet_info/modified"
    # INFO_DIR = "/home/hasalman/datasets/IMAGENET/imagenet/imagenet_info/modified"
    INFO_DIR = os.path.join(ds_path,'imagenet_info/modified')
    DG = BreedsDatasetGenerator(INFO_DIR)
    if ds_name == "Living-11":
        # Living things 11 superclasses, 5 subclasses per
        subclass_ranges, label_map, subclass_tuple, _, _ = DG.get_superclasses(level=5, 
                                                                        ancestor="n00004258",
                                                                        Nsubclasses=5, 
                                                                        split=None, 
                                                                        balanced=True, 
                                                                        random_seed=2,
                                                                        verbose=False)
    elif ds_name == "Dogs-8":
        # Dogs, 8 superclasses, 2 subclasses per
        subclass_ranges, label_map, subclass_tuple, _, _ = DG.get_superclasses(level=6, 
                                                                        ancestor="n02084071",
                                                                        Nsubclasses=2, 
                                                                        split=None, 
                                                                        balanced=True, 
                                                                        random_seed=2,
                                                                        verbose=False)
    elif ds_name == "NonLiving-9":
        # Non-living things 11 superclasses, 5 subclasses per
        subclass_ranges, label_map, subclass_tuple, superclasses, _ = DG.get_superclasses(level=4, 
                                                                        ancestor="n00021939",
                                                                        Nsubclasses=12, 
                                                                        split=None, 
                                                                        balanced=True, 
                                                                        random_seed=2,
                                                                        verbose=False)
        skip = [3, 4, 5, 6, 10, 11]
        subclass_ranges = [s for si, s in enumerate(subclass_ranges) if si not in skip]

        label_map_orig = {k: v for k, v in label_map.items()}
        label_map = {}
        count = 0
        for i in range(len(label_map_orig)):
            if i not in skip:
                label_map[count] = label_map_orig[i]
                count += 1

        assert len(subclass_ranges) == len(label_map)
    else:
        raise Exception('Unkown dataset.')

    INFO_DIR = os.path.join(ds_path,'imagenet_info/modified')
    dataset = datasets.CustomImageNet(ds_path, subclass_ranges)
    return dataset

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
