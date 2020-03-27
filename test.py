from robustness import datasets
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy

from IPython import embed

# in_path = '/home/hasalman/datasets/IMAGENET/imagenet'
in_path = '/home/hasalman/datasets/imagenet_zipped'
in_info_path = '/home/hasalman/datasets/imagenet_info'

in_hier = ImageNetHierarchy(in_path, in_info_path)
superclass_wnid = common_superclass_wnid('living_9')
class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)

custom_dataset = datasets.CustomImageNet(in_path, class_ranges)
custom_dataset.custom_class = 'Zipped'
train_loader, test_loader = custom_dataset.make_loaders(workers=20,
                                                        batch_size=256)
embed()
