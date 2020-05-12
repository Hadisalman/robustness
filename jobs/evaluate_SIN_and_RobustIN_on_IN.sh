#!/bin/bash
# Stylized IN eval on IN
python main_trainer.py --arch resnet50 --dataset imagenet --data ~/datasets/IMAGENET/imagenet/ --eval-only 1 --adv-eval 0 --batch-size 64 --out-dir ~/azure-madry/stylized_imagenet/evaluations/on_imagenet --exp-name SINresnet50_batch512_eps0_eps_test_0 --model-path ~/azure-madry/stylized_imagenet/resnet50_batch512_eps0/checkpoint.pt.latest &&

# Robust IN L2 eps 0
python main_trainer.py --arch resnet50 --dataset imagenet --data ~/datasets/IMAGENET/imagenet/ --eval-only 1 --adv-eval 0 --batch-size 64 --out-dir ~/azure-madry/stylized_imagenet/evaluations/on_imagenet --exp-name INresnet50_batch512_eps0_eps_test_0 --model-path ~/azure-madry/imagenet_experiments/l2/resnet50/resnet50_batch512_3steps_eps0/checkpoint.pt.latest &&

# Robust IN L2 eps 1
python main_trainer.py --arch resnet50 --dataset imagenet --data ~/datasets/IMAGENET/imagenet/ --eval-only 1 --adv-eval 0 --batch-size 64 --out-dir ~/azure-madry/stylized_imagenet/evaluations/on_imagenet --exp-name INresnet50_batch512_eps1_eps_test_0 --model-path ~/azure-madry/imagenet_experiments/l2/resnet50/resnet50_batch512_3steps_eps1/checkpoint.pt.latest &&

# Robust IN L2 eps 2
python main_trainer.py --arch resnet50 --dataset imagenet --data ~/datasets/IMAGENET/imagenet/ --eval-only 1 --adv-eval 0 --batch-size 64 --out-dir ~/azure-madry/stylized_imagenet/evaluations/on_imagenet --exp-name INresnet50_batch512_eps2_eps_test_0 --model-path ~/azure-madry/imagenet_experiments/l2/resnet50/resnet50_batch512_3steps_eps2/checkpoint.pt.latest &&

# Robust IN L2 eps 3
python main_trainer.py --arch resnet50 --dataset imagenet --data ~/datasets/IMAGENET/imagenet/ --eval-only 1 --adv-eval 0 --batch-size 64 --out-dir ~/azure-madry/stylized_imagenet/evaluations/on_imagenet --exp-name INresnet50_batch512_eps3_eps_test_0 --model-path ~/azure-madry/imagenet_experiments/l2/resnet50/resnet50_batch512_3steps_eps3/checkpoint.pt.latest &&

# Robust IN L2 eps 4
python main_trainer.py --arch resnet50 --dataset imagenet --data ~/datasets/IMAGENET/imagenet/ --eval-only 1 --adv-eval 0 --batch-size 64 --out-dir ~/azure-madry/stylized_imagenet/evaluations/on_imagenet --exp-name INresnet50_batch512_eps4_eps_test_0 --model-path ~/azure-madry/imagenet_experiments/l2/resnet50/resnet50_batch512_3steps_eps4/checkpoint.pt.latest &&

# Robust IN L2 eps 5
python main_trainer.py --arch resnet50 --dataset imagenet --data ~/datasets/IMAGENET/imagenet/ --eval-only 1 --adv-eval 0 --batch-size 64 --out-dir ~/azure-madry/stylized_imagenet/evaluations/on_imagenet --exp-name INresnet50_batch512_eps5_eps_test_0 --model-path ~/azure-madry/imagenet_experiments/l2/resnet50/resnet50_batch512_3steps_eps5/checkpoint.pt.latest &&

# Robust IN L2 eps 6
python main_trainer.py --arch resnet50 --dataset imagenet --data ~/datasets/IMAGENET/imagenet/ --eval-only 1 --adv-eval 0 --batch-size 64 --out-dir ~/azure-madry/stylized_imagenet/evaluations/on_imagenet --exp-name INresnet50_batch512_eps6_eps_test_0 --model-path ~/azure-madry/imagenet_experiments/l2/resnet50/resnet50_batch512_3steps_eps6/checkpoint.pt.latest &&

# Robust IN L2 eps 7
python main_trainer.py --arch resnet50 --dataset imagenet --data ~/datasets/IMAGENET/imagenet/ --eval-only 1 --adv-eval 0 --batch-size 64 --out-dir ~/azure-madry/stylized_imagenet/evaluations/on_imagenet --exp-name INresnet50_batch512_eps7_eps_test_0 --model-path ~/azure-madry/imagenet_experiments/l2/resnet50/resnet50_batch512_3steps_eps7/checkpoint.pt.latest &&

# Robust IN L2 eps 8
python main_trainer.py --arch resnet50 --dataset imagenet --data ~/datasets/IMAGENET/imagenet/ --eval-only 1 --adv-eval 0 --batch-size 64 --out-dir ~/azure-madry/stylized_imagenet/evaluations/on_imagenet --exp-name INresnet50_batch512_eps8_eps_test_0 --model-path ~/azure-madry/imagenet_experiments/l2/resnet50/resnet50_batch512_3steps_eps8/checkpoint.pt.latest &&

# Robust IN L2 eps 9
python main_trainer.py --arch resnet50 --dataset imagenet --data ~/datasets/IMAGENET/imagenet/ --eval-only 1 --adv-eval 0 --batch-size 64 --out-dir ~/azure-madry/stylized_imagenet/evaluations/on_imagenet --exp-name INresnet50_batch512_eps9_eps_test_0 --model-path ~/azure-madry/imagenet_experiments/l2/resnet50/resnet50_batch512_3steps_eps9/checkpoint.pt.latest &&

# Robust IN L2 eps 10
python main_trainer.py --arch resnet50 --dataset imagenet --data ~/datasets/IMAGENET/imagenet/ --eval-only 1 --adv-eval 0 --batch-size 64 --out-dir ~/azure-madry/stylized_imagenet/evaluations/on_imagenet --exp-name INresnet50_batch512_eps10_eps_test_0 --model-path ~/azure-madry/imagenet_experiments/l2/resnet50/resnet50_batch512_3steps_eps10/checkpoint.pt.latest &&

# Robust IN L2 eps 11
python main_trainer.py --arch resnet50 --dataset imagenet --data ~/datasets/IMAGENET/imagenet/ --eval-only 1 --adv-eval 0 --batch-size 64 --out-dir ~/azure-madry/stylized_imagenet/evaluations/on_imagenet --exp-name INresnet50_batch512_eps11_eps_test_0 --model-path ~/azure-madry/imagenet_experiments/l2/resnet50/resnet50_batch512_3steps_eps11/checkpoint.pt.latest &&

# Robust IN L2 eps 12
python main_trainer.py --arch resnet50 --dataset imagenet --data ~/datasets/IMAGENET/imagenet/ --eval-only 1 --adv-eval 0 --batch-size 64 --out-dir ~/azure-madry/stylized_imagenet/evaluations/on_imagenet --exp-name INresnet50_batch512_eps12_eps_test_0 --model-path ~/azure-madry/imagenet_experiments/l2/resnet50/resnet50_batch512_3steps_eps12/checkpoint.pt.latest
