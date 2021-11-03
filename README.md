# VS loss

Implementation of VS loss for deep-net experiments on imbalanced classification reported in the NeurIPS 2021 paper:

> Ganesh Ramachandra Kini, Orestis Paraskevas, Samet Oymak, Christos Thrampoulidis
>
> [Label-Imbalanced and Group-Sensitive Classification under Overparameterization](https://arxiv.org/abs/2103.01550)

## Abstract

The goal in label-imbalanced and group-sensitive classification is to optimize relevant metrics such as balanced
error and equal opportunity. Classical methods, such as weighted cross-entropy, fail when training deep nets to the
terminal phase of training (TPT), that is training beyond zero training error. This observation has motivated recent
flurry of activity in developing heuristic alternatives following the intuitive mechanism of promoting larger margin
for minorities. In contrast to previous heuristics, we follow a principled analysis explaining how different loss
adjustments affect margins. First, we prove that for all linear classifiers trained in TPT, it is necessary to
introduce multiplicative, rather than additive, logit adjustments so that the interclass margins change appropriately.
To show this, we discover a connection of the multiplicative CE modification to the cost-sensitive support-vector
machines. Perhaps counterintuitively, we also find that, at the start of training, the same multiplicative weights can
actually harm the minority classes. Thus, while additive adjustments are ineffective in the TPT, we show that they can
speed up convergence by countering the initial negative effect of the multiplicative weights. Motivated by these
findings, we formulate the vector-scaling (VS) loss, that captures existing techniques as special cases. Moreover,
we introduce a natural extension of the VS-loss to group-sensitive classification, thus treating the two common types
of imbalances (label/group) in a unifying way. Importantly, our experiments on state-of-the-art datasets are fully
consistent with our theoretical insights and confirm the superior performance of our algorithms. Finally, for
imbalanced Gaussian-mixtures data, we perform a generalization analysis, revealing tradeoffs between balanced
/ standard error and equal opportunity.

## Dependencies

- python 3.8.10
- matplotlib 3.0.3
- numpy 1.17.4
- pandas 0.25.3
- Pillow 8.4.0
- pytorch_transformers 1.2.0
- scikit-learn 1.0.1
- torch 1.10.0
- torchvision 0.10.0
- TensorboardX 2.1
- tqdm 4.62.3

## Class Imbalance 

We built on the implementation of [LDAM-DRW](https://github.com/kaidic/LDAM-DRW/), to which we add the implementation of
VS loss. LDAM loss was featured on the paper:

> Kaidi Cao, Colin Wei, Adrien Gaidon, Nikos Arechiga, Tengyu Ma
>
> [Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss](https://arxiv.org/abs/1906.07413)

### Datasets

* This repository contains our experiments on the **CIFAR10** and **CIFAR100** datasets. You can load these datasets
with the STEP and LT imbalance profiles and with different imbalance ratios using `generate_cifar.py`.
* For the experiments on synthetic data, MNIST and ImageNet please contact the authors.

### Sample commands

To train a network on CIFAR with one of the available losses, you will need to run the `train.py` file from the
`class_imbalance` directory. Find some example commands below:

* To train on CIFAR100 with the STEP-10 imbalance profile using the wCE loss run:

```bash
python class_imbalance/train.py --dataset cifar100 --loss_type CE --imb_type step --imb_factor 0.1 --train_rule Reweight
```

* To train on CIFAR10 with the LT-100 profile using the VS loss with parameters gamma=0.2 and tau=1.2 run:

```bash
python class_imbalance/train.py --dataset cifar10 --loss_type VS --imb_type exp --imb_factor 0.01 --gamma 0.2 --tau 1.2
```
The LA and CDT losses are also implemented and can be used as options. All losses can be used with standard ERM,
resampling or the DRW training rule.

## Group Imbalance

We built on the implementation of [group DRO](https://github.com/kohpangwei/group_DRO), to which we add the
implementation of group-VS loss. Group DRO was featured on the paper:

> Shiori Sagawa*, Pang Wei Koh*, Tatsunori Hashimoto, and Percy Liang
>
> [Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization](https://arxiv.org/abs/1911.08731)

### Dataset

For the group-sensitive classification experiments we use the **Waterbirds** dataset, which consists of 2 classes
(waterbirds and landbirds) and 4 groups (waterbirds and landbirds on land and water backgrounds). For more details about
the waterbirds dataset please refer to the [group DRO](https://github.com/kohpangwei/group_DRO) repository and paper.

Before running any group-sensitive experiments the dataset must be downloaded and placed
in the `group_imbalance/data` folder. You can do this by running the following commands:
```bash
wget -P group_imbalance/data https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz
tar -xf group_imbalance/data/waterbird_complete95_forest2water2.tar.gz -C group_imbalance/data
```
### Sample commands 

To perform group-sensitive classification on the Waterbirds dataset with VS loss, you will need to run the `run_expt.py`
file from the `group_imbalance` directory. You can also use DRO in conjunction with any of the losses.

* To train on Waterbirds with VS-loss and DRO:

```bash
python group_imbalance/run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.001 --batch_size 64 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustment 0 --loss vs --vs_alpha 0.3 --dont_set_seed 1 --robust
```

(Note: the hyperparameter vs_alpha is same as the hyperparameter gamma=0.3 used in the paper)

## Reference

If you find our paper or this repository helpful for your research, please consider citing it as:

```
@article{kini2021label,
  title={Label-Imbalanced and Group-Sensitive Classification under Overparameterization},
  author={Kini, Ganesh Ramachandra and Paraskevas, Orestis and Oymak, Samet and Thrampoulidis, Christos},
  journal={arXiv preprint arXiv:2103.01550},
  year={2021}
}
```


