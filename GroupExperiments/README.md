Waterbirds experiments with Group-VS loss

We make use of the code from the authors of the following paper:

> Shiori Sagawa\*, Pang Wei Koh\*, Tatsunori Hashimoto, and Percy Liang
>
> [Distributionally Robust Neural Networks for Group Shifts: On the Importance of Regularization for Worst-Case Generalization](https://arxiv.org/abs/1911.08731)

We add to the above an implementation of the Group VS-loss function.
Please read "Sagawa_et_al_README.md" for prerequisites for the Waterbirds dataset. Below is an example command to run VS-loss with DRO:

python run_expt.py -s confounder -d CUB -t waterbird_complete95 -c forest2water2 --lr 0.001 --batch_size 64 --weight_decay 0.0001 --model resnet50 --n_epochs 300  --gamma 0.1 --generalization_adjustm    ent 0 --root_dir . --loss vs --vs_alpha 0.3 --dont_set_seed 1 --robust

(Note that the hyperparameter vs_alpha is same as the hyperparameter gamma=0.3 used in the paper)


