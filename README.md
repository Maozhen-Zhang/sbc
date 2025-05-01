# sbc

# The visual pormpt,

1. At first train the shadow visual prompt,
```shell
python -m src.main_scratch --name exp_normal -data CIFAR10 --pois_ratio --method padding  0 --checkpoints ckpt
```


2. Then, generate the trigger and visual prompt,
```shell
python -m src.main --name exp_generate_trigger -data CIFAR10 --checkpoints ckpt --batch_size 128 --eps '4/255' --method badnoise \
--resume_c ./ckpt/exp_normal-model_ViT-B_32-datasets_CIFAR10-seed_0-pois_ratio_0.0/checkpoint.pth.tar \
```

3. train by victim


# Eval asr, vp acc, ori acc
```shell
python -m src.main --name eval_ -data CIFAR10 --batch_size 128 --evaluate \
--reume ./ckpt/exp_normal-model_ViT-B_32-datasets_CIFAR10-seed_0-pois_ratio_0.0/checkpoint.pth.tar
```

# eval ft
```shell
python -m src.ft_eval --name eval_ft -data CIFAR10 --batch_size 128 --evaluate --method badnoise \
--resume_c /home/zmz/code/vp/results/exp_scratch_train_newclean-model_ViT-B_32-datasets_CIFAR10-seed_0-pois_ratio_0.0/checkpoint.pth.tar \
--resume /home/zmz/code/vp/res/exp_train-model_ViT-B_32-datasets_CIFAR10-seed_0-pois_ratio_0.1-eps_4_255-all2one/checkpoint.pth.tar \
--clip_model_name checkpoint_epoch_1_step_0.pth --backdoor

```