## CrossMAE: Rethinking Patch Dependence for Masked Autoencoders
by <a href="https://max-fu.github.io">Letian Fu*</a>, <a href="https://tonylian.com">Long Lian*</a>, <a href="https://renwang435.github.io">Renhao Wang</a>, <a href="https://bfshi.github.io">Baifeng Shi</a>, <a href="https://people.eecs.berkeley.edu/~xdwang">Xudong Wang</a>, <a href="https://www.adamyala.org">Adam Yala†</a>, <a href="https://people.eecs.berkeley.edu/~trevor">Trevor Darrell†</a>, <a href="https://people.eecs.berkeley.edu/~efros">Alexei A. Efros†</a>, <a href="https://goldberg.berkeley.edu">Ken Goldberg†</a> at UC Berkeley and UCSF

[[Paper](https://arxiv.org/abs/2401.14391)] | [[Project Page](https://crossmae.github.io/)] | [[Citation](#citation)]


<p align="center">
  <img src="https://crossmae.github.io/crossmae2.jpg" width="800">
</p>

This is a PyTorch implementation of the CrossMAE paper [Rethinking Patch Dependence for Masked Autoencoders](https://crossmae.github.io/). The code is based on the original [MAE](https://github.com/facebookresearch/mae) repo. The codebase supports CrossMAE and MAE, with `timm==0.9.7`, `torch==2.0.0`, and flash-attn 2.

## Models
The encoder part of CrossMAE matches exactly with MAE. Therefore, we use the same code for fine-tuning. We also encourage you to try CrossMAE checkpoints in your downstream applications. These models are trained on ImageNet-1k for 800 epochs (except that 448 models are trained for 400 epochs), with masking ratio and kept mask ratio both set to 0.75.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Small</th>
<th valign="bottom">ViT-Base</th>
<th valign="bottom">ViT-Base<sub>448</sub></th>
<th valign="bottom">ViT-Large</th>
<!-- TABLE BODY -->
<tr><td align="left">pretrained checkpoint</td>
<td align="center"><a href='https://huggingface.co/longlian/CrossMAE/resolve/main/vits-mr0.75-kmr0.75-dd12/imagenet-mae-cross-vits-pretrain-wfm-mr0.75-kmr0.75-dd12-ep800-ui.pth?download=true'>download</a></td>
<td align="center"><a href='https://huggingface.co/longlian/CrossMAE/resolve/main/vitb-mr0.75-kmr0.75-dd12/imagenet-mae-cross-vitb-pretrain-wfm-mr0.75-kmr0.75-dd12-ep800-ui.pth?download=true'>download</a></td>
<td align="center"><a href='https://huggingface.co/longlian/CrossMAE/resolve/main/vitb-mr0.75-kmr0.75-dd12-448-400/imagenet-mae-cross-vitb-pretrain-wfm-mr0.75-kmr0.25-dd12-ep400-ui-res-448.pth?download=true'>download</a></td>
<td align="center"><a href='https://huggingface.co/longlian/CrossMAE/resolve/main/vitl-mr0.75-kmr0.75-dd12/imagenet-mae-cross-vitl-pretrain-wfm-mr0.75-kmr0.75-dd12-ep800-ui.pth?download=true'>download</a></td>
</tr>
<tr><td align="left">fine-tuned checkpoint</td>
<td align="center"><a href='https://huggingface.co/longlian/CrossMAE/resolve/main/vits-mr0.75-kmr0.75-dd12/imagenet-mae-cross-vits-finetune-wfm-mr0.75-kmr0.75-dd12-ep800-ui.pth?download=true'>download</a></td>
<td align="center"><a href='https://huggingface.co/longlian/CrossMAE/resolve/main/vitb-mr0.75-kmr0.75-dd12/imagenet-mae-cross-vitb-finetune-wfm-mr0.75-kmr0.75-dd12-ep800-ui.pth?download=true'>download</a></td>
<td align="center"><a href='https://huggingface.co/longlian/CrossMAE/resolve/main/vitb-mr0.75-kmr0.75-dd12-448-400/imagenet-mae-cross-vitb-finetune-wfm-mr0.75-kmr0.25-dd12-ep400-ui-res-448.pth?download=true'>download</a></td>
<td align="center"><a href='https://huggingface.co/longlian/CrossMAE/resolve/main/vitl-mr0.75-kmr0.75-dd12/imagenet-mae-cross-vitl-finetune-wfm-mr0.75-kmr0.75-dd12-ep800-ui.pth?download=true'>download</a></td>
</tr>
<tr><td align="left">reference ImageNet accuracy</td>
<td align="center">79.318</td>
<td align="center">83.722</td>
<td align="center">84.598</td>
<td align="center">85.432</td>
</tr>
</tbody></table>

## Train CrossMAE on **one single RTX 4090**
With the efficiency of CrossMAE, it's possible to train CrossMAE on **one single RTX 4090** on a personal computer. The CPU is i9-14900k, with 96GB RAM.

<details>
  <summary>Instructions and trained models</summary>

The training and fine-tuning command (with `${IMAGENET_DIR}` the directory for imagenet, ViT-S as an example):
```sh
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --master_port 2780 main_pretrain.py --batch_size 512 --accum_iter 8 --model mae_vit_small_patch16 --norm_pix_loss --blr 1.5e-4 --weight_decay 0.05 --data_path ${IMAGENET_DIR} --num_workers 16 --multi_epochs_dataloader --output_dir output/imagenet-crossmae-vits-pretrain-wfm-mr0.75-kmr0.25-dd12-ep800 --cross_mae --weight_fm --decoder_depth 12 --mask_ratio 0.75 --kept_mask_ratio 0.75 --epochs 800 --warmup_epochs 40 --use_input

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 torchrun --nproc_per_node=1 --master_port 2860 main_finetune.py --batch_size 512 --accum_iter 2 --model vit_small_patch16 --finetune output/imagenet-crossmae-vits-pretrain-wfm-mr0.75-kmr0.25-dd12-ep800/checkpoint.pth --epoch 100 --blr 5e-4 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --data_path ${IMAGENET_DIR} --num_workers 12 --output_dir output/imagenet-crossmae-vits-finetune-wfm-mr0.75-kmr0.25-dd12-ep800 --multi_epochs_dataloader
# Reference results:
# * Acc@1 79.462 Acc@5 94.864 loss 0.907
```

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>

<!-- TABLE BODY -->
<tr><td align="left">pretrained checkpoint</td><td align="left">fine-tuned checkpoint</td><td align="left">reference ImageNet accuracy</td></tr>
<tr>
<td align="center"><a href='https://huggingface.co/longlian/CrossMAE/resolve/main/vits-mr0.75-kmr0.25-dd12/imagenet-mae-cross-vits-pretrain-wfm-mr0.75-kmr0.75-dd12-ep800-ui.pth?download=true'>download</a></td>
<td align="center"><a href='https://huggingface.co/longlian/CrossMAE/resolve/main/vits-mr0.75-kmr0.25-dd12/imagenet-mae-cross-vits-finetune-wfm-mr0.75-kmr0.75-dd12-ep800-ui.pth?download=true'>download</a></td>
<td align="center">79.462</td>
</tr>
</tbody></table>
</details>

## Instructions
Please install the dependencies in `requirements.txt`:
```sh
# Optionally create a conda environment
conda create -n crossmae python=3.10 -y
conda activate crossmae
# Install dependencies
pip install -r requirements.txt
```

### Pre-training CrossMAE
To pre-train ViT-Base, run the following on 4 GPUs:
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 1234 main_pretrain.py --batch_size 1024 --model mae_vit_base_patch16 --norm_pix_loss --blr 1.5e-4 --weight_decay 0.05 --data_path ${IMAGENET_DIR} --num_workers 20 --enable_flash_attention2 --multi_epochs_dataloader --output_dir output/imagenet-crossmae-vitb-pretrain-wfm-mr0.75-kmr0.25-dd12-ep800 --cross_mae --weight_fm --decoder_depth 12 --mask_ratio 0.75 --kept_mask_ratio 0.25 --epochs 800 --warmup_epochs 40 --use_input
```

To train ViT-Small or ViT-Large, set `--model mae_vit_small_patch16` or `--model mae_vit_large_patch16`. You can use `--accum_iter` to perform gradient accumulation if your hardware could not fit the batch size. [FlashAttention 2](https://github.com/Dao-AILab/flash-attention) should be installed with `pip install flash-attn --no-build-isolation`.

### Fine-tuning CrossMAE
To pre-train ViT-Base, run the following on 4 GPUs:
```sh
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 1234 main_finetune.py --batch_size 256 --model vit_base_patch16 --finetune output/imagenet-crossmae-vitb-pretrain-wfm-mr0.75-kmr0.25-dd12-ep800/checkpoint.pth --epoch 100 --blr 5e-4 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --dist_eval --data_path ${IMAGENET_DIR} --output_dir output/imagenet-crossmae-vitb-finetune-wfm-mr0.75-kmr0.25-dd12-ep800 --enable_flash_attention2 --multi_epochs_dataloader
```

## Evaluation
Evaluate ViT-Base in a single GPU (`${IMAGENET_DIR}` is a directory containing `{train, val}` sets of ImageNet). `${FINETUNED_CHECKPOINT_PATH}` is the path to the fine-tuned checkpoint:
```sh
python main_finetune.py --eval --resume ${FINETUNED_CHECKPOINT_PATH} --model vit_base_patch16 --batch_size 16 --data_path ${IMAGENET_DIR}
```
This should give:
```
* Acc@1 83.722 Acc@5 96.686 loss 0.729
```

You could replace `vit_base_patch16` with `vit_small_patch16` or `vit_large_patch16` to evaluate ViT-S or ViT-L. To work with 448 input resolution, please append `--input_size 448` to the command line.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## Citation
Please give us a star 🌟 on Github to support us!

Please cite our work if you find our work inspiring or use our code in your work:
```
@article{fu2024rethinking,
    title={Rethinking Patch Dependence for Masked Autoencoders}, 
    author={Letian Fu and Long Lian and Renhao Wang and Baifeng Shi and Xudong Wang and Adam Yala and Trevor Darrell and Alexei A. Efros and Ken Goldberg},
    journal={arXiv preprint arXiv:2401.14391},
    year={2024}
}
```
