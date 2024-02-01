from functools import partial
import torch
from torch import nn
from models_cross import MaskedAutoencoderViT
from huggingface_hub import PyTorchModelHubMixin
import argparse


class MaskedAutoencoderViTHF(MaskedAutoencoderViT, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__(**config, norm_layer=partial(nn.LayerNorm, eps=1e-6))


def ckpt_to_hub(ckpt_path, hub_path, config):
    model = MaskedAutoencoderViTHF(config)
    if ckpt_path.startswith("https://"):
        state_dict = torch.hub.load_state_dict_from_url(ckpt_path, map_location="cpu")
    else:
        state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    model.push_to_hub(hub_path, config=config)


base_config = dict(
    norm_pix_loss=True,
    weight_fm=True,
    decoder_depth=12,
    use_fm=[-1],
    use_input=True,
    self_attn=False,
)
config_dict = {
    "vit_small_patch16": dict(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        decoder_embed_dim=256,
        decoder_num_heads=8,
        mlp_ratio=4,
    ),
    "vit_base_patch16": dict(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        mlp_ratio=4,
    ),
    "vit_large_patch16": dict(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        mlp_ratio=4,
    ),
    "vit_huge_patch16": dict(
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_num_heads=16,
        mlp_ratio=4,
    ),
}


def main():
    parser = argparse.ArgumentParser("Push trained CrossMAE checkpoints to hub")
    parser.add_argument(
        "--model",
        default="vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--ckpt-path", type=str, help="Path to CrossMAE checkpoint")
    parser.add_argument(
        "--hub-path", type=str, help="path on hub ([username]/[model name])"
    )
    parser.add_argument("--input_size", default=224, type=int, help="images input size")

    args = parser.parse_args()
    config = {**base_config, **config_dict[args.model], "img_size": args.input_size}

    ckpt_path = args.ckpt_path
    hub_path = args.hub_path
    ckpt_to_hub(ckpt_path, hub_path, config)


if __name__ == "__main__":
    # Example:
    # python -m util.hf_hub --model vit_base_patch16 --ckpt-path "https://huggingface.co/longlian/CrossMAE/resolve/main/vitb-mr0.75-kmr0.75-dd12/imagenet-mae-cross-vitb-pretrain-wfm-mr0.75-kmr0.75-dd12-ep800-ui.pth?download=true" --hub-path longlian/crossmae-base-patch16

    main()
