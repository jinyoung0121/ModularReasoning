import os
from omegaconf import OmegaConf
import argparse
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser()
    
    # llama argument
    llama = parser.add_argument_group("llama")
    llama.add_argument('--llama.model_path', type=str, default='./pretrained_model/Meta-Llama-3.1-8B-Instruct', help='llama path')

    # internlm argument
    internlm = parser.add_argument_group("internlm")
    internlm.add_argument('--internlm.model_path', type=str, default='./pretrained_model/internlm2_5-7b-chat', help='llama path')

    # blip argument
    blip = parser.add_argument_group("blip")
    blip.add_argument('--blip.model_path', type=str, default='./pretrained_model/blip2-flan-t5-xxl', help='blip path')
    
    # internvl argument
    internvl = parser.add_argument_group("internvl")
    internvl.add_argument('--internvl.model_path', type=str, default='./pretrained_model/InternVL2-8B', help='internvl2 path')

    # qd_detr argument
    qd_detr = parser.add_argument_group("qd_detr")
    qd_detr.add_argument('--qd_detr.model_checkpoint_path', type=str, default='./pretrained_model/QD_DETR/run_on_video/qd_detr_ckpt/model_best.ckpt', help='qd_detr best checkpoint path')
    qd_detr.add_argument('--qd_detr.clip_model', type=str, default='ViT-B/32', help='clip model type in qd_detr')

    # owlvit argument
    owlvit = parser.add_argument_group("owlvit")
    owlvit.add_argument('--owlvit.model_path', type=str, default='./pretrained_model/owlvit-base-patch32', help='owlvit path')
    owlvit.add_argument('--owlvit.threshold', type=float, default=0.1, help='bbox threshold of owlvit')

    # clip argument
    clip = parser.add_argument_group("clip")
    clip.add_argument('--clip.model_path', type=str, default='RN50', help='clip model type')
    clip.add_argument('--clip.threshold', type=float, default=0.7, help='clip image-text comparison threshold')

    # wandb arguments
    wandb = parser.add_argument_group("wandb")
    wandb.add_argument("--wandb.use_wandb", action='store_true')
    wandb.add_argument("--wandb.project", default="morevqa")
    wandb.add_argument("--wandb.group", default="default")

    # distributed arguments
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--num_workers', default=2, type=int)

    # others
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()

    return args

def _build_opt_list(opts):
    opts_dot_list = _convert_to_dot_list(opts)
    return OmegaConf.from_dotlist(opts_dot_list)

def _convert_to_dot_list(opts):
    if opts is None:
        opts = []

    if len(opts) == 0:
        return opts

    has_equal = opts[0].find("=") != -1

    if has_equal:
        return opts

    return [(opt + "=" + value) for opt, value in zip(opts[0::2], opts[1::2])]


def build_custom_config(args):
    custom_config = defaultdict(dict)
    for (k, v) in args._get_kwargs():
        if len(k.split('.')) == 2:
            custom_config[k.split('.')[0]][k.split('.')[1]] = v
        else:
            custom_config[k] = v
    return dict(custom_config)
    
configs = OmegaConf.load('configs/base_config.yaml')

# unsafe_merge makes the individual configs unusable, but it is faster
user_config = _build_opt_list(parse_args().options)
custom_config = build_custom_config(parse_args())
config = OmegaConf.merge(configs, user_config, custom_config)