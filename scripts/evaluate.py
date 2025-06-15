import argparse
import numpy as np
import os

from cleanfid import fid
from rectified_flow.metrics.clip_score import calculate_clipscore


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_clipscore",
        action="store_true",
        help="Evaluate clip score",
    )
    parser.add_argument(
        "--eval_fid",
        action="store_true",
        help="Evaluate FID",
    )
    parser.add_argument(
        "--gen_img_dir",
        type=str,
        default=None,
        help="Path to generated image directory for FID / CLIP evaluation",
    )
    parser.add_argument(
        "--gt_img_dir",
        type=str,
        default=None,
        help="Path to ground truth image directory for FID evaluation",
    )
    parser.add_argument(
        "--fid_model",
        type=str,
        default="clip_vit_b_32",
        choices=["inception_v3", "clip_vit_b_32", "clip_vit_l_14"],
        help="FID model to use for evaluation",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="ViT-L/14",
        choices=["ViT-B/32", "ViT-L/14"],
        help="CLIP model to use for evaluation",
    )
    parser.add_argument(
        "--prompt_txt_path",
        type=str,
        default=None,
        help="Path to prompt text file for CLIP evaluation",
    )

    args = parser.parse_args()
    return args


def eval_fid(args):
    fid_score = fid.compute_fid(
        args.gt_img_dir, args.gen_img_dir, mode="clean", model_name=args.fid_model
    )
    print("FID Score: {:.4f}".format(fid_score))


def eval_clipscore(args):
    image_list = [
        os.path.join(args.gen_img_dir, f)
        for f in os.listdir(args.gen_img_dir)
        if f.endswith(".png") or f.endswith(".jpg")
    ]
    image_list.sort()

    # read prompt
    with open(args.prompt_txt_path, "r") as fr:
        text_list = fr.readlines()
        text_list = [_.strip() for _ in text_list]

    assert len(image_list) == len(text_list), "Number of images and texts do not match"

    clipscore = calculate_clipscore(image_list, text_list)

    print("CLIP Score: {:.4f}".format(clipscore))


def main(args):
    if args.eval_fid:
        assert (
            args.gt_img_dir is not None
        ), "Please provide path to ground truth image directory"
        eval_fid(args)
    elif args.eval_clipscore:
        assert (
            args.prompt_txt_path is not None
        ), "Please provide path to prompt text file"
        eval_clipscore(args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
