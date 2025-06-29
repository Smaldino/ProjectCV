import argparse
import numpy as np
import os

from cleanfid import fid

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_fid", type=bool,default=True, help="Evaluate FID")
    parser.add_argument("--gen_img_dir", type=str, default=None, help="Path to generated image directory for FID evaluation")
    parser.add_argument("--gt_img_dir", type=str, default=None, help="Path to ground truth image directory for FID evaluation")
    parser.add_argument("--fid_model", type=str, default="clip_vit_b_32", choices=["inception_v3", "clip_vit_b_32", "clip_vit_l_14"], help="FID model to use for evaluation")
    args = parser.parse_args()
    return args


def eval_fid(args):
    fid_score = fid.compute_fid(args.gen_img_dir, args.gt_img_dir, mode="clean", model_name=args.fid_model, num_workers=0)
    print("FID Score: {:.4f}".format(fid_score))

# Main 
def main(args):
    if args.eval_fid:
        assert (
            args.gt_img_dir is not None
        ), "Please provide path to ground truth image directory"
        eval_fid(args)

if __name__ == "__main__":
    args = parse_args()
    main(args)
 