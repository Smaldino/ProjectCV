
# This script evaluates the FID score of generated images against ground truth images.

# Set paths
$gt_img_dir = ".\data\cities\Tokyo"
$gen_img_dir = "data\cities_trasfromed\BariToTokyo\training_001_transformation"
$fid_model = "clip_vit_b_32"  # Options: inception_v3, clip_vit_b_32, clip_vit_l_14

# Run evaluate.py
# .\run_scripts\run_evaluate.ps1

python .\scripts\evaluate.py `
    --gen_img_dir "$gen_img_dir" `
    --gt_img_dir "$gt_img_dir" `
    --fid_model "$fid_model"