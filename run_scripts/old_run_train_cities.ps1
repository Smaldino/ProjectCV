# This script sets up the environment and runs the training script for a UNet model
# $env:PYTHONPATH = "C:\Users\Fabbro\Documents\Universita\CaseStudy"


# Set paths
$data_souce = "./data/cities/Bari"
$data_target = "./data/cities/Tokyo"
$output_dir = "./output/BariToTokyo"
$interp = "straight" # "straight", "slerp" or "ddim"

# .\scripts\run_training.ps1

python .\scripts\train_cities.py `
  --data_source "$data_souce" `
  --data_target "$data_target" `
  --output_dir "$output_dir" `
  --interp "$interp" `
  --use_ema `
  --resolution 128 `
  --train_batch_size 16 `
  --max_train_steps 10000 `
  --num_train_epochs 50 `
  --checkpointing_steps 2000 `
  --checkpoints_total_limit 5 `
  --learning_rate 1e-3 `
  --lr_scheduler "constant_with_warmup" `
  --lr_warmup_steps 500 `
  --gradient_accumulation_steps 2 `
  --random_flip `
  --mixed_precision "bf16"