
# Runs the training script for a UNet model

# Set paths
$data_source = "./data/cities/Bari"
$data_target = "./data/cities/Tokyo"
$output_dir = "./output/BariToTokyo"
$interp = "straight"  # "straight", "slerp", or "ddim"

# Run train_cities.py
# .\run_scripts\run_train_cities.ps1

python .\scripts\train_cities.py `
  --data_source "$data_source" `
  --data_target "$data_target" `
  --output_dir "$output_dir" `
  --resolution 256 `
  --interp "$interp" `
  --train_batch_size 8 `
  --num_train_epochs 50 `
  --max_train_steps 10000 `
  --learning_rate 1e-4 `
  --lr_scheduler "constant_with_warmup" `
  --lr_warmup_steps 500 `
  --gradient_accumulation_steps 2 `
  --mixed_precision "bf16" `
  --random_flip `
  --allow_tf32 `
  --checkpointing_steps 2000 `
  --checkpoints_total_limit 5


  # tensorboard --logdir=./output/logs