# run_training.ps1
python rectified_flow/pipelines/train_unet_cifar.py `
  --data_source "./data/cities/Bari/1" `
  --data_target "./data/cities/Tokyo/1" `
  --output_dir "./output" `
  --interp "straight" `
  --use_ema `
  --resolution 256 `
  --train_batch_size 8 `
  --max_train_steps 100 `
  --num_train_epochs 10 `
  --checkpointing_steps 10 `
  --learning_rate 2e-4 `
  --random_flip