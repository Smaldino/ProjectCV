export OUTPUT_DIR=""
export DATA_ROOT=""

accelerate launch -m rectified_flow.pipelines.train_dit_cifar \
  --output_dir="$OUTPUT_DIR" \
  --resume_from_checkpoint="latest" \
  --data_root="$DATA_ROOT" \
  --seed=0 \
  --resolution=32 \
  --train_batch_size=128 \
  --max_train_steps=1000000 \
  --checkpointing_steps=20000 \
  --learning_rate=2e-4 \
  --adam_beta1=0.99 \
  --adam_beta2=0.999 \
  --adam_weight_decay=0.03 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=2500 \
  --random_flip \
  --allow_tf32 \
  --interp="straight" \
  --source_distribution="normal" \
  --is_independent_coupling=True \
  --train_time_distribution="lognormal" \
  --train_time_weight="uniform" \
  --criterion="mse" \
  --use_ema