import os
import math
import logging
import argparse

import shutil
from pathlib import Path

import torch
import copy
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (DistributedDataParallelKwargs, ProjectConfiguration, set_seed,)
from diffusers.optimization import get_scheduler
from torchvision import transforms
from tqdm.auto import tqdm

from rectified_flow.models.unet import SongUNet, SongUNetConfig
from rectified_flow.RFlow import RectifiedFlow
from rectified_flow.datasets.coupling_dataset import CouplingDataset
from rectified_flow.datasets.coupling_dataset import coupling_collate_fn
from rectified_flow.datasets.flat_image_dataset import FlatImageDataset

from torch.utils.data import DataLoader
from functools import partial
from datetime import datetime

# Logger setup
logger = get_logger(__name__)

# EMA Class
class EMAModel:
    def __init__(
        self,
        net: torch.nn.Module,
        ema_halflife_kimg: float = 2000.0,
        ema_rampup_ratio: float = 0.05,
    ):
        """
        Initialize an EMA (Exponential Moving Average) model for stabilizing training.

        Args:
            net (torch.nn.Module): The main model to track.
            ema_halflife_kimg (float): Half-life of EMA in thousands of images.
            ema_rampup_ratio (float): Ratio of total seen images used for ramp-up phase.
        """
        self.net = net
        self.ema = copy.deepcopy(net).eval().float()
        for param in self.ema.parameters():
            param.requires_grad_(False)
        self.ema_halflife_kimg = ema_halflife_kimg
        self.ema_rampup_ratio = ema_rampup_ratio

    @torch.no_grad()
    def update(self, cur_nimg: int, batch_size: int):
        """
        Update EMA parameters using a half-life strategy.

        Args:
            cur_nimg (int): Total number of images processed so far.
            batch_size (int): Current global batch size.
        """
        ema_halflife_nimg = self.ema_halflife_kimg * 1000  # Convert kimg to nimg

        if self.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * self.ema_rampup_ratio)

        beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))

        for p_ema, p_net in zip(self.ema.parameters(), self.net.parameters()):
            p_ema.copy_(p_net.float().lerp(p_ema, beta))

    def apply_shadow(self):
        """
        Copy EMA parameters back to the original `net`.
        Useful for inference after training.
        """
        for p_net, p_ema in zip(self.net.parameters(), self.ema.parameters()):
            p_net.data.copy_(p_ema.data.to(p_net.dtype))

    def save_pretrained(self, save_directory: str, filename: str = "unet"):
        """
        Save the EMA model parameters to a file.

        Args:
            save_directory (str): Directory to save the weights.
            filename (str): Base filename (without suffix).
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save state dict to CPU to avoid GPU memory leaks
        state_dict_cpu = {k: v.cpu() for k, v in self.ema.state_dict().items()}
        output_model_file = os.path.join(save_directory, f"{filename}_ema.pt")
        torch.save(state_dict_cpu, output_model_file)
        print(f"EMA model weights saved to {output_model_file}")

    def load_pretrained(self, save_directory: str, filename: str = "unet"):
        """
        Load EMA model parameters from a file.

        Args:
            save_directory (str): Path to checkpoint directory.
            filename (str): Base filename (without suffix).
        """
        output_model_file = os.path.join(save_directory, f"{filename}_ema.pt")
        if os.path.exists(output_model_file):
            state_dict = torch.load(output_model_file, map_location="cpu")
            self.ema.load_state_dict(state_dict, strict=True)
            net_device = next(self.net.parameters()).device
            self.ema.to(device=net_device, dtype=torch.float32)
            print(f"EMA weights loaded from {output_model_file}")
        else:
            raise FileNotFoundError(f"No EMA weights found at {output_model_file}") #Added error handling for missing EMA weights
    
# Image transform
def get_transform(resolution):
    transform_list = [
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    if args.random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    return transforms.Compose(transform_list)

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Rectified Flow Training")

    # Dataset args
    parser.add_argument("--data_source", type=str, default=None, help="Path to source domain images.")
    parser.add_argument("--data_target", type=str, default=None, help="Path to target domain images.")

    # Output settings
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory.")
    parser.add_argument("--validation_epochs", type=int, default=5, help="Run validation every X epochs.")
    parser.add_argument("--checkpointing_steps", type=int, default=10000, help="Save a checkpoint every X steps.")
    parser.add_argument("--checkpoints_total_limit", type=int, default=None, help="Maximum number of checkpoints to keep.")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If non-null, resume the specified training istance.")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution to train at.")

    # Training config
    parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size per device.")
    parser.add_argument("--max_train_steps", type=int, default=500000, help="Total number of training steps.")
    parser.add_argument("--num_train_epochs",type=int, default=1, help="Total number of training epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients over N steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, choices=[1e-5, 5e-5, 1e-4, 5e-4, 1e-3], help="Initial learning rate.")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])
    parser.add_argument("--scale_lr", action="store_true", default=False, help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.")

    # Rectified Flow args
    parser.add_argument("--interp", type=str, default="straight", choices=["straight", "slerp", "ddim"])
    parser.add_argument("--source_distribution", type=str, default="normal")
    parser.add_argument("--is_independent_coupling", action="store_true")
    parser.add_argument("--train_time_distribution", type=str, default="uniform")
    parser.add_argument("--train_time_weight", type=str, default="uniform")
    parser.add_argument("--criterion", type=str, default="mse", help="Criterion for the rectified flow. Choose between ['mse', 'l1', 'lpips'].")

    # Misc
    parser.add_argument("--random_flip", action="store_true", help="Random horizontal flip")
    parser.add_argument("--use_ema", type=bool,default=True, help="Use EMA model during training")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--report_to", type=str, default="tensorboard", choices=["tensorboard", "wandb", "comet_ml", "none"], help="The integration to report the results and logs to.")
    parser.add_argument("--unpaired", type=bool, default=True, help="Train using unpaired data")
    parser.add_argument("--logging_dir", type=str, default=None, help="[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.")
    parser.add_argument("--allow_tf32", action="store_true", help="Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices")
    
    args = parser.parse_args()

    return args

# Main function
def main(args):

    # Clear GPU and check availability
    torch.cuda.empty_cache()
    print("Using GPU:", torch.cuda.is_available())

    # Setup accelerator
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.logging_dir)

    accelerator = Accelerator(
        log_with=args.report_to,
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        project_config=project_config,
        kwargs_handlers=[kwargs]
    )

    # Custom save/load hooks 
    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(accelerator.unwrap_model(model), SongUNet):
                    unwrap_model = accelerator.unwrap_model(model)
                    unwrap_model.save_pretrained(output_dir, filename="unet")
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")
        weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            model = models.pop()
            if isinstance(accelerator.unwrap_model(model), SongUNet):
                load_model = SongUNet.from_pretrained(input_dir, filename="unet")
                model.load_state_dict(load_model.state_dict())
            else:
                raise ValueError(f"Wrong model supplied: {type(model)=}.")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook) 

    # Set weight dtype based on mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Set up logging
    logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.WARNING,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Seed and device
    if args.seed is not None:
        set_seed(args.seed)

    # Prepare model
    logger.info("******  Preparing Model  ******")
    model_config = SongUNetConfig(
        img_resolution=args.resolution,
        in_channels=3,
        out_channels=3,
    )
    model = SongUNet(model_config)
    print("[DEBUG] Model created")
    model.to(accelerator.device, dtype=weight_dtype)
    model.train().requires_grad_(True)
    print(f"[DEBUG] Model moved to device {accelerator.device}")

    # EMA
    if args.use_ema:
        model_ema = EMAModel(model)

    # Prepare datasets
    logger.info("******  Preparing datasets  ******")
    transform = get_transform(args.resolution)
    dataset_A = FlatImageDataset(root=args.data_source, transform=transform, ext="jpg")
    dataset_B = FlatImageDataset(root=args.data_target, transform=transform, ext="jpg")

    train_dataloader_A = DataLoader(dataset_A, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    train_dataloader_B = DataLoader(dataset_B, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Scale learning rate
    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Optimizer and scheduler
    model_params_with_lr = {"params": model.parameters(), "lr": args.learning_rate}
    params_to_optimize = [model_params_with_lr]
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 betas=(args.adam_beta1, args.adam_beta2),
                                 eps=args.adam_epsilon)

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power
    )

    # Prepare with accelerator
    models_to_prepare = [model]
    if args.use_ema:
        models_to_prepare.append(model_ema)

    if args.use_ema:
        model, model_ema, optimizer, train_dataloader_A, train_dataloader_B, lr_scheduler = accelerator.prepare(
            model, model_ema, optimizer, train_dataloader_A, train_dataloader_B, lr_scheduler
        )
    else:
        model, optimizer, train_dataloader_A, train_dataloader_B, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader_A, train_dataloader_B, lr_scheduler
        )

    # Rectified Flow setup
    rectified_flow = RectifiedFlow(
        data_shape=(3, args.resolution, args.resolution),
        interp=args.interp,
        source_distribution=args.source_distribution,
        is_independent_coupling=args.is_independent_coupling,
        train_time_distribution=args.train_time_distribution,
        train_time_weight=args.train_time_weight,
        criterion=args.criterion,
        velocity_field=model,
        device=accelerator.device,
        dtype=torch.float32
    )

    # Setup output directories
    global_step = 0
    first_epoch = 0
    initial_global_step = 0
 
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

        # Find existing runs
        existing_runs = [d for d in os.listdir(args.output_dir) if d.startswith("training_") and os.path.isdir(os.path.join(args.output_dir, d))]
        existing_runs.sort(key=lambda x: int(x.split("_")[1]))

        # If resuming form a specific run
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint == "latest":
                args.resume_from_checkpoint = existing_runs[-1] if len(existing_runs) > 0 else None

            if args.resume_from_checkpoint and args.resume_from_checkpoint.startswith("training_"):
                resume_run_path = os.path.join(args.output_dir, args.resume_from_checkpoint)

                if os.path.isdir(resume_run_path):
                    # Find checkpoints inside this run
                    dirs = os.listdir(resume_run_path)
                    dirs = [d for d in dirs if d.startswith("checkpoint")]
                    dirs.sort(key=lambda x: int(x.split("-")[1]))
                    if len(dirs) > 0:
                        path = dirs[-1]  # Latest checkpoint
                        args.run_output_dir = resume_run_path
                        args.run_name = args.resume_from_checkpoint
                        checkpoint_path = os.path.join(resume_run_path, path)
                        logger.info(f"Resuming from {checkpoint_path}")
                        accelerator.load_state(checkpoint_path)

                        if args.use_ema:
                            try:
                                model_ema.load_pretrained(checkpoint_path, filename="unet")
                                logger.info("Loaded EMA weights")
                            except Exception as e:
                                logger.warning(f"Could not load EMA weights: {e}")

                        global_step = int(path.split("-")[1])
                        first_epoch = global_step // math.ceil(len(train_dataloader_A) / args.gradient_accumulation_steps)
                        initial_global_step = global_step
                    else:
                        logger.info("No valid checkpoints found in the specified run. Starting fresh.")
                        args.resume_from_checkpoint = None
                        args.run_name = f"training_{1:03d}"
                        args.run_output_dir = os.path.join(args.output_dir, args.run_name)
                        os.makedirs(args.run_output_dir, exist_ok=False)
                else:
                    logger.warning(f"Specified training run '{args.resume_from_checkpoint}' does not exist. Starting new run.")
                    args.resume_from_checkpoint = None
                    next_run_number = 1 if not existing_runs else max(int(d.split('_')[1]) for d in existing_runs) + 1
                    args.run_name = f"training_{next_run_number:03d}"
                    args.run_output_dir = os.path.join(args.output_dir, args.run_name)
                    os.makedirs(args.run_output_dir, exist_ok=False)
            else:
                logger.warning(f"Invalid resume value: {args.resume_from_checkpoint}. Starting new run.")
                args.resume_from_checkpoint = None
                next_run_number = 1 if not existing_runs else existing_runs[-1].split("_")[1] + 1
                args.run_name = f"training_{next_run_number:03d}"
                args.run_output_dir = os.path.join(args.output_dir, args.run_name)
                os.makedirs(args.run_output_dir, exist_ok=False)

        else:
            # No resume flag → create new run
            next_run_number = 1
            if existing_runs:
                nums = [int(d.split('_')[1]) for d in existing_runs]
                next_run_number = max(nums) + 1
            args.run_name = f"training_{next_run_number:03d}"
            args.run_output_dir = os.path.join(args.output_dir, args.run_name)
            os.makedirs(args.run_output_dir, exist_ok=False)
            logger.info(f"Created new training run: {args.run_name}")
    else:
        args.run_output_dir = None

    accelerator.wait_for_everyone()

    # Calculate steps per epoch 
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader_A) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        if args.num_train_epochs is None:
            raise ValueError("Either num_train_epochs or max_train_steps must be provided.")
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    total_batch_size = (args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps)

    if accelerator.is_main_process:
        tracker_name = "1rf-dit-cifar"
        accelerator.init_trackers(tracker_name, config=vars(args))
        

    # Progress bar
    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step,
                       desc="Steps", disable=not accelerator.is_local_main_process)

    # Iterators for unpaired sampling
    dataloader_A_iter = iter(train_dataloader_A)
    dataloader_B_iter = iter(train_dataloader_B)

    logger.info("****** Starting training ******")

    while global_step < args.max_train_steps:
        model.train()

        # Sample from each domain independently
        try:
            batch_A = next(dataloader_A_iter)
        except StopIteration:
            dataloader_A_iter = iter(train_dataloader_A)
            batch_A = next(dataloader_A_iter)

        try:
            batch_B = next(dataloader_B_iter)
        except StopIteration:
            dataloader_B_iter = iter(train_dataloader_B)
            batch_B = next(dataloader_B_iter)

        x_A = batch_A.to(accelerator.device)
        x_B = batch_B.to(accelerator.device)

        # Sample time
        t = rectified_flow.sample_train_time(x_A.shape[0])

        # Compute loss 
        if args.unpaired:
            loss_AB = rectified_flow.get_loss(x_0=x_A, x_1=x_B, t=t).mean()
        else:
            train_dataset = CouplingDataset(D0=dataset_A, D1=dataset_B, reflow=True)
            x_0, x_1 = train_dataset[global_step % len(train_dataset)]
            x_0 = x_0.unsqueeze(0).to(accelerator.device)
            x_1 = x_1.unsqueeze(0).to(accelerator.device)
            loss_AB = rectified_flow.get_loss(x_0=x_0, x_1=x_1, t=t).mean()

        # Backward pass
        accelerator.backward(loss_AB)

        # Update EMA
        if args.use_ema:
            model_ema.update(cur_nimg=global_step * args.train_batch_size, batch_size=args.train_batch_size)

        # Optimize and update LR
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        # Log
        logs = {"loss": loss_AB.item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)
        accelerator.log(logs, step=global_step)
        progress_bar.update(1)
        global_step += 1

        # Save checkpoint
        if accelerator.is_main_process:
            if global_step % args.checkpointing_steps == 0:
                save_path = os.path.join(args.run_output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

                if args.use_ema:
                    model_ema.save_pretrained(save_path, filename="unet")
                    logger.info(f"Saved EMA model to {save_path}")

                # Prune old checkpoints
                if args.checkpoints_total_limit is not None:
                    checkpoints = os.listdir(args.output_dir)
                    checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        for to_remove in checkpoints[:len(checkpoints) - args.checkpoints_total_limit + 1]:
                            shutil.rmtree(os.path.join(args.output_dir, to_remove))

    # Final save
    if accelerator.is_main_process:
        if args.checkpointing_steps > 0:
            save_path = os.path.join(args.run_output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(save_path)
            logger.info(f"Saved final state to {save_path}")
            if args.use_ema:
                model_ema.save_pretrained(save_path, filename="unet")
                logger.info(f"Saved final EMA model to {save_path}")

    # End training
    accelerator.end_training()

# Run the main function
if __name__ == "__main__":
    args = parse_args()
    main(args)