import argparse
import logging
import math
import os
import shutil
from pathlib import Path

import torch
import copy
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    ProjectConfiguration,
    set_seed,
)
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

logger = get_logger(__name__)

class EMAModel:
    def __init__(
        self, 
        net: torch.nn.Module, 
        ema_halflife_kimg: float = 2000.0, 
        ema_rampup_ratio: float = 0.05,
    ):
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
            cur_nimg (int): The current number of images (could be total images processed so far).
            batch_size (int): The global batch size.
        """
        ema_halflife_nimg = self.ema_halflife_kimg * 1000

        if self.ema_rampup_ratio is not None:
            ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * self.ema_rampup_ratio)

        beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))

        for p_ema, p_net in zip(self.ema.parameters(), self.net.parameters()):
            p_ema.copy_((p_net.float()).lerp(p_ema, beta))

    def apply_shadow(self):
        """
        Copy EMA parameters back to the original `net`.
        """
        for p_net, p_ema in zip(self.net.parameters(), self.ema.parameters()):
            p_net.data.copy_(p_ema.data.to(p_net.dtype))

    def save_pretrained(self, save_directory: str, filename: str = "unet"):
        """
        Save the EMA model parameters to a file.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        state_dict_cpu = {k: v.cpu() for k, v in self.ema.state_dict().items()}
        output_model_file = os.path.join(save_directory, f"{filename}_ema.pt")
        torch.save(state_dict_cpu, output_model_file)
        print(f"EMA model weights saved to {output_model_file}")
    
    def load_pretrained(self, save_directory: str, filename: str = "unet"):
        """
        Load EMA model parameters from a file.
        """
        output_model_file = os.path.join(save_directory, f"{filename}_ema.pt")
        if os.path.exists(output_model_file):
            state_dict = torch.load(output_model_file, map_location="cpu")
            self.ema.load_state_dict(state_dict, strict=True)
            net_device = next(self.net.parameters()).device
            self.ema.to(device=net_device, dtype=torch.float32)
            print(f"EMA weights loaded from {output_model_file}")
        else:
            print(f"No EMA weights found at {output_model_file}")

# 1. Parse Arguments
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
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="If non-null, resume training from this checkpoint.")

    # Model config
    parser.add_argument("--resolution", type=int, default=256, help="Resolution to train at.")
    parser.add_argument("--model_channels", type=int, default=64)
    parser.add_argument("--channel_mult", nargs='+', type=int, default=[2, 2, 2])
    parser.add_argument("--num_blocks", type=int, default=2)

    # Training config
    parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size per device.")
    parser.add_argument("--max_train_steps", type=int, default=500000, help="Total number of training steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Accumulate gradients over N steps.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"])

    # Rectified Flow args
    parser.add_argument("--interp", type=str, default="straight", choices=["straight", "slerp", "ddim"])
    parser.add_argument("--source_distribution", type=str, default="normal")
    parser.add_argument("--is_independent_coupling", action="store_true")
    parser.add_argument("--train_time_distribution", type=str, default="uniform")
    parser.add_argument("--train_time_weight", type=str, default="uniform")
    parser.add_argument("--criterion", type=str, default="mse")

    # Misc
    parser.add_argument("--random_flip", action="store_true", help="Random horizontal flip")
    parser.add_argument("--use_ema", action="store_true", help="Use EMA model during training")
    parser.add_argument("--seed", type=int, default=None, help="Seed for reproducibility")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--report_to",type=str,default="tensorboard",choices=["tensorboard", "wandb", "comet_ml", "none"],help="The integration to report the results and logs to.",)

    # Mode selection
    parser.add_argument("--unpaired", action="store_true", help="Train using unpaired data (no scene alignment)")

    args = parser.parse_args()

    return args
    print(f"[DEBUG] Parsed arguments: {args}")

def main(args):

    # Clear GPU and check availability
    torch.cuda.empty_cache()
    print("Using GPU:", torch.cuda.is_available())

    # Setup accelerator
    logging_dir = Path(args.output_dir, args.logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.WARNING,
    )
    logger.info(accelerator.state, main_process_only=False)

    # Set weight dtype based on mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Seed and device
    if args.seed is not None:
        set_seed(args.seed)

    # Logging and output setup
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

            existing_runs = [d for d in os.listdir(args.output_dir) if d.startswith("training_") and os.path.isdir(os.path.join(args.output_dir, d))]
            if existing_runs:
                nums = [int(d.split('_')[1]) for d in existing_runs]
                next_run_number = max(nums) + 1
            else:
                next_run_number = 1
            args.run_name = f"training_{next_run_number:03d}"
            args.run_output_dir = os.path.join(args.output_dir, args.run_name)
            os.makedirs(args.run_output_dir, exist_ok=False)
        else:
            args.run_output_dir = None
    else:
        args.run_output_dir = None

    # 
    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Prepare model
    logger.info("******  Preparing models  ******")
    model_config = SongUNetConfig(
        img_resolution=args.resolution,
        in_channels=3,
        out_channels=3,
        model_channels=args.model_channels,
        channel_mult=args.channel_mult,
        num_blocks=args.num_blocks
    )
    model = SongUNet(model_config)
    print("[DEBUG] Model created")
    model.to(accelerator.device, dtype=weight_dtype)
    model.train().requires_grad_(True)
    print(f"[DEBUG] Model moved to device {accelerator.device}")

    # EMA model
    if args.use_ema:
        model_ema = EMAModel(model)

    # Define transform 
    transform_list = []
    if args.random_flip:
        transform_list.append(transforms.RandomHorizontalFlip(0.5))
    transform_list.extend(
        [
            transforms.Resize((args.resolution, args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
            ),
        ]
    )
    transform = transforms.Compose(transform_list)

    # Prepare datasets
    logger.info("******  Preparing datasets  ******")

    dataset_source = FlatImageDataset(root=args.data_source, transform=transform, ext="jpg")
    dataset_target = FlatImageDataset(root=args.data_target, transform=transform, ext="jpg")

    train_dataloader_A = DataLoader(dataset_source, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)
    train_dataloader_B = DataLoader(dataset_target, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_workers)

    # Prepare optimizers and schedulers
    model_params_with_lr = {"params": model.parameters(), "lr": args.learning_rate}
    params_to_optimize = [model_params_with_lr]

    optimizer = torch.optim.Adam(
        params_to_optimize,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    
    # Calculate number of update steps per epoch and possibly override max_train_steps
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    # Traing setup
    logger.info("******  Preparing for training  ******")

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for i, model in enumerate(models):
                if isinstance(accelerator.unwrap_model(model), SongUNet):
                    unwrap_model = accelerator.unwrap_model(model)
                    unwrap_model.save_pretrained(output_dir, filename="unet")
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")
            for model in models:
                if isinstance(accelerator.unwrap_model(model), SongUNet):
                    unwrap_model = accelerator.unwrap_model(model)
                    unwrap_model.save_pretrained(output_dir, filename="unet")
                else:
                    raise ValueError(f"Wrong model supplied: {type(model)=}.")
                weights.pop()

    def load_model_hook(models, input_dir):
        for _ in range(len(models)):
            model = models.pop()
            print(f"Loading model {type(model)} from {input_dir}")

            if isinstance(accelerator.unwrap_model(model), SongUNet):
                load_model = SongUNet.from_pretrained(input_dir, filename="unet")
                model.load_state_dict(load_model.state_dict())
            else:
                raise ValueError(f"Wrong model supplied: {type(model)=}.")

            del load_model

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
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
        dtype=weight_dtype,
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        tracker_name = "1rf-dit-cifar"
        accelerator.init_trackers(tracker_name, config=vars(args))

    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    # Training 
    logger.info("***** Running training *****")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Resume from checkpoint
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            if args.use_ema:
                model_ema.load_pretrained(os.path.join(args.output_dir, path), filename="unet")
            
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Progress bar 
    progress_bar = tqdm(range(0, args.max_train_steps), initial=initial_global_step,
        desc="Steps", disable=not accelerator.is_local_main_process,)
    
    # Training loop
    logger.info("******  Starting training  ******")

    for epoch in range(first_epoch, args.num_train_epochs):
        print(f"[DEBUG] Starting epoch {epoch+1}/{args.num_train_epochs}")
        model.train()

        for step, batch in enumerate(train_dataloader):
            print(f"[DEBUG] Epoch {epoch+1}, Step {step+1}")
            models_to_accumulate = [model]

            with accelerator.accumulate(models_to_accumulate):
                batch_dict = batch
                x_0 = batch_dict[0].to(accelerator.device, dtype=weight_dtype)
                x_1 = batch_dict[1].to(accelerator.device, dtype=weight_dtype)
                print(f"[DEBUG] Batch shapes: x_0 {x_0.shape}, x_1 {x_1.shape}")

                t = rectified_flow.sample_train_time(x_1.shape[0])
                print(f"[DEBUG] Sampled train time t: {t}")

                loss = rectified_flow.get_loss(x_0=x_0, x_1=x_1, t=t)
                print(f"[DEBUG] Computed loss: {loss.item()}")

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if args.use_ema:
                    model_ema.update(global_step*total_batch_size, total_batch_size)
                
            global_step += 1   

            # Save checkpoint and manage checkpoint retention
            if accelerator.is_main_process:
                if global_step % args.checkpointing_steps == 0:
                    # Remove old checkpoints if exceeding the total limit
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = (len(checkpoints) - args.checkpoints_total_limit + 1)
                            removing_checkpoints = checkpoints[0:num_to_remove]
                            logger.info(
                                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                            )
                            logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")
                            for removing_checkpoint in removing_checkpoints:
                                removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                shutil.rmtree(removing_checkpoint)

                    save_path = os.path.join(args.run_output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                    if args.use_ema:
                        model_ema.save_pretrained(save_path, filename="unet")
                        logger.info(f"Saved EMA model to {save_path}")

            # Update progress bar
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            progress_bar.update(1)

            if global_step >= args.max_train_steps:
                print("[DEBUG] Reached max_train_steps, breaking training loop.")
                break

        if accelerator.is_main_process:
            if epoch % args.validation_epochs == 0:
                print(f"[DEBUG] Validation would run at epoch {epoch+1}")
                pass

    # Final save and cleanup
    if accelerator.is_main_process:
        if args.checkpointing_steps > 0:
            last_checkpoint_step = (global_step // args.checkpointing_steps) * args.checkpointing_steps
            if global_step != last_checkpoint_step:
                save_path = os.path.join(args.run_output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved final state to {save_path}")
                if args.use_ema:
                    model_ema.save_pretrained(save_path, filename="unet")
                    logger.info(f"Saved final EMA model to {save_path}")

    accelerator.end_training()

if __name__ == "__main__":
    args = parse_args()
    main(args)
