import argparse
import os
from pathlib import Path
import torch
from rectified_flow.models.unet import SongUNet, SongUNetConfig
from rectified_flow.RFlow import RectifiedFlow
from rectified_flow.samplers.euler_sampler import EulerSampler
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Inference script for city image transformation using Rectified Flow.")

    parser.add_argument("--model_path", type=str, default=None, help="Path to the trained model weights (e.g., unet_ema.pt). Optional if output_dir is provided.")
    parser.add_argument("--input_dir", type=str, default="./data/input_cities", help="Directory containing input images (city A).")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to load models from if model_path not given.")
    parser.add_argument("--resolution", type=int, default=128, help="Resolution to resize input images to before inference.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for inference ('cuda' or 'cpu').")
    parser.add_argument("--num_steps", type=int, default=100, help="Number of steps for the solver.")
    parser.add_argument("--interp", type=str, default="straight", choices=["straight", "slerp", "ddim"], help="Interpolation method used during training. Must match training configuration.")
    parser.add_argument("--source_distribution", type=str, default="normal", help="Distribution used for source samples during training.")
    parser.add_argument("--is_independent_coupling", type=bool, default=True, help="Whether independent coupling was used in training.")
    parser.add_argument("--use_ema", type=bool, default=True, help="Load the EMA version of the model if available.")
    return parser.parse_args()


def get_latest_checkpoint(output_dir):
    """Returns the path to the latest checkpoint directory in output_dir."""
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory '{output_dir}' does not exist.")

    all_checkpoints = [
        d for d in os.listdir(output_dir) if d.startswith("checkpoint-")
    ]

    if not all_checkpoints:
        raise FileNotFoundError(f"No checkpoints found in '{output_dir}'.")

    def extract_step(checkpoint_name):
        return int(checkpoint_name.split("-")[1])

    latest_checkpoint = max(all_checkpoints, key=extract_step)
    return os.path.join(output_dir, latest_checkpoint)


# Load Pretrained Model
def load_model(model_path, resolution, device="cuda", use_ema=True):
    """
    Loads either EMA or regular model from the given checkpoint directory.
    """
    print("[INFO] Searching for latest checkpoint...")
    model_dir = get_latest_checkpoint(model_path)
    model_dir = os.path.normpath(model_dir)

    print(f"[INFO] Using model from: {model_dir}")

    ema_model_path = os.path.join(model_dir, "unet_ema.pt")
    reg_model_path = os.path.join(model_dir, "unet_model.pt")

    state_dict = None 
    loaded_ema = False

    if use_ema:
        if os.path.isfile(ema_model_path):
            state_dict = torch.load(ema_model_path, map_location=device)
            loaded_ema = True
            print("[INFO] Loading EMA model weights")
        else:
            print("[WARNING] EMA model not found. Falling back to standard model.")

    if state_dict is None:
        if os.path.isfile(reg_model_path):
            state_dict = torch.load(reg_model_path, map_location=device)
            print("[INFO] Loading standard model weights")
        else:
            raise FileNotFoundError(f"No valid model weights found in {model_dir}")

    config = SongUNetConfig(
        img_resolution=resolution,
        in_channels=3,
        out_channels=3,
    )
    model = SongUNet(config).to(device)
    model.eval()
    model.load_state_dict(state_dict, strict=True)

    print(f"[INFO] Loaded {'EMA' if loaded_ema else 'regular'} model.")
    return model


def transform_image_with_euler(rectified_flow, image_tensor, num_steps, x_cond=None):
  with torch.no_grad():
        sampler = EulerSampler(rectified_flow=rectified_flow, num_steps=num_steps)

        # Manually set required fields
        sampler.x_t = image_tensor  # Set input image
        sampler.t_index = 0         # Start from first step

        # Set initial t and t_next
        sampler.t = sampler.time_grid[sampler.t_index]
        sampler.t_next = sampler.time_grid[sampler.t_index + 1]

        # Create time grid if not already set
        if not hasattr(sampler, "time_grid"):
            sampler.time_grid = torch.linspace(0, 1, num_steps + 1).to(image_tensor.device)

        # Run the sampling loop manually
        for i in range(num_steps):
            sampler.step()  # Perform one step
            sampler.t_index += 1
            if sampler.t_index < num_steps:
                sampler.t = sampler.time_grid[sampler.t_index]
                sampler.t_next = sampler.time_grid[sampler.t_index + 1]
  
        # Get the final result
        transformed_tensor = sampler.x_t  # or however final tensor is stored
        return transformed_tensor

# Main Function
def main(args):
    device = torch.device(args.device)
    print("Using GPU:", torch.cuda.is_available())

    # Load model
    model = load_model(args.model_path, resolution=args.resolution, device=device)

    # Initialize RectifiedFlow
    rectified_flow = RectifiedFlow(
        data_shape=(3, args.resolution, args.resolution),
        velocity_field=model,
        interp=args.interp,
        source_distribution=args.source_distribution,
        is_independent_coupling=args.is_independent_coupling,
        device=device,
        dtype=torch.float32,
    )

    # Create output subdirectory based on model name
    training_folder = os.path.basename(args.model_path)
    transformation_folder = f"{training_folder}_transformation"
    transformation_dir = os.path.join(args.output_dir, transformation_folder)
    os.makedirs(transformation_dir, exist_ok=True)
    print(f"[INFO] Saving transformed images to: {transformation_dir}")

    # Define transforms
    transform_input = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    transform_output = transforms.ToPILImage()

    def denormalize(x):
        return (x * 0.5 + 0.5).clamp(0, 1)

    # Process each image
    input_paths = sorted([os.path.join(args.input_dir, fname) for fname in os.listdir(args.input_dir)
                          if fname.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    total_images = len(input_paths)
    print(f"[INFO] Found {total_images} images to process")

    # Add progress bar for image processing
    progress_bar = tqdm(total=total_images, desc="Processing Images", position=0, leave=True)

    for idx, input_path in enumerate(input_paths):
        try:
            # Load and preprocess image
            image_pil = Image.open(input_path).convert("RGB").resize((args.resolution, args.resolution))
            image_tensor = transform_input(image_pil).unsqueeze(0).to(device)  # Add batch dim

            # Transform image
            transformed_tensor = transform_image_with_euler(rectified_flow, image_tensor, args.num_steps,x_cond=image_tensor)

            # Denormalize and save
            transformed_tensor = denormalize(transformed_tensor.squeeze(0).cpu())
            transformed_image = transform_output(transformed_tensor)
            output_path = os.path.join(transformation_dir, os.path.basename(input_path))
            transformed_image.save(output_path)
            
            # Update progress bar
            progress_bar.set_postfix_str(f"Saved: {os.path.basename(input_path)}")
            progress_bar.update()

        except Exception as e:
            tqdm.write(f"[ERROR] Failed processing {input_path}: {str(e)}")
            progress_bar.update()

    progress_bar.close()
    print("[INFO] Transformation complete.")
    

if __name__ == "__main__":
    args = parse_args()
    main(args)