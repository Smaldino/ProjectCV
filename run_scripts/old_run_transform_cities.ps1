
# This script runs the transformation of city images using a pre-trained model.

# Set paths
$model_path = "./output/TokyoToManhattan/training_009"
$input_dir = "./data/cities/Tokyo"
$output_dir = "./data/cities_trasfromed/TokyoToManhattan"

# Run transform_cities.py
# .\scripts\run_transform_cities.ps1

python .\scripts\transform_cities.py `
--model_path "$model_path" `
--input_dir "$input_dir" `
--output_dir "$output_dir" `
--resolution 64 `
--device "cuda" `
--num_steps 400 `
--interp "straight" `
--source_distribution "normal" `
--is_independent_coupling