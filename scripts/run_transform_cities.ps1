
# This script runs the transformation of city images using a pre-trained model.

# .\scripts\run_transform_cities.ps1

python .\scripts\transform_cities.py `
--model_path "./output/TokyoToManhattan" `
--input_dir "./data/cities/Tokyo" `
--output_dir "./data/cities_trasfromed/TokyoToManhattan" `
--resolution 64 `
--device "cuda" `
--num_steps 400 `
--interp "straight" `
--source_distribution "normal" `
--is_independent_coupling