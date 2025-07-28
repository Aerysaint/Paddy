import os
import subprocess

# These are the paths inside the Docker container
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

# Find the first JSON input file in the input directory
input_json_file = None
for filename in os.listdir(INPUT_DIR):
    if filename.lower().endswith('.json'):
        input_json_file = filename
        break

if not input_json_file:
    print(f"Error: No input JSON file found in '{INPUT_DIR}'.")
    exit(1)

# Construct the full paths
input_path = os.path.join(INPUT_DIR, input_json_file)
# The output file will be named according to the challenge requirements.
# Let's call it 'challenge1b_output.json'
output_path = os.path.join(OUTPUT_DIR, "challenge1b_output.json")

print(f"Starting processing for {input_path}")

# This builds the command you are used to, but with the container's paths
# 'python main.py /app/input/challenge1b_input.json --final-results 5 --output-file /app/output/challenge1b_output.json'
command = [
    "python",
    "main.py",
    input_path,
    "--final-results",
    "5",
    "--output-file",
    output_path
]

# Execute the command
subprocess.run(command)

print(f"Processing complete. Output saved to {output_path}")