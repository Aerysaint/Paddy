import os
import subprocess

# These are the paths inside the Docker container
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

# Get a list of all files in the input directory
try:
    pdf_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith('.pdf')]
except FileNotFoundError:
    print(f"Error: Input directory '{INPUT_DIR}' not found.")
    exit(1)

# Process each PDF file found
for pdf_file in pdf_files:
    input_path = os.path.join(INPUT_DIR, pdf_file)
    
    # Create the output filename by changing the extension from .pdf to .json
    output_filename = os.path.splitext(pdf_file)[0] + ".json"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    print(f"Processing {input_path} -> {output_path}")
    
    # This runs the command you are used to, but with the correct paths
    command = [
        "python",
        "./heading_extractor.py",
        input_path,
        "-o",
        output_path
    ]
    
    # Execute the command
    subprocess.run(command)

print("Processing complete.")