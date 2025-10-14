import subprocess
import os

# --- 1. CONFIGURE YOUR FOLDERS HERE ---
pdf_folder = "documents"      # Changed to relative path
output_folder = "output"        # Changed to relative path

# -----------------------------------------

print(f"Starting batch processing for PDFs in: '{pdf_folder}'")
os.makedirs(output_folder, exist_ok=True) # Ensure output folder exists

for filename in os.listdir(pdf_folder):
    if filename.lower().endswith(".pdf"):
        input_path = os.path.join(pdf_folder, filename)
        
        command = [
            "marker_single",
            input_path,
            "--output_dir", output_folder,
            "--use_llm",
            "--gemini_model_name", "models/gemini-2.5-flash",
            "--timeout", "300",
            "--max_retries", "3"
        ]
        
        print("-" * 50)
        print(f"Processing {filename}...")
        
        try:
            subprocess.run(command, check=True)
            print(f"Successfully processed {filename}.")
        except subprocess.CalledProcessError as e:
            print(f"ERROR: Failed to process {filename}.")
            print(f"Return Code: {e.returncode}")
            print(f"Output: {e.output}")

print("-" * 50)
print("Batch processing complete.")