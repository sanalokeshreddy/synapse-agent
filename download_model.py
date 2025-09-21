# download_model.py (Simplified for a single, required model)

import os
from huggingface_hub import hf_hub_download

# --- Define the single, required model for the project ---
MODEL_REPO_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
MODEL_FILENAME = "mistral-7b-instruct-v0.2.Q2_K.gguf"
MODEL_DESCRIPTION = "Mistral-7B Instruct v0.2 (Q2_K, ~2.37GB)"

def download_model_if_needed():
    """
    Checks if the required model exists locally and downloads it if not.
    """
    if os.path.exists(MODEL_FILENAME):
        print(f"✅ Model already exists: {MODEL_FILENAME}")
        return

    print(f"Model not found. Downloading '{MODEL_DESCRIPTION}'...")
    print(f"Repo: {MODEL_REPO_ID}")
    print(f"File: {MODEL_FILENAME}")
    
    try:
        hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            local_dir="." # Download to the current directory
        )
        print(f"✅ Download complete: {MODEL_FILENAME}")
        print("You are all set to run the application.")

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("Please check your internet connection and try again.")
        print(f"You can also manually download the file from Hugging Face and place it in this directory.")

if __name__ == "__main__":
    download_model_if_needed()
