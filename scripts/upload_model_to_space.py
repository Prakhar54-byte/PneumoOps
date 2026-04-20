"""
PneumoOps — Upload Models Directly to Hugging Face Space
=========================================================
Uploads local model files to your Hugging Face Space repo, 
bypassing GitHub since the large model files are ignored in .gitignore.

Usage:
    export HF_SPACE_REPO="your-username/pneumoops-space"
    python scripts/upload_model_to_space.py
"""

import os
from pathlib import Path
from huggingface_hub import HfApi

# ─── Config ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "chestmnist_mobilenetv3"

HF_TOKEN = os.getenv("HF_TOKEN")  # optional if already logged in via CLI
SPACE_REPO = os.getenv("HF_SPACE_REPO", "")

if not SPACE_REPO:
    print("\n⚠️  Please set HF_SPACE_REPO environment variable, e.g.:")
    print('   export HF_SPACE_REPO="your-hf-username/pneumoops-space"')
    raise SystemExit(1)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    api = HfApi(token=HF_TOKEN)

    if not MODEL_DIR.exists():
        print(f"\n❌ Error: Model directory {MODEL_DIR} not found.")
        print("Please ensure you have trained or downloaded the models first.")
        raise SystemExit(1)

    print(f"\n📦 Verifying space repository: {SPACE_REPO}")
    # Ensure the space exists (this will create it if it doesn't, but typically you already have one)
    api.create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="docker",
        exist_ok=True,
        token=HF_TOKEN,
    )

    print(f"\n⬆️  Uploading models to https://huggingface.co/spaces/{SPACE_REPO}")
    
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        path_in_repo="models/chestmnist_mobilenetv3",
        repo_id=SPACE_REPO,
        repo_type="space",
        commit_message="Upload models to Space",
        token=HF_TOKEN,
    )

    print(f"\n✅ All models uploaded successfully!")
    print(f"   View your space at: https://huggingface.co/spaces/{SPACE_REPO}")

if __name__ == "__main__":
    main()
