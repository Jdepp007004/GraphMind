"""
scripts/download_models.py

Downloads Gemma 2B from HuggingFace (requires HF_TOKEN with Gemma access).
"""

import sys
import os
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    """Download Gemma 2B model."""
    token = os.getenv("HF_TOKEN")
    if not token:
        logger.error("HF_TOKEN not set. Export HF_TOKEN=your_token and retry.")
        sys.exit(1)
    os.makedirs(settings.GEMMA_LOCAL_PATH, exist_ok=True)
    logger.info(f"Downloading {settings.GEMMA_MODEL_ID} to {settings.GEMMA_LOCAL_PATH}")
    snapshot_download(
        repo_id=settings.GEMMA_MODEL_ID,
        local_dir=settings.GEMMA_LOCAL_PATH,
        token=token
    )
    logger.info("Gemma 2B download complete.")
