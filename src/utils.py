"""Small filesystem helpers shared across the project."""

import os


def ensure_file(path, description="file"):
    """Raise a clear error if a required file is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{description} not found at: {path}")


def ensure_dir(path):
    """Create a directory (and parents) if it does not already exist."""
    os.makedirs(path, exist_ok=True)
