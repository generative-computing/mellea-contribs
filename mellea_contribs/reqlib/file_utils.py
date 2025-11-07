"""File I/O utilities for auto-fixing Python Requirements.

This module provides file type predicates and I/O functions for
creating dummy files when auto-fixing missing file dependencies.
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import imageio.v3 as imageio
except ImportError:
    imageio = None

from .data_generators import lorem_paragraph


def is_table(path: str) -> bool:
    """Check if file is a table format (CSV, TSV, XLSX, JSON)."""
    ext = Path(path).suffix.lower()
    return ext in {".csv", ".tsv", ".xlsx", ".json"}


def is_image(path: str) -> bool:
    """Check if file is an image format (PNG, JPEG, TIFF, GIF)."""
    ext = Path(path).suffix.lower()
    return ext in {".png", ".jpeg", ".jpg", ".tiff", ".gif"}


def is_audio(path: str) -> bool:
    """Check if file is an audio format (WAV, MP3, MP4, OGG)."""
    ext = Path(path).suffix.lower()
    return ext in {".wav", ".mp3", ".mp4", ".ogg"}


def is_structured(path: str) -> bool:
    """Check if file is a structured format (XML, HTML, JSON, YAML)."""
    ext = Path(path).suffix.lower()
    return ext in {".xml", ".html", ".json", ".yaml"}


def read_table(path: str) -> Optional[object]:
    """Read table file into DataFrame if pandas available."""
    if pd is None:
        return None

    ext = Path(path).suffix.lower()
    try:
        if ext == ".csv":
            return pd.read_csv(path)
        elif ext == ".tsv":
            return pd.read_csv(path, sep="\t")
        elif ext == ".xlsx":
            return pd.read_excel(path)
        elif ext == ".json":
            return pd.read_json(path)
    except Exception:
        return None
    return None


def write_table(path: str, df: object) -> bool:
    """Write DataFrame to table file if pandas available."""
    if pd is None or df is None:
        return False

    ext = Path(path).suffix.lower()
    try:
        if ext == ".csv":
            df.to_csv(path, index=False)
        elif ext == ".tsv":
            df.to_csv(path, index=False, sep="\t")
        elif ext == ".xlsx":
            df.to_excel(path, index=False)
        elif ext == ".json":
            df.to_json(path)
        else:
            return False
        return True
    except Exception:
        return False


def create_dummy_table(path: str, num_rows: int = 5) -> bool:
    """Create dummy table file with basic structure."""
    if pd is None:
        return False

    try:
        # Create basic DataFrame with ID column
        df = pd.DataFrame({
            "id": list(range(num_rows))
        })
        return write_table(path, df)
    except Exception:
        return False


def create_dummy_image(path: str, width: int = 100, height: int = 100) -> bool:
    """Create dummy image file (black image)."""
    if imageio is None:
        return False

    try:
        # Create black image
        image = np.zeros((height, width, 3), dtype=np.uint8)
        imageio.imwrite(path, image)
        return True
    except Exception:
        return False


def create_dummy_text(path: str) -> bool:
    """Create dummy text file."""
    try:
        with open(path, "w") as f:
            f.write(lorem_paragraph())
        return True
    except Exception:
        return False


def create_dummy_file(path: str) -> bool:
    """Create appropriate dummy file based on extension."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if is_table(path):
        return create_dummy_table(path)
    elif is_image(path):
        return create_dummy_image(path)
    elif Path(path).suffix.lower() == ".txt":
        return create_dummy_text(path)
    else:
        # Create empty file for unknown types
        try:
            Path(path).touch()
            return True
        except Exception:
            return False


def add_column_to_table(path: str, column_name: str, values: list) -> bool:
    """Add column with values to existing table file."""
    if pd is None:
        return False

    try:
        df = read_table(path)
        if df is None:
            return False

        # Ensure values list matches DataFrame length
        if len(values) != len(df):
            # Repeat or truncate values to match
            if len(values) < len(df):
                values = (values * ((len(df) // len(values)) + 1))[:len(df)]
            else:
                values = values[:len(df)]

        df[column_name] = values
        return write_table(path, df)
    except Exception:
        return False


def get_all_files_by_type(directory: str = "data", predicate_func=None) -> list[str]:
    """Get all files in directory matching predicate.

    Args:
        directory: Directory to scan
        predicate_func: Function to filter files (e.g., is_table)

    Returns:
        List of file paths
    """
    if not os.path.exists(directory):
        return []

    files = []
    try:
        for filename in os.listdir(directory):
            full_path = os.path.join(directory, filename)
            if os.path.isfile(full_path):
                if predicate_func is None or predicate_func(filename):
                    files.append(full_path)
    except (OSError, PermissionError):
        pass

    return files