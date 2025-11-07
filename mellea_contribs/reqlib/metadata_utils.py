"""Metadata utilities for directory structure conversion.

This module provides functions to extract metadata from directories
and recreate directory structures from metadata.
"""

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .file_utils import read_table, is_table, is_image, create_dummy_file, get_all_files_by_type
from .data_generators import get_generator_for_column


def directory_to_metadata(directory: str) -> List[Dict[str, Any]]:
    """Generate metadata for files in the given directory.

    Args:
        directory: Directory path (should not end with "/")

    Returns:
        List of metadata dictionaries for each file
    """
    directory = directory.rstrip("/")
    metadata = []

    for path, subdirs, files in os.walk(directory):
        for name in files:
            full_path = os.path.join(path, name)
            # Convert to data/ relative path
            relative_path = re.sub(f"^{re.escape(directory)}/", "data/", full_path)

            try:
                stat_result = os.stat(full_path)
                file_metadata = {
                    "filename": relative_path,
                    "atime": datetime.fromtimestamp(stat_result.st_atime).isoformat(),
                    "mtime": datetime.fromtimestamp(stat_result.st_mtime).isoformat(),
                    "size": stat_result.st_size,
                }

                # Add format-specific metadata
                if is_table(full_path):
                    df = read_table(full_path)
                    if df is not None:
                        try:
                            file_metadata["column_names"] = list(df.columns)
                            file_metadata["number_of_rows"] = len(df)
                        except Exception:
                            pass

                elif is_image(full_path):
                    try:
                        # Try to get image dimensions
                        import imageio.v3 as imageio
                        image = imageio.imread(full_path)
                        file_metadata["height"] = image.shape[0]
                        file_metadata["width"] = image.shape[1]
                        if len(image.shape) > 2:
                            file_metadata["channels"] = image.shape[2]
                    except Exception:
                        # Fallback dimensions
                        file_metadata["height"] = 100
                        file_metadata["width"] = 100

                metadata.append(file_metadata)

            except (OSError, PermissionError):
                # Skip files we can't access
                continue

    return metadata


def metadata_to_directory(metadata: List[Dict[str, Any]], target_directory: str) -> bool:
    """Create directory structure from metadata.

    Args:
        metadata: List of file metadata dictionaries
        target_directory: Target directory to create (should not end with "/")

    Returns:
        True if successful, False otherwise
    """
    target_directory = target_directory.rstrip("/")
    success = True

    for file_info in metadata:
        if "filename" not in file_info:
            continue

        filename = file_info["filename"]
        # Clean up filename
        if filename.startswith("./"):
            filename = filename[2:]
        if not filename.startswith("data/"):
            filename = os.path.join("data", filename)

        # Convert to target directory path
        target_path = re.sub("^data/", f"{target_directory}/", filename)

        try:
            # Ensure parent directory exists
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            # Create file based on type and metadata
            if is_table(target_path):
                success &= _create_table_from_metadata(target_path, file_info)
            elif is_image(target_path):
                success &= _create_image_from_metadata(target_path, file_info)
            else:
                success &= create_dummy_file(target_path)

            # Set file timestamps if provided
            if "atime" in file_info or "mtime" in file_info:
                _set_file_timestamps(target_path, file_info)

        except Exception:
            success = False
            continue

    return success


def _create_table_from_metadata(path: str, metadata: Dict[str, Any]) -> bool:
    """Create table file from metadata."""
    try:
        import pandas as pd
    except ImportError:
        return create_dummy_file(path)

    if "column_names" not in metadata or "number_of_rows" not in metadata:
        return create_dummy_file(path)

    try:
        # Create DataFrame with specified columns and rows
        df_data = {}
        for column in metadata["column_names"]:
            generator = get_generator_for_column(column)
            df_data[column] = [generator() for _ in range(metadata["number_of_rows"])]

        df = pd.DataFrame(df_data)

        # Write to file
        from .file_utils import write_table
        return write_table(path, df)

    except Exception:
        return create_dummy_file(path)


def _create_image_from_metadata(path: str, metadata: Dict[str, Any]) -> bool:
    """Create image file from metadata."""
    try:
        import numpy as np
        import imageio.v3 as imageio
    except ImportError:
        return create_dummy_file(path)

    try:
        height = metadata.get("height", 100)
        width = metadata.get("width", 100)
        channels = metadata.get("channels", 3)

        # Create image array
        if channels == 1:
            image = np.zeros((height, width), dtype=np.uint8)
        else:
            image = np.zeros((height, width, channels), dtype=np.uint8)

        imageio.imwrite(path, image)
        return True

    except Exception:
        return create_dummy_file(path)


def _set_file_timestamps(path: str, metadata: Dict[str, Any]) -> None:
    """Set file timestamps from metadata."""
    try:
        from subprocess import run
        if "atime" in metadata:
            run(["touch", "-a", "-d", metadata["atime"], path], check=False)
        if "mtime" in metadata:
            run(["touch", "-m", "-d", metadata["mtime"], path], check=False)
    except Exception:
        # Ignore timestamp setting failures
        pass


