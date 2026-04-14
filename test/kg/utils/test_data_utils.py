"""Tests for mellea_contribs.kg.utils.data_utils module."""

import json
import random
from pathlib import Path

import pytest

from mellea_contribs.kg.utils import (
    append_jsonl,
    batch_iterator,
    load_jsonl,
    save_jsonl,
    shuffle_jsonl,
    truncate_jsonl,
    validate_jsonl_schema,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for tests."""
    return tmp_path


class TestLoadAndSaveJsonl:
    """Tests for load_jsonl and save_jsonl functions."""

    def test_save_and_load_empty_list(self, temp_dir):
        """Test saving and loading empty JSONL."""
        path = temp_dir / "empty.jsonl"
        data = []
        save_jsonl(data, path)
        loaded = list(load_jsonl(path))
        assert loaded == []

    def test_save_and_load_single_item(self, temp_dir):
        """Test saving and loading single item."""
        path = temp_dir / "single.jsonl"
        data = [{"id": 1, "name": "test"}]
        save_jsonl(data, path)
        loaded = list(load_jsonl(path))
        assert loaded == data

    def test_save_and_load_multiple_items(self, temp_dir):
        """Test saving and loading multiple items."""
        path = temp_dir / "multiple.jsonl"
        data = [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
            {"id": 3, "value": "c"},
        ]
        save_jsonl(data, path)
        loaded = list(load_jsonl(path))
        assert loaded == data

    def test_load_nonexistent_file(self, temp_dir):
        """Test loading nonexistent file raises error."""
        path = temp_dir / "nonexistent.jsonl"
        with pytest.raises(FileNotFoundError):
            list(load_jsonl(path))

    def test_load_invalid_json(self, temp_dir):
        """Test loading invalid JSON raises error."""
        path = temp_dir / "invalid.jsonl"
        path.write_text("not valid json\n")
        with pytest.raises(json.JSONDecodeError):
            list(load_jsonl(path))

    def test_save_creates_parent_directory(self, temp_dir):
        """Test that save_jsonl creates parent directories."""
        path = temp_dir / "subdir" / "nested" / "file.jsonl"
        data = [{"test": 1}]
        save_jsonl(data, path)
        assert path.exists()
        loaded = list(load_jsonl(path))
        assert loaded == data


class TestAppendJsonl:
    """Tests for append_jsonl function."""

    def test_append_to_empty_file(self, temp_dir):
        """Test appending to empty file."""
        path = temp_dir / "append.jsonl"
        append_jsonl({"id": 1}, path)
        loaded = list(load_jsonl(path))
        assert loaded == [{"id": 1}]

    def test_append_to_existing_file(self, temp_dir):
        """Test appending to existing file."""
        path = temp_dir / "append.jsonl"
        save_jsonl([{"id": 1}], path)
        append_jsonl({"id": 2}, path)
        loaded = list(load_jsonl(path))
        assert loaded == [{"id": 1}, {"id": 2}]

    def test_append_multiple_items(self, temp_dir):
        """Test appending multiple items."""
        path = temp_dir / "append.jsonl"
        for i in range(5):
            append_jsonl({"id": i}, path)
        loaded = list(load_jsonl(path))
        assert len(loaded) == 5
        assert loaded[0]["id"] == 0
        assert loaded[4]["id"] == 4


class TestBatchIterator:
    """Tests for batch_iterator function."""

    def test_batch_exact_division(self):
        """Test batch iterator with exact division."""
        items = list(range(9))
        batches = list(batch_iterator(items, 3))
        assert len(batches) == 3
        assert batches[0] == [0, 1, 2]
        assert batches[1] == [3, 4, 5]
        assert batches[2] == [6, 7, 8]

    def test_batch_uneven_division(self):
        """Test batch iterator with uneven division."""
        items = list(range(10))
        batches = list(batch_iterator(items, 3))
        assert len(batches) == 4
        assert batches[0] == [0, 1, 2]
        assert batches[3] == [9]

    def test_batch_single_item(self):
        """Test batch iterator with batch size 1."""
        items = [1, 2, 3]
        batches = list(batch_iterator(items, 1))
        assert len(batches) == 3
        assert all(len(b) == 1 for b in batches)

    def test_batch_empty_list(self):
        """Test batch iterator with empty list."""
        batches = list(batch_iterator([], 3))
        assert batches == []

    def test_batch_size_larger_than_list(self):
        """Test batch iterator with batch size larger than list."""
        items = [1, 2, 3]
        batches = list(batch_iterator(items, 10))
        assert len(batches) == 1
        assert batches[0] == [1, 2, 3]


class TestTruncateJsonl:
    """Tests for truncate_jsonl function."""

    def test_truncate_larger_than_file(self, temp_dir):
        """Test truncating with limit larger than file."""
        input_path = temp_dir / "input.jsonl"
        output_path = temp_dir / "output.jsonl"
        data = [{"id": i} for i in range(5)]
        save_jsonl(data, input_path)

        truncate_jsonl(input_path, output_path, 10)
        loaded = list(load_jsonl(output_path))
        assert loaded == data

    def test_truncate_smaller_than_file(self, temp_dir):
        """Test truncating with limit smaller than file."""
        input_path = temp_dir / "input.jsonl"
        output_path = temp_dir / "output.jsonl"
        data = [{"id": i} for i in range(10)]
        save_jsonl(data, input_path)

        truncate_jsonl(input_path, output_path, 3)
        loaded = list(load_jsonl(output_path))
        assert len(loaded) == 3
        assert loaded == data[:3]

    def test_truncate_zero(self, temp_dir):
        """Test truncating with limit 0."""
        input_path = temp_dir / "input.jsonl"
        output_path = temp_dir / "output.jsonl"
        data = [{"id": i} for i in range(5)]
        save_jsonl(data, input_path)

        truncate_jsonl(input_path, output_path, 0)
        loaded = list(load_jsonl(output_path))
        assert loaded == []


class TestShuffleJsonl:
    """Tests for shuffle_jsonl function."""

    def test_shuffle_preserves_all_items(self, temp_dir):
        """Test that shuffle preserves all items."""
        input_path = temp_dir / "input.jsonl"
        output_path = temp_dir / "output.jsonl"
        data = [{"id": i} for i in range(20)]
        save_jsonl(data, input_path)

        shuffle_jsonl(input_path, output_path)
        loaded = list(load_jsonl(output_path))
        assert len(loaded) == len(data)
        # Check all items are present (sorted to compare)
        assert sorted(loaded, key=lambda x: x["id"]) == data

    def test_shuffle_changes_order(self, temp_dir):
        """Test that shuffle changes order (probabilistically)."""
        # Note: This test could theoretically fail if shuffle produces same order
        # but probability is extremely low with 20 items
        input_path = temp_dir / "input.jsonl"
        output_path = temp_dir / "output.jsonl"
        data = [{"id": i} for i in range(20)]
        save_jsonl(data, input_path)

        shuffle_jsonl(input_path, output_path)
        loaded = list(load_jsonl(output_path))
        # Extract just the order of IDs
        loaded_ids = [item["id"] for item in loaded]
        original_ids = list(range(20))
        # Very unlikely to get same order
        assert loaded_ids != original_ids


class TestValidateJsonlSchema:
    """Tests for validate_jsonl_schema function."""

    def test_valid_schema(self, temp_dir):
        """Test validation with valid schema."""
        path = temp_dir / "valid.jsonl"
        data = [
            {"id": 1, "name": "a"},
            {"id": 2, "name": "b"},
        ]
        save_jsonl(data, path)

        valid, errors = validate_jsonl_schema(path, ["id", "name"])
        assert valid is True
        assert errors == []

    def test_missing_required_field(self, temp_dir):
        """Test validation with missing required field."""
        path = temp_dir / "invalid.jsonl"
        data = [
            {"id": 1, "name": "a"},
            {"id": 2},  # Missing 'name'
        ]
        save_jsonl(data, path)

        valid, errors = validate_jsonl_schema(path, ["id", "name"])
        assert valid is False
        assert len(errors) > 0
        assert "Line 2" in errors[0]

    def test_empty_file_valid(self, temp_dir):
        """Test validation with empty file."""
        path = temp_dir / "empty.jsonl"
        save_jsonl([], path)

        valid, errors = validate_jsonl_schema(path, ["id", "name"])
        assert valid is True
        assert errors == []

    def test_partial_schema_validation(self, temp_dir):
        """Test validation with partial schema match."""
        path = temp_dir / "partial.jsonl"
        data = [
            {"id": 1, "name": "a", "extra": "field"},
            {"id": 2, "name": "b", "extra": "field"},
        ]
        save_jsonl(data, path)

        valid, errors = validate_jsonl_schema(path, ["id"])
        assert valid is True
        assert errors == []


class TestIntegration:
    """Integration tests for data_utils functions."""

    def test_workflow_create_truncate_load(self, temp_dir):
        """Test workflow: create → truncate → load."""
        input_path = temp_dir / "input.jsonl"
        output_path = temp_dir / "output.jsonl"

        # Create
        data = [{"id": i, "value": f"item_{i}"} for i in range(100)]
        save_jsonl(data, input_path)

        # Truncate
        truncate_jsonl(input_path, output_path, 10)

        # Load and verify
        loaded = list(load_jsonl(output_path))
        assert len(loaded) == 10
        assert loaded[0]["id"] == 0
        assert loaded[9]["id"] == 9

    def test_workflow_batch_processing(self, temp_dir):
        """Test workflow: create → batch processing."""
        path = temp_dir / "data.jsonl"
        data = [{"id": i} for i in range(25)]
        save_jsonl(data, path)

        loaded = list(load_jsonl(path))
        batches = list(batch_iterator(loaded, 5))
        assert len(batches) == 5
        assert all(len(b) == 5 for b in batches)

    def test_workflow_append_shuffle_validate(self, temp_dir):
        """Test workflow: append → shuffle → validate."""
        path = temp_dir / "data.jsonl"

        # Append items
        for i in range(10):
            append_jsonl({"id": i, "status": "active"}, path)

        # Shuffle
        output_path = temp_dir / "shuffled.jsonl"
        shuffle_jsonl(path, output_path)

        # Validate
        valid, errors = validate_jsonl_schema(output_path, ["id", "status"])
        assert valid is True
        assert errors == []
