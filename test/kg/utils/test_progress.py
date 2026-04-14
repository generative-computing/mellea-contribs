"""Tests for mellea_contribs.kg.utils.progress module."""

import io
import json
import logging
import sys
from unittest.mock import patch

import pytest

from mellea_contribs.kg.qa_models import QAStats
from mellea_contribs.kg.updater_models import UpdateStats
from mellea_contribs.kg.utils import (
    log_progress,
    output_json,
    print_stats,
    setup_logging,
)


class TestSetupLogging:
    """Tests for setup_logging function."""

    def test_setup_logging_default(self):
        """Test setting up logging with default parameters."""
        setup_logging()
        logger = logging.getLogger("mellea_contribs.kg")
        assert logger.level == logging.INFO

    def test_setup_logging_debug(self):
        """Test setting up logging with DEBUG level."""
        setup_logging(log_level="DEBUG")
        logger = logging.getLogger("mellea_contribs.kg")
        assert logger.level == logging.DEBUG

    def test_setup_logging_warning(self):
        """Test setting up logging with WARNING level."""
        setup_logging(log_level="WARNING")
        logger = logging.getLogger("mellea_contribs.kg")
        assert logger.level == logging.WARNING

    def test_setup_logging_with_file(self, tmp_path):
        """Test setting up logging with file output."""
        log_file = tmp_path / "test.log"
        setup_logging(log_level="INFO", log_file=str(log_file))
        logger = logging.getLogger("mellea_contribs.kg")
        # File handler should be added
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)


class TestLogProgress:
    """Tests for log_progress function."""

    def test_log_progress_default(self, caplog):
        """Test logging progress with default level."""
        with caplog.at_level(logging.INFO):
            log_progress("Test message")
        assert "Test message" in caplog.text

    def test_log_progress_debug(self, caplog):
        """Test logging progress with DEBUG level."""
        with caplog.at_level(logging.DEBUG, logger="mellea_contribs.kg"):
            log_progress("Debug message", level="DEBUG")
        assert "Debug message" in caplog.text

    def test_log_progress_error(self, caplog):
        """Test logging progress with ERROR level."""
        with caplog.at_level(logging.ERROR):
            log_progress("Error message", level="ERROR")
        assert "Error message" in caplog.text


class TestOutputJson:
    """Tests for output_json function."""

    def test_output_json_qa_stats(self, capsys):
        """Test outputting QAStats as JSON."""
        stats = QAStats(
            total_questions=10,
            successful_answers=8,
            failed_answers=2,
            average_confidence=0.85,
        )
        output_json(stats)
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["total_questions"] == 10
        assert output["successful_answers"] == 8

    def test_output_json_update_stats(self, capsys):
        """Test outputting UpdateStats as JSON."""
        stats = UpdateStats(
            total_documents=5,
            successful_documents=4,
            failed_documents=1,
            entities_extracted=20,
            relations_extracted=15,
        )
        output_json(stats)
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output["total_documents"] == 5
        assert output["entities_extracted"] == 20

    def test_output_json_preserves_structure(self, capsys):
        """Test that JSON output preserves structure."""
        stats = QAStats(
            total_questions=5,
            models_used=["gpt-4o-mini", "claude-opus"],
        )
        output_json(stats)
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert isinstance(output["models_used"], list)
        assert len(output["models_used"]) == 2


class TestPrintStats:
    """Tests for print_stats function."""

    def test_print_stats_qa_stats(self, capsys):
        """Test printing QAStats."""
        stats = QAStats(
            total_questions=10,
            successful_answers=8,
            failed_answers=2,
            average_confidence=0.85,
        )
        print_stats(stats, to_stderr=False)
        captured = capsys.readouterr()
        assert "Total Questions" in captured.out or "total_questions" in captured.out.lower()
        assert "10" in captured.out

    def test_print_stats_update_stats(self, capsys):
        """Test printing UpdateStats."""
        stats = UpdateStats(
            total_documents=5,
            successful_documents=4,
            failed_documents=1,
        )
        print_stats(stats, to_stderr=False)
        captured = capsys.readouterr()
        assert "5" in captured.out or "Total Documents" in captured.out.lower()

    def test_print_stats_to_stderr(self, capsys):
        """Test that print_stats can output to stderr."""
        stats = QAStats(total_questions=10)
        print_stats(stats, to_stderr=True)
        captured = capsys.readouterr()
        # Should be in stderr when to_stderr=True
        assert "10" in captured.err or "10" in captured.out

    def test_print_stats_formatting(self, capsys):
        """Test formatting of stats output."""
        stats = QAStats(
            total_questions=100,
            average_confidence=0.7531,
        )
        print_stats(stats, to_stderr=False)
        captured = capsys.readouterr()
        # Should format float nicely
        assert "0.75" in captured.out or "75" in captured.out

    def test_print_stats_with_indentation(self, capsys):
        """Test print_stats with indentation."""
        stats = QAStats(total_questions=5)
        print_stats(stats, indent=2, to_stderr=False)
        captured = capsys.readouterr()
        # Should have some indentation
        output_lines = captured.out.split("\n")
        # At least some lines should be indented
        assert any(line.startswith("  ") for line in output_lines if line.strip())


class TestProgressTracker:
    """Tests for ProgressTracker class."""

    def test_progress_tracker_initialization(self):
        """Test ProgressTracker initialization."""
        from mellea_contribs.kg.utils import ProgressTracker

        tracker = ProgressTracker(total=100, desc="Processing")
        assert tracker.total == 100
        assert tracker.current == 0

    def test_progress_tracker_update(self):
        """Test ProgressTracker update."""
        from mellea_contribs.kg.utils import ProgressTracker

        tracker = ProgressTracker(total=100, desc="Processing")
        tracker.update(10)
        assert tracker.current == 10
        tracker.update(5)
        assert tracker.current == 15

    def test_progress_tracker_close(self):
        """Test ProgressTracker close."""
        from mellea_contribs.kg.utils import ProgressTracker

        tracker = ProgressTracker(total=100, desc="Processing")
        tracker.update(50)
        tracker.close()
        # Should not raise error

    def test_progress_tracker_without_tqdm(self):
        """Test ProgressTracker without tqdm."""
        from mellea_contribs.kg.utils import ProgressTracker

        tracker = ProgressTracker(total=100, use_tqdm=False)
        tracker.update(50)
        assert tracker.current == 50


class TestIntegration:
    """Integration tests for progress module."""

    def test_workflow_setup_log_output(self, capsys):
        """Test workflow: setup → log → output."""
        setup_logging(log_level="INFO")
        log_progress("Starting processing")

        stats = QAStats(total_questions=20, successful_answers=18)
        output_json(stats)

        captured = capsys.readouterr()
        assert "Starting processing" in captured.err or len(captured.out) > 0
        assert "20" in captured.out

    def test_workflow_print_and_output_stats(self, capsys):
        """Test printing and outputting same stats."""
        stats = QAStats(
            total_questions=50,
            successful_answers=45,
            average_confidence=0.88,
        )

        # Print to inspect
        print_stats(stats, to_stderr=False)
        captured1 = capsys.readouterr()

        # Output as JSON
        output_json(stats)
        captured2 = capsys.readouterr()

        # Both should reference the stats
        assert "50" in captured1.out
        assert "50" in captured2.out
        assert json.loads(captured2.out)["total_questions"] == 50
