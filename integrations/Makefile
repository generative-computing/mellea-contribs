# Makefile for Mellea Integration Tests

.PHONY: help install-deps test test-all test-dspy test-crewai test-langchain list clean

help:
	@echo "Mellea Integration Tests - Available Commands:"
	@echo ""
	@echo "  make install-deps     - Install all integration test dependencies"
	@echo "  make test-all         - Run all integration tests"
	@echo "  make test-dspy        - Run DSPy integration tests"
	@echo "  make test-crewai      - Run CrewAI integration tests"
	@echo "  make test-langchain   - Run LangChain integration tests"
	@echo "  make list             - List available frameworks"
	@echo "  make clean            - Clean test artifacts"
	@echo ""
	@echo "Advanced:"
	@echo "  make test ARGS='...'  - Run with custom pytest arguments"
	@echo ""

install-deps:
	@echo "Installing integration test dependencies..."
	pip install -r requirements-integration-tests.txt

test-all:
	python3 run_integration_tests.py

test-dspy:
	python3 run_integration_tests.py --frameworks dspy

test-crewai:
	python3 run_integration_tests.py --frameworks crewai

test-langchain:
	python3 run_integration_tests.py --frameworks langchain

test:
	python3 run_integration_tests.py $(ARGS)

list:
	python3 run_integration_tests.py --list

clean:
	@echo "Cleaning test artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Clean complete!"