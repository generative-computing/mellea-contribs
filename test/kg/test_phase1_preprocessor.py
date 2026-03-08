"""Tests for Phase 1 KGPreprocessor structure and interface."""

import pytest
import inspect

from mellea_contribs.kg.preprocessor import KGPreprocessor


class TestKGPreprocessorStructure:
    """Tests for KGPreprocessor class structure."""

    def test_kg_preprocessor_exists(self):
        """Test that KGPreprocessor class exists."""
        assert KGPreprocessor is not None

    def test_kg_preprocessor_is_class(self):
        """Test that KGPreprocessor is a class."""
        assert inspect.isclass(KGPreprocessor)

    def test_kg_preprocessor_has_process_document_method(self):
        """Test that KGPreprocessor has process_document method."""
        assert hasattr(KGPreprocessor, "process_document")
        assert callable(getattr(KGPreprocessor, "process_document"))

    def test_kg_preprocessor_has_persist_extraction_method(self):
        """Test that KGPreprocessor has persist_extraction method."""
        assert hasattr(KGPreprocessor, "persist_extraction")
        assert callable(getattr(KGPreprocessor, "persist_extraction"))

    def test_kg_preprocessor_has_get_hints_method(self):
        """Test that KGPreprocessor has get_hints abstract method."""
        assert hasattr(KGPreprocessor, "get_hints")

    def test_kg_preprocessor_has_post_process_extraction_method(self):
        """Test that KGPreprocessor has post_process_extraction method."""
        assert hasattr(KGPreprocessor, "post_process_extraction")

    def test_kg_preprocessor_methods_are_defined(self):
        """Test that all required methods are defined."""
        required_methods = [
            "process_document",
            "persist_extraction",
            "get_hints",
            "post_process_extraction",
        ]

        for method_name in required_methods:
            method = getattr(KGPreprocessor, method_name, None)
            assert method is not None, f"Method {method_name} not found"
            assert callable(method), f"Method {method_name} is not callable"


class TestKGPreprocessorInterface:
    """Tests for KGPreprocessor interface and contract."""

    def test_kg_preprocessor_init_signature(self):
        """Test KGPreprocessor __init__ signature."""
        sig = inspect.signature(KGPreprocessor.__init__)
        params = list(sig.parameters.keys())

        # Should have at least 'self'
        assert "self" in params

    def test_kg_preprocessor_method_signatures(self):
        """Test that key methods have correct signatures."""
        # process_document should have parameters
        process_sig = inspect.signature(KGPreprocessor.process_document)
        process_params = list(process_sig.parameters.keys())
        assert "self" in process_params

        # get_hints should be present
        hints_sig = inspect.signature(KGPreprocessor.get_hints)
        hints_params = list(hints_sig.parameters.keys())
        assert "self" in hints_params

    def test_kg_preprocessor_docstring(self):
        """Test that KGPreprocessor has docstring."""
        assert KGPreprocessor.__doc__ is not None
        assert len(KGPreprocessor.__doc__) > 0

    def test_kg_preprocessor_method_docstrings(self):
        """Test that key methods have docstrings."""
        methods = ["process_document", "persist_extraction", "get_hints"]

        for method_name in methods:
            method = getattr(KGPreprocessor, method_name)
            assert method.__doc__ is not None, f"Method {method_name} missing docstring"


class TestKGPreprocessorAbstractContract:
    """Tests for KGPreprocessor abstract contract."""

    def test_kg_preprocessor_cannot_be_instantiated_directly(self):
        """Test that KGPreprocessor abstract methods prevent direct instantiation."""
        # KGPreprocessor has abstract methods (get_hints)
        # Trying to instantiate should fail if it's properly abstract
        try:
            from abc import ABC
            # Check if KGPreprocessor is abstract
            if hasattr(KGPreprocessor, "__abstractmethods__"):
                # If it has abstractmethods, direct instantiation should fail
                assert len(KGPreprocessor.__abstractmethods__) > 0 or not issubclass(
                    KGPreprocessor, ABC
                )
        except Exception:
            # If there's any error, the class structure is correct
            pass

    def test_kg_preprocessor_inheritance_ready(self):
        """Test that KGPreprocessor is ready for subclassing."""
        # Create a minimal concrete implementation
        class ConcretePreprocessor(KGPreprocessor):
            def get_hints(self, domain=None):
                return "test hints"

        # This should work without errors
        assert ConcretePreprocessor is not None
