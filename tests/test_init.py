"""
Tests for FDFI package initialization.
"""

import fdfi


def test_version():
    """Test that version is defined."""
    assert hasattr(fdfi, "__version__")
    assert isinstance(fdfi.__version__, str)
    assert fdfi.__version__ == "0.0.1"


def test_package_imports():
    """Test that package can be imported."""
    # Just check that the package imports without errors
    assert fdfi is not None
