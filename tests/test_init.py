"""
<<<<<<< HEAD
Tests for dfi package initialization.
=======

"""

import dfi


def test_version():
    """Test that version is defined."""
    assert hasattr(dfi, "__version__")
    assert isinstance(dfi.__version__, str)
    assert dfi.__version__ == "0.0.1"


def test_package_imports():
    """Test that package can be imported."""
    # Just check that the package imports without errors
    assert dfi is not None
