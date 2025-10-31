"""
ef plugins - Extend ef with additional components.

This package provides plugin system for ef:
- simple: Built-in toy implementations (works out-of-the-box)
- imbed: Bridge to production imbed package (requires pip install imbed)

Usage:
    >>> from ef import Project
    >>> from ef.plugins import simple
    >>>
    >>> project = Project.create('test')
    >>> simple.register_simple_components(project)
    >>>
    >>> # Or with imbed (if installed):
    >>> # from ef.plugins import imbed
    >>> # imbed.register(project)
"""

# Import plugin modules
from ef.plugins import simple_plugin as simple
from ef.plugins import imbed_plugin as imbed

__all__ = ['simple', 'imbed']
