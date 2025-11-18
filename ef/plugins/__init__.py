"""
ef plugins - Extend ef with additional components.

This package provides plugin system for ef:
- simple: Built-in toy implementations (works out-of-the-box)
- imbed: Bridge to production imbed package (requires pip install imbed)
- segmenter_registry: External segmenter registration (LangChain, spaCy, NLTK, etc.)

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
    >>>
    >>> # Or register external segmenters (conditional on package availability):
    >>> from ef.plugins import segmenter_registry
    >>> segmenter_registry.register_all_segmenters(project)
"""

# Import plugin modules
from ef.plugins import simple_plugin as simple
from ef.plugins import imbed_plugin as imbed
from ef.plugins import segmenter_registry

__all__ = ['simple', 'imbed', 'segmenter_registry']
