"""
ef plugins - Extend ef with additional components.

This package provides plugin system for ef with comprehensive segmentation features:

**Core Plugins:**
- simple: Built-in toy implementations (works out-of-the-box)
- imbed: Bridge to production imbed package (requires pip install imbed)
- segmenter_registry: External segmenter registration (LangChain, spaCy, NLTK, etc.)
- advanced_segmenters: Advanced segmentation strategies
- segmenter_utils: Segmentation utilities

**Advanced Features (15 new modules):**
1. segmenter_composition: Chain and combine segmenters
2. segmenter_quality: Quality scoring and optimization
3. segmenter_streaming: Streaming/incremental segmentation
4. segmenter_config: Configuration management
5. segmenter_viz: Visualization tools
6. adaptive_segmentation: Adaptive/smart segmentation
7. multilingual_segmenters: Multilingual support
8. domain_segmenters: Domain-specific segmenters (legal, scientific, medical, code)
9. embedding_optimized: Embedding-aware segmentation
10. ab_testing: A/B testing framework
11. profiler: Performance profiling
12. marketplace: Plugin marketplace
13. cli_tools: CLI tools
14. vector_db_integration: Vector database integration
15. jupyter_widgets: Jupyter notebook integration

Usage:
    >>> from ef import Project
    >>> from ef.plugins import simple
    >>>
    >>> # Quick start with auto-registration
    >>> project = Project.create('test', auto_register_segmenters=True)
    >>>
    >>> # Or manual registration:
    >>> project = Project.create('test')
    >>> simple.register_simple_components(project)
    >>>
    >>> # Advanced features:
    >>> from ef.plugins import segmenter_composition, segmenter_quality
    >>> composed = segmenter_composition.compose_segmenters(project, ['paragraphs', 'sentences'])
    >>> quality = segmenter_quality.evaluate_quality(segments)
"""

# Core plugins
from ef.plugins import simple_plugin as simple
from ef.plugins import imbed_plugin as imbed
from ef.plugins import segmenter_registry
from ef.plugins import advanced_segmenters
from ef.plugins import segmenter_utils

# Advanced features (15 modules)
from ef.plugins import segmenter_composition
from ef.plugins import segmenter_quality
from ef.plugins import segmenter_streaming
from ef.plugins import segmenter_config
from ef.plugins import segmenter_viz
from ef.plugins import adaptive_segmentation
from ef.plugins import multilingual_segmenters
from ef.plugins import domain_segmenters
from ef.plugins import embedding_optimized
from ef.plugins import ab_testing
from ef.plugins import profiler
from ef.plugins import marketplace
from ef.plugins import cli_tools
from ef.plugins import vector_db_integration
from ef.plugins import jupyter_widgets

__all__ = [
    # Core plugins
    'simple',
    'imbed',
    'segmenter_registry',
    'advanced_segmenters',
    'segmenter_utils',
    # Advanced features
    'segmenter_composition',
    'segmenter_quality',
    'segmenter_streaming',
    'segmenter_config',
    'segmenter_viz',
    'adaptive_segmentation',
    'multilingual_segmenters',
    'domain_segmenters',
    'embedding_optimized',
    'ab_testing',
    'profiler',
    'marketplace',
    'cli_tools',
    'vector_db_integration',
    'jupyter_widgets',
]
