"""
ef (Embedding Flow) - Lightweight framework for embedding pipelines.

This package provides:
- Easy project creation with flexible storage
- Component registries as mapping stores
- Automatic pipeline composition
- Flexible "out of the box" for simple situations
- Plugin system for adding production implementations

Example:
    >>> from ef import Project
    >>>
    >>> # Create project (works immediately with built-in components)
    >>> project = Project.create('my_project', backend='memory')
    >>>
    >>> # Add data
    >>> project.add_source('doc1', 'Sample text to analyze')
    >>>
    >>> # Create and run pipeline
    >>> project.create_pipeline('analysis', embedder='simple', clusterer='simple_kmeans')
    >>> results = project.run_pipeline('analysis')
    >>>
    >>> # Access persisted results
    >>> print(len(project.embeddings))
"""

from ef.projects import Project, Projects
from ef.base import (
    ComponentRegistry,
    SegmentKey,
    Segment,
    Vector,
    PlanarVector,
    ClusterIndex,
)


# ============================================================================
# Public API
# ============================================================================

__all__ = [
    'Project',
    'Projects',
    'ComponentRegistry',
    'SegmentKey',
    'Segment',
    'Vector',
    'PlanarVector',
    'ClusterIndex',
]
