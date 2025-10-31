"""
Simple built-in components for ef.

These toy implementations allow ef to work out-of-the-box without
requiring heavy ML dependencies.
"""

from collections.abc import Mapping
from typing import Any


def register_simple_components(project):
    """
    Register all simple (toy) components with a project.

    Args:
        project: An ef.Project instance

    Example:
        >>> from ef import Project
        >>> from ef.plugins import simple
        >>> project = Project.create('test', backend='memory')
        >>> simple.register_simple_components(project)
        >>> 'simple' in project.embedders
        True
    """
    _register_segmenters(project)
    _register_embedders(project)
    _register_planarizers(project)
    _register_clusterers(project)


def _register_segmenters(project):
    """Register simple segmentation components."""

    @project.segmenters.register('identity')
    def identity_segmenter(source: Any) -> dict[str, str]:
        """Return source as-is (no segmentation)."""
        if isinstance(source, str):
            return {'main': source}
        return source

    @project.segmenters.register('lines')
    def line_segmenter(source: str) -> dict[str, str]:
        """Split text into lines."""
        lines = source.split('\n')
        return {f'line_{i}': line for i, line in enumerate(lines) if line.strip()}

    @project.segmenters.register('sentences')
    def sentence_segmenter(source: str) -> dict[str, str]:
        """Split text into sentences (simple period-based)."""
        import re

        sentences = re.split(r'[.!?]+', source)
        return {
            f'sent_{i}': sent.strip()
            for i, sent in enumerate(sentences)
            if sent.strip()
        }


def _register_embedders(project):
    """Register simple embedding components."""

    @project.embedders.register('simple', dimension=3)
    def simple_embedder(segments: Mapping[str, str]) -> dict[str, list[float]]:
        """
        Simple embedder for testing.

        Counts characters, words, and punctuation.
        """
        result = {}
        for key, text in segments.items():
            n_chars = len(text)
            n_words = len(text.split())
            n_punct = sum(1 for c in text if c in '.,!?;:')
            result[key] = [float(n_chars), float(n_words), float(n_punct)]
        return result

    @project.embedders.register('char_counts', dimension=26)
    def char_count_embedder(segments: Mapping[str, str]) -> dict[str, list[float]]:
        """
        Embed text as character frequency vector.

        Returns a 26-dimensional vector of letter frequencies.
        """
        result = {}
        for key, text in segments.items():
            text_lower = text.lower()
            counts = [float(text_lower.count(chr(ord('a') + i))) for i in range(26)]
            result[key] = counts
        return result


def _register_planarizers(project):
    """Register simple planarization components."""

    @project.planarizers.register('simple_2d')
    def simple_planarizer(
        embeddings: Mapping[str, list[float]],
    ) -> dict[str, tuple[float, float]]:
        """
        Simple 2D projection (just takes first two dimensions).
        """
        return {
            key: (vec[0] if len(vec) > 0 else 0.0, vec[1] if len(vec) > 1 else 0.0)
            for key, vec in embeddings.items()
        }

    @project.planarizers.register('normalize_2d')
    def normalize_2d_planarizer(
        embeddings: Mapping[str, list[float]],
    ) -> dict[str, tuple[float, float]]:
        """
        Project to 2D and normalize to unit circle.
        """
        import math

        result = {}
        for key, vec in embeddings.items():
            x = vec[0] if len(vec) > 0 else 0.0
            y = vec[1] if len(vec) > 1 else 0.0

            # Normalize
            magnitude = math.sqrt(x * x + y * y)
            if magnitude > 0:
                x, y = x / magnitude, y / magnitude

            result[key] = (x, y)

        return result


def _register_clusterers(project):
    """Register simple clustering components."""

    @project.clusterers.register('simple_kmeans')
    def simple_clusterer(
        embeddings: Mapping[str, list[float]], *, n_clusters: int = 3
    ) -> dict[str, int]:
        """
        Simple clustering based on first dimension.

        Sorts by first dimension and splits into n groups.
        """
        import numpy as np

        keys = list(embeddings.keys())
        vecs = list(embeddings.values())

        if not vecs:
            return {}

        # Sort by first dimension and split into groups
        first_dims = [v[0] if v else 0.0 for v in vecs]
        sorted_indices = np.argsort(first_dims)

        clusters = {}
        items_per_cluster = len(keys) // n_clusters + 1

        for i, idx in enumerate(sorted_indices):
            cluster_id = i // items_per_cluster
            clusters[keys[idx]] = min(cluster_id, n_clusters - 1)

        return clusters

    @project.clusterers.register('threshold')
    def threshold_clusterer(
        embeddings: Mapping[str, list[float]], *, threshold: float = 10.0
    ) -> dict[str, int]:
        """
        Binary clustering based on magnitude threshold.

        Cluster 0: magnitude < threshold
        Cluster 1: magnitude >= threshold
        """
        import math

        result = {}
        for key, vec in embeddings.items():
            magnitude = math.sqrt(sum(x * x for x in vec))
            result[key] = 0 if magnitude < threshold else 1

        return result
