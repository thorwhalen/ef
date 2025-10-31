"""
Core types and protocols for ef (Embedding Flow).

This module provides:
- ComponentRegistry: Mapping-based store for pipeline components
- Type aliases for common data types
- Core protocols and interfaces
"""

from collections.abc import Iterator, Callable, MutableMapping
from typing import Any


# ============================================================================
# Type Aliases
# ============================================================================

SegmentKey = str
Segment = str
Vector = list[float]
PlanarVector = tuple[float, float]
ClusterIndex = int


# ============================================================================
# Component Registry
# ============================================================================


class ComponentRegistry(MutableMapping):
    """
    A registry that stores pipeline components (functions) in a Mapping interface.

    This allows components to be accessed like a dictionary while maintaining
    additional metadata about each component.

    >>> registry = ComponentRegistry('embedders')
    >>> registry['simple'] = lambda x: [1.0, 2.0, 3.0]
    >>> 'simple' in registry
    True
    >>> result = registry['simple']("test text")
    """

    def __init__(self, name: str):
        self.name = name
        self._components: dict[str, Callable] = {}
        self._metadata: dict[str, dict] = {}

    def __getitem__(self, key: str) -> Callable:
        return self._components[key]

    def __setitem__(self, key: str, value: Callable) -> None:
        if not callable(value):
            raise TypeError(f"Component must be callable, got {type(value)}")
        self._components[key] = value

    def __delitem__(self, key: str) -> None:
        del self._components[key]
        if key in self._metadata:
            del self._metadata[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._components)

    def __len__(self) -> int:
        return len(self._components)

    def register(
        self, name: str | None = None, **metadata
    ) -> Callable[[Callable], Callable]:
        """
        Decorator to register a component.

        >>> registry = ComponentRegistry('embedders')
        >>> @registry.register('my_embedder', dimension=128)
        ... def embed(text):
        ...     return [1.0] * 128
        >>> 'my_embedder' in registry
        True
        """

        def decorator(func: Callable) -> Callable:
            key = name or func.__name__
            self[key] = func
            if metadata:
                self._metadata[key] = metadata
            return func

        return decorator

    def get_metadata(self, key: str) -> dict:
        """Get metadata for a component."""
        return self._metadata.get(key, {})
