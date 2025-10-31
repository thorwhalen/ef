"""
Pipeline assembly using DAG composition.

This module provides:
- DAG assembly from components
- Simple DAG execution that collects all intermediate results
- Optional future integration with meshed for advanced features
"""

from typing import Callable, Any
from collections.abc import Mapping
import inspect


# ============================================================================
# Function Node
# ============================================================================


class FuncNode:
    """A node in the pipeline DAG representing a function."""

    def __init__(self, func, name=None, bind=None, out=None):
        self.func = func
        self.name = name or func.__name__
        self.bind = bind or {}
        self.out = out or 'result'


# ============================================================================
# Simple Pipeline DAG
# ============================================================================


class DAG:
    """
    Simple DAG implementation for pipeline execution.

    Executes a sequence of functions, collecting all intermediate results.
    Returns a dictionary with all outputs.
    """

    def __init__(self, nodes):
        """
        Initialize DAG.

        Args:
            nodes: List of FuncNode objects or a single FuncNode
        """
        self.nodes = nodes if isinstance(nodes, list) else [nodes]
        self.graph = {node.name: node for node in self.nodes}

    def __call__(self, **kwargs) -> dict:
        """
        Execute the DAG.

        Args:
            **kwargs: Initial inputs to the pipeline

        Returns:
            Dictionary containing all intermediate and final results
        """
        results = dict(kwargs)

        for node in self.nodes:
            # Get inputs for this node
            sig = inspect.signature(node.func)
            func_kwargs = {}

            # Use bind if specified (explicit mapping)
            if node.bind:
                for param, source in node.bind.items():
                    if source in results:
                        func_kwargs[param] = results[source]
            else:
                # Auto-match parameters by name
                for param in sig.parameters:
                    if param in results:
                        func_kwargs[param] = results[param]

            # Execute function
            try:
                output = node.func(**func_kwargs)
                results[node.out] = output
            except TypeError as e:
                # Provide helpful error message
                raise RuntimeError(
                    f"Error calling {node.name}: {e}\n"
                    f"  Available in results: {list(results.keys())}\n"
                    f"  Tried to pass: {list(func_kwargs.keys())}\n"
                    f"  Function signature: {sig}"
                )

        return results


# ============================================================================
# Pipeline Assembly
# ============================================================================


def assemble_pipeline_dag(
    *,
    segmenter: Callable | None = None,
    embedder: Callable | None = None,
    planarizer: Callable | None = None,
    clusterer: Callable | None = None,
) -> DAG:
    """
    Assemble a DAG from pipeline components.

    Components are connected automatically based on their input/output names.

    Args:
        segmenter: Optional segmentation function (source -> segments)
        embedder: Optional embedding function (segments -> embeddings)
        planarizer: Optional planarization function (embeddings -> planar_embeddings)
        clusterer: Optional clustering function (embeddings -> clusters)

    Returns:
        DAG that can be executed with source data

    >>> def seg(source): return {'main': source}
    >>> def emb(segments): return {k: [1.0, 2.0] for k in segments}
    >>> dag = assemble_pipeline_dag(segmenter=seg, embedder=emb)
    >>> results = dag(source='test')
    >>> 'embeddings' in results
    True
    """
    nodes = []

    # Add segmenter if provided
    if segmenter:
        nodes.append(FuncNode(segmenter, name='segment_func', out='segments'))

    # Add embedder if provided (depends on segments)
    if embedder:
        nodes.append(
            FuncNode(
                embedder,
                name='embed_func',
                bind={'segments': 'segments'},  # Connect to segmenter output
                out='embeddings',
            )
        )

    # Add planarizer if provided (depends on embeddings)
    if planarizer:
        nodes.append(
            FuncNode(
                planarizer,
                name='planarize_func',
                bind={'embeddings': 'embeddings'},  # Connect to embedder output
                out='planar_embeddings',
            )
        )

    # Add clusterer if provided (depends on embeddings)
    if clusterer:
        nodes.append(
            FuncNode(
                clusterer,
                name='cluster_func',
                bind={'embeddings': 'embeddings'},  # Connect to embedder output
                out='clusters',
            )
        )

    return DAG(nodes)
