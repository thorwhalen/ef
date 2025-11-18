"""
Project management for ef (Embedding Flow).

This module provides the main Project and Projects classes for managing
embedding pipelines.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any
from types import SimpleNamespace
from collections.abc import MutableMapping, Iterator
import os
import tempfile

from ef.base import (
    ComponentRegistry,
    SegmentKey,
    Segment,
    Vector,
    PlanarVector,
    ClusterIndex,
)
from ef.storage import mk_project_mall
from ef.dag import assemble_pipeline_dag, DAG


# ============================================================================
# Default Component Registries
# ============================================================================


def _mk_default_registries() -> SimpleNamespace:
    """
    Create default component registries with simple (toy) implementations.

    Returns a SimpleNamespace with registries for:
    - segmenters
    - embedders
    - planarizers
    - clusterers

    These registries come pre-loaded with simple implementations so ef
    works out-of-the-box without requiring heavy dependencies.
    """
    registries = SimpleNamespace(
        segmenters=ComponentRegistry('segmenters'),
        embedders=ComponentRegistry('embedders'),
        planarizers=ComponentRegistry('planarizers'),
        clusterers=ComponentRegistry('clusterers'),
    )

    # Auto-register simple components from plugin
    from ef.plugins import simple

    simple.register_simple_components(
        # Create minimal project-like object for registration
        type(
            '_Project',
            (),
            {
                'segmenters': registries.segmenters,
                'embedders': registries.embedders,
                'planarizers': registries.planarizers,
                'clusterers': registries.clusterers,
            },
        )()
    )

    return registries


# ============================================================================
# Project Class
# ============================================================================


@dataclass
class Project:
    """
    Main project interface for ef pipelines.

    Provides:
    - Component registries (as Mapping stores)
    - Data storage (via mall - store of stores)
    - Pipeline assembly (via DAG composition)
    - Automatic persistence

    Example:
        >>> from ef import Project
        >>> project = Project.create('my_project', backend='memory')
        >>> project.add_source('doc1', 'Sample text')
        >>> _ = project.create_pipeline('test', embedder='simple')
        >>> results = project.run_pipeline('test')
        >>> 'embeddings' in results
        True
    """

    project_id: str
    mall: dict[str, MutableMapping] = field(default_factory=dict)
    registries: SimpleNamespace = field(default_factory=_mk_default_registries)
    pipelines: dict[str, DAG] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        project_id: str,
        *,
        root_dir: str | None = None,
        backend: str = 'files',
        registries: SimpleNamespace | None = None,
        auto_register_segmenters: bool = False,
    ) -> 'Project':
        """
        Create a new project with storage and component registries.

        Args:
            project_id: Unique project identifier
            root_dir: Storage root directory
            backend: Storage backend ('files' or 'memory')
            registries: Custom registries (uses defaults if None)
            auto_register_segmenters: If True, automatically register all available
                external segmenters (LangChain, spaCy, NLTK, etc.) on project creation

        Returns:
            New Project instance

        Example:
            >>> project = Project.create('test', backend='memory')
            >>> list(project.mall.keys())
            ['segments', 'embeddings', 'planar_embeddings', 'clusters']

            >>> # With auto-registration of external segmenters
            >>> project = Project.create('test', backend='memory', auto_register_segmenters=True)
            >>> 'langchain_recursive_1000' in project.list_components()['segmenters']
            True  # If langchain-text-splitters is installed
        """
        mall = mk_project_mall(project_id, root_dir, backend=backend)

        if registries is None:
            registries = _mk_default_registries()

        project = cls(
            project_id=project_id,
            mall=mall,
            registries=registries,
        )

        # Auto-register external segmenters if requested
        if auto_register_segmenters:
            from ef.plugins import segmenter_registry
            segmenter_registry.register_all_segmenters(project, verbose=False)

        return project

    # -------- Component Access --------

    @property
    def segmenters(self) -> ComponentRegistry:
        """Access to segmentation components."""
        return self.registries.segmenters

    @property
    def embedders(self) -> ComponentRegistry:
        """Access to embedding components."""
        return self.registries.embedders

    @property
    def planarizers(self) -> ComponentRegistry:
        """Access to planarization components."""
        return self.registries.planarizers

    @property
    def clusterers(self) -> ComponentRegistry:
        """Access to clustering components."""
        return self.registries.clusterers

    # -------- Data Access --------

    @property
    def segments(self) -> MutableMapping[SegmentKey, Segment]:
        """Access to segments store."""
        return self.mall['segments']

    @property
    def embeddings(self) -> MutableMapping[SegmentKey, Vector]:
        """Access to embeddings store."""
        return self.mall['embeddings']

    @property
    def planar_embeddings(self) -> MutableMapping[SegmentKey, PlanarVector]:
        """Access to planar embeddings store."""
        return self.mall['planar_embeddings']

    @property
    def clusters(self) -> MutableMapping[SegmentKey, ClusterIndex]:
        """Access to clusters store."""
        return self.mall['clusters']

    # -------- Data Operations --------

    def add_source(self, key: str, source_data: Any) -> None:
        """
        Add source data to the project.

        The source will be stored as-is. You can later run a pipeline
        to segment and process it.

        Args:
            key: Unique identifier for this source
            source_data: The data to store (will be converted to string)

        Example:
            >>> project = Project.create('test', backend='memory')
            >>> project.add_source('doc1', 'Sample text')
            >>> 'doc1' in project.segments
            True
        """
        self.segments[key] = str(source_data)

    def list_components(self, component_type: str | None = None) -> dict:
        """
        List available components, optionally filtered by type.

        Args:
            component_type: Type to filter ('segmenters', 'embedders', etc.)
                          If None, returns all components

        Returns:
            Dictionary of component types to lists of component names

        Example:
            >>> project = Project.create('test', backend='memory')
            >>> components = project.list_components()
            >>> 'embedders' in components
            True
            >>> 'simple' in components['embedders']
            True
        """
        all_components = {
            'segmenters': list(self.segmenters.keys()),
            'embedders': list(self.embedders.keys()),
            'planarizers': list(self.planarizers.keys()),
            'clusterers': list(self.clusterers.keys()),
        }

        if component_type:
            return {component_type: all_components.get(component_type, [])}

        return all_components

    # -------- Pipeline Management --------

    def create_pipeline(
        self,
        name: str,
        *,
        segmenter: str | None = None,
        embedder: str | None = None,
        planarizer: str | None = None,
        clusterer: str | None = None,
        **component_params,
    ) -> DAG:
        """
        Create a pipeline from component names.

        Args:
            name: Pipeline name
            segmenter: Name of segmenter component (or None to skip)
            embedder: Name of embedder component (or None to skip)
            planarizer: Name of planarizer component (or None to skip)
            clusterer: Name of clusterer component (or None to skip)
            **component_params: Parameters to pass to components (e.g., n_clusters=5)

        Returns:
            The assembled DAG pipeline

        Example:
            >>> project = Project.create('test', backend='memory')
            >>> _ = project.create_pipeline('my_pipeline', embedder='simple')
            >>> 'my_pipeline' in project.pipelines
            True
        """
        # Get component functions
        seg_func = self.segmenters[segmenter] if segmenter else None
        emb_func = self.embedders[embedder] if embedder else None
        pla_func = self.planarizers[planarizer] if planarizer else None
        clu_func = self.clusterers[clusterer] if clusterer else None

        # Apply any parameters via partial
        if component_params:
            if clu_func and 'n_clusters' in component_params:
                clu_func = partial(clu_func, n_clusters=component_params['n_clusters'])

        # Assemble DAG
        dag = assemble_pipeline_dag(
            segmenter=seg_func,
            embedder=emb_func,
            planarizer=pla_func,
            clusterer=clu_func,
        )

        # Store pipeline
        self.pipelines[name] = dag

        return dag

    def run_pipeline(
        self,
        pipeline_name: str,
        *,
        source_key: str | None = None,
        source_data: Any | None = None,
        persist: bool = True,
    ) -> dict:
        """
        Run a pipeline on source data.

        Args:
            pipeline_name: Name of the pipeline to run
            source_key: Key of source data in segments store (if already added)
            source_data: Source data to process (if not already in store)
            persist: Whether to persist intermediate results to stores

        Returns:
            Dictionary with pipeline results

        Example:
            >>> project = Project.create('test', backend='memory')
            >>> _ = project.create_pipeline('simple', embedder='simple')
            >>> project.segments['doc1'] = 'Test document'
            >>> results = project.run_pipeline('simple', source_key='doc1')
            >>> 'embeddings' in results
            True
        """
        if pipeline_name not in self.pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not found")

        dag = self.pipelines[pipeline_name]

        # Prepare input - treat as segments for the pipeline
        if source_data is not None:
            # Use provided source data
            if isinstance(source_data, str):
                segments = {'main': source_data}
            else:
                segments = source_data
        elif source_key is not None:
            # Load from segments store
            segments = {source_key: self.segments[source_key]}
        else:
            # Use all segments
            segments = dict(self.segments)

        # Run the DAG with segments as input
        results = dag(segments=segments)

        # Persist results if requested
        if persist:
            if 'segments' in results:
                self.segments.update(results['segments'])
            if 'embeddings' in results:
                self.embeddings.update(results['embeddings'])
            if 'planar_embeddings' in results:
                self.planar_embeddings.update(results['planar_embeddings'])
            if 'clusters' in results:
                self.clusters.update(results['clusters'])

        return results

    def list_pipelines(self) -> list[str]:
        """
        List all created pipelines.

        Example:
            >>> project = Project.create('test', backend='memory')
            >>> _ = project.create_pipeline('pipe1', embedder='simple')
            >>> 'pipe1' in project.list_pipelines()
            True
        """
        return list(self.pipelines.keys())

    # -------- Convenience Methods --------

    def quick_embed(
        self,
        source: str | dict[str, str],
        *,
        embedder: str = 'simple',
        segmenter: str | None = None,
    ) -> dict[str, Vector]:
        """
        Quick embedding without creating a named pipeline.

        Args:
            source: Text or dict of texts to embed
            embedder: Name of embedder to use
            segmenter: Optional segmenter name

        Returns:
            Dictionary of embeddings

        Example:
            >>> project = Project.create('test', backend='memory')
            >>> embeddings = project.quick_embed('Test text', embedder='simple')
            >>> 'main' in embeddings
            True
        """
        # Create temporary pipeline
        temp_name = f'_temp_{id(self)}'
        self.create_pipeline(
            temp_name,
            segmenter=segmenter,
            embedder=embedder,
        )

        # Run and cleanup
        try:
            results = self.run_pipeline(
                temp_name,
                source_data=source,
                persist=False,
            )
            return results.get('embeddings', {})
        finally:
            del self.pipelines[temp_name]

    def summary(self) -> dict:
        """
        Get a summary of the project state.

        Returns:
            Dictionary with project statistics

        Example:
            >>> project = Project.create('test', backend='memory')
            >>> summary = project.summary()
            >>> 'project_id' in summary
            True
        """
        return {
            'project_id': self.project_id,
            'n_segments': len(self.segments),
            'n_embeddings': len(self.embeddings),
            'n_planar_embeddings': len(self.planar_embeddings),
            'n_clusters': len(self.clusters),
            'n_pipelines': len(self.pipelines),
            'available_components': self.list_components(),
        }


# ============================================================================
# Projects Manager
# ============================================================================


class Projects(MutableMapping[str, Project]):
    """
    A store of projects, following the "mall" pattern.

    This allows you to manage multiple projects with a dict-like interface.

    Example:
        >>> from ef import Projects
        >>> projects = Projects()
        >>> proj = projects.create_project('proj1', backend='memory')
        >>> 'proj1' in projects
        True
    """

    def __init__(self, root_dir: str | None = None):
        """
        Initialize projects manager.

        Args:
            root_dir: Root directory for all projects
        """
        self.root_dir = root_dir or os.path.join(tempfile.gettempdir(), 'ef_projects')
        self._projects: dict[str, Project] = {}

    def __getitem__(self, key: str) -> Project:
        if key not in self._projects:
            # Try to load from disk
            project_dir = os.path.join(self.root_dir, key)
            if os.path.exists(project_dir):
                self._projects[key] = Project.create(
                    key, root_dir=self.root_dir, backend='files'
                )
            else:
                raise KeyError(f"Project '{key}' not found")
        return self._projects[key]

    def __setitem__(self, key: str, value: Project) -> None:
        if not isinstance(value, Project):
            raise TypeError(f"Value must be a Project, got {type(value)}")
        self._projects[key] = value

    def __delitem__(self, key: str) -> None:
        del self._projects[key]
        # Optionally delete from disk
        import shutil

        project_dir = os.path.join(self.root_dir, key)
        if os.path.exists(project_dir):
            shutil.rmtree(project_dir)

    def __iter__(self) -> Iterator[str]:
        # List all projects in memory and on disk
        disk_projects = set()
        if os.path.exists(self.root_dir):
            disk_projects = {
                d
                for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            }
        return iter(set(self._projects.keys()) | disk_projects)

    def __len__(self) -> int:
        return len(list(iter(self)))

    def create_project(self, project_id: str, **kwargs) -> Project:
        """
        Create a new project and add it to the manager.

        Args:
            project_id: Unique project identifier
            **kwargs: Additional arguments for Project.create()

        Returns:
            New Project instance

        Example:
            >>> projects = Projects()
            >>> proj = projects.create_project('new_proj', backend='memory')
            >>> proj.project_id
            'new_proj'
        """
        project = Project.create(project_id, root_dir=self.root_dir, **kwargs)
        self[project_id] = project
        return project
