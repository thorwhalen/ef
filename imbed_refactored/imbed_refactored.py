"""
Refactored imbed framework with integrated dol, meshed, and larder support.

This module provides:
- Easy project creation with flexible storage
- Component registries as mapping stores
- Automatic pipeline composition via meshed
- Automatic persistence via larder
- Clean, composable interfaces
"""

from dataclasses import dataclass, field
from functools import partial, cached_property
from typing import Callable, Iterable, Mapping, MutableMapping, Any
from collections.abc import Iterator
from types import SimpleNamespace
import os
import tempfile

# Third-party imports (with fallbacks for demo purposes)
try:
    from dol import (
        Files, 
        wrap_kvs, 
        add_ipython_key_completions,
        mk_dirs_if_missing
    )
    from dol.trans import store_decorator
    HAVE_DOL = True
except ImportError:
    print("Note: Using fallback implementations (install dol for full functionality)")
    HAVE_DOL = False
    # Simple fallback: no-op decorator
    def add_ipython_key_completions(obj):
        return obj
    
try:
    from meshed import DAG, FuncNode
    HAVE_MESHED = True
except ImportError:
    print("Note: Using fallback implementations (install meshed for full functionality)")
    HAVE_MESHED = False
    
    # Simple fallback DAG implementation
    class FuncNode:
        """Minimal FuncNode implementation for demo."""
        def __init__(self, func, name=None, bind=None, out=None):
            self.func = func
            self.name = name or func.__name__
            self.bind = bind or {}
            self.out = out or 'result'
    
    class DAG:
        """Minimal DAG implementation for demo."""
        def __init__(self, nodes):
            self.nodes = nodes if isinstance(nodes, list) else [nodes]
            self.graph = {node.name: node for node in self.nodes}
        
        def __call__(self, **kwargs):
            """Execute the DAG."""
            results = dict(kwargs)
            
            for node in self.nodes:
                # Get inputs for this node
                import inspect
                sig = inspect.signature(node.func)
                func_kwargs = {}
                
                # Use bind if specified
                if node.bind:
                    for param, source in node.bind.items():
                        if source in results:
                            func_kwargs[param] = results[source]
                else:
                    # Auto-match parameters
                    for param in sig.parameters:
                        if param in results:
                            func_kwargs[param] = results[param]
                
                # Execute function
                try:
                    output = node.func(**func_kwargs)
                    results[node.out] = output
                except TypeError as e:
                    # Debug: show what was attempted
                    print(f"Error calling {node.name}: {e}")
                    print(f"  Available in results: {list(results.keys())}")
                    print(f"  Tried to pass: {list(func_kwargs.keys())}")
                    raise
            
            return results
    
try:
    from larder import store_on_output
    HAVE_LARDER = True
except ImportError:
    print("Note: Using fallback implementations (install larder for full functionality)")
    HAVE_LARDER = False
    
    # Simple fallback: decorator that does nothing
    def store_on_output(*args, **kwargs):
        def decorator(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return decorator


# ============================================================================
# Type Aliases
# ============================================================================

SegmentKey = str
Segment = str
Vector = list[float]
PlanarVector = tuple[float, float]
ClusterIndex = int


# ============================================================================
# Storage Layer (using dol)
# ============================================================================

def mk_extension_based_store(rootdir: str, *, extension: str = 'pkl') -> MutableMapping:
    """
    Create a storage backend that handles different file types.
    
    Uses dol to provide a MutableMapping interface to filesystem storage.
    
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     store = mk_extension_based_store(tmpdir, extension='json')
    ...     store['test'] = {'key': 'value'}
    ...     assert store['test'] == {'key': 'value'}
    """
    import pickle
    import json
    
    os.makedirs(rootdir, exist_ok=True)
    
    if HAVE_DOL:
        from dol import Files, wrap_kvs
        
        # Base file store
        base_store = Files(rootdir)
        
        # Add extension handling and serialization
        if extension == 'pkl':
            encode = pickle.dumps
            decode = pickle.loads
        elif extension == 'json':
            encode = lambda x: json.dumps(x).encode()
            decode = lambda x: json.loads(x.decode())
        else:
            encode = lambda x: str(x).encode()
            decode = lambda x: x.decode()
        
        # Wrap with key transformation (add extension) and value codec
        def _add_ext(k):
            return f"{k}.{extension}"
        
        def _remove_ext(k):
            return k.rsplit('.', 1)[0] if '.' in k else k
        
        store = wrap_kvs(
            base_store,
            key_of_id=_add_ext,
            id_of_key=_remove_ext,
            obj_of_data=decode,
            data_of_obj=encode
        )
        
        return add_ipython_key_completions(store)
    else:
        # Fallback: simple file-based store
        class SimpleFileStore(MutableMapping):
            def __init__(self, rootdir, extension):
                self.rootdir = rootdir
                self.extension = extension
            
            def _filepath(self, key):
                return os.path.join(self.rootdir, f"{key}.{self.extension}")
            
            def __getitem__(self, key):
                filepath = self._filepath(key)
                with open(filepath, 'rb') as f:
                    data = f.read()
                
                if self.extension == 'pkl':
                    return pickle.loads(data)
                elif self.extension == 'json':
                    return json.loads(data.decode())
                else:
                    return data.decode()
            
            def __setitem__(self, key, value):
                filepath = self._filepath(key)
                
                if self.extension == 'pkl':
                    data = pickle.dumps(value)
                elif self.extension == 'json':
                    data = json.dumps(value).encode()
                else:
                    data = str(value).encode()
                
                with open(filepath, 'wb') as f:
                    f.write(data)
            
            def __delitem__(self, key):
                os.remove(self._filepath(key))
            
            def __iter__(self):
                for filename in os.listdir(self.rootdir):
                    if filename.endswith(f'.{self.extension}'):
                        yield filename.rsplit('.', 1)[0]
            
            def __len__(self):
                return sum(1 for _ in self)
        
        return SimpleFileStore(rootdir, extension)


def mk_project_mall(
    project_id: str,
    root_dir: str | None = None,
    *,
    backend: str = 'files',
) -> dict[str, MutableMapping]:
    """
    Create a "mall" (store of stores) for a project.
    
    A mall provides separate storage for each pipeline stage:
    - segments: text segments
    - embeddings: vector embeddings
    - planar_embeddings: 2D coordinates
    - clusters: cluster assignments
    
    Args:
        project_id: Unique identifier for the project
        root_dir: Base directory for storage (uses temp if None)
        backend: Storage backend ('files', 'memory')
    
    Returns:
        Dictionary mapping stage names to storage objects
        
    >>> mall = mk_project_mall('test_project', backend='memory')
    >>> list(mall.keys())
    ['segments', 'embeddings', 'planar_embeddings', 'clusters']
    """
    if root_dir is None:
        root_dir = os.path.join(tempfile.gettempdir(), 'imbed_projects', project_id)
    
    if backend == 'memory':
        # Use simple dicts for in-memory storage
        return {
            'segments': {},
            'embeddings': {},
            'planar_embeddings': {},
            'clusters': {}
        }
    elif backend == 'files':
        # Use filesystem storage with appropriate serialization
        return {
            'segments': mk_extension_based_store(
                os.path.join(root_dir, 'segments'), extension='txt'
            ),
            'embeddings': mk_extension_based_store(
                os.path.join(root_dir, 'embeddings'), extension='pkl'
            ),
            'planar_embeddings': mk_extension_based_store(
                os.path.join(root_dir, 'planar_embeddings'), extension='json'
            ),
            'clusters': mk_extension_based_store(
                os.path.join(root_dir, 'clusters'), extension='json'
            ),
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")


# ============================================================================
# Component Registries (Mapping-based stores)
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
        self, 
        name: str | None = None, 
        **metadata
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


def _mk_default_registries() -> SimpleNamespace:
    """
    Create default component registries for common pipeline stages.
    
    Returns a SimpleNamespace with registries for:
    - segmenters
    - embedders
    - planarizers
    - clusterers
    """
    registries = SimpleNamespace(
        segmenters=ComponentRegistry('segmenters'),
        embedders=ComponentRegistry('embedders'),
        planarizers=ComponentRegistry('planarizers'),
        clusterers=ComponentRegistry('clusterers'),
    )
    
    # Register some basic components
    
    # Segmenter: identity (no segmentation)
    @registries.segmenters.register('identity')
    def identity_segmenter(source):
        """Return source as-is."""
        if isinstance(source, str):
            return {'main': source}
        return source
    
    # Segmenter: lines
    @registries.segmenters.register('lines')
    def line_segmenter(source: str) -> dict[str, str]:
        """Split text into lines."""
        lines = source.split('\n')
        return {f'line_{i}': line for i, line in enumerate(lines) if line.strip()}
    
    # Embedder: simple (for testing)
    @registries.embedders.register('simple', dimension=3)
    def simple_embedder(segments: Mapping[str, str]) -> dict[str, list[float]]:
        """Simple embedder for testing (counts characters, words, punctuation)."""
        result = {}
        for key, text in segments.items():
            n_chars = len(text)
            n_words = len(text.split())
            n_punct = sum(1 for c in text if c in '.,!?;:')
            result[key] = [float(n_chars), float(n_words), float(n_punct)]
        return result
    
    # Planarizer: PCA-like (fake for testing)
    @registries.planarizers.register('simple_2d')
    def simple_planarizer(embeddings: Mapping[str, Vector]) -> dict[str, PlanarVector]:
        """Simple 2D projection (just takes first two dimensions)."""
        return {
            key: (vec[0] if len(vec) > 0 else 0.0, vec[1] if len(vec) > 1 else 0.0)
            for key, vec in embeddings.items()
        }
    
    # Clusterer: simple k-means style (fake for testing)
    @registries.clusterers.register('simple_kmeans')
    def simple_clusterer(
        embeddings: Mapping[str, Vector], *, n_clusters: int = 3
    ) -> dict[str, int]:
        """Simple clustering based on first dimension."""
        import numpy as np
        
        keys = list(embeddings.keys())
        vecs = list(embeddings.values())
        
        if not vecs:
            return {}
        
        # Simple clustering: sort by first dimension and split into n groups
        first_dims = [v[0] if v else 0.0 for v in vecs]
        sorted_indices = np.argsort(first_dims)
        
        clusters = {}
        items_per_cluster = len(keys) // n_clusters + 1
        
        for i, idx in enumerate(sorted_indices):
            cluster_id = i // items_per_cluster
            clusters[keys[idx]] = min(cluster_id, n_clusters - 1)
        
        return clusters
    
    return registries


# ============================================================================
# Pipeline Assembly (using meshed)
# ============================================================================

def _assemble_pipeline_dag(
    *,
    segmenter: Callable | None,
    embedder: Callable | None,
    planarizer: Callable | None,
    clusterer: Callable | None,
) -> DAG:
    """
    Assemble a DAG from pipeline components using meshed.
    
    Components are connected automatically based on their input/output names.
    """
    nodes = []
    
    # Add segmenter if provided
    if segmenter:
        nodes.append(
            FuncNode(segmenter, name='segment_func', out='segments')
        )
    
    # Add embedder if provided (depends on segments)
    if embedder:
        nodes.append(
            FuncNode(
                embedder, 
                name='embed_func',
                bind={'segments': 'segments'},  # Connect to segmenter output
                out='embeddings'
            )
        )
    
    # Add planarizer if provided (depends on embeddings)
    if planarizer:
        nodes.append(
            FuncNode(
                planarizer,
                name='planarize_func',
                bind={'embeddings': 'embeddings'},  # Connect to embedder output
                out='planar_embeddings'
            )
        )
    
    # Add clusterer if provided (depends on embeddings)
    if clusterer:
        nodes.append(
            FuncNode(
                clusterer,
                name='cluster_func',
                bind={'embeddings': 'embeddings'},  # Connect to embedder output
                out='clusters'
            )
        )
    
    return DAG(nodes)


# ============================================================================
# Project Class (main interface)
# ============================================================================

@dataclass
class Project:
    """
    Main project interface for imbed pipelines.
    
    Provides:
    - Component registries (as Mapping stores)
    - Data storage (via mall - store of stores)
    - Pipeline assembly (via meshed DAGs)
    - Automatic persistence (via larder)
    
    Example:
        >>> project = Project.create('my_project', backend='memory')
        >>> # Add some data
        >>> project.add_source('doc1', 'This is a test document.')
        >>> # Run pipeline
        >>> project.run_pipeline('simple_pipeline')
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
    ) -> 'Project':
        """
        Create a new project with storage and component registries.
        
        Args:
            project_id: Unique project identifier
            root_dir: Storage root directory  
            backend: Storage backend ('files' or 'memory')
            registries: Custom registries (uses defaults if None)
            
        >>> project = Project.create('test', backend='memory')
        >>> list(project.mall.keys())
        ['segments', 'embeddings', 'planar_embeddings', 'clusters']
        """
        mall = mk_project_mall(project_id, root_dir, backend=backend)
        
        if registries is None:
            registries = _mk_default_registries()
        
        return cls(
            project_id=project_id,
            mall=mall,
            registries=registries,
        )
    
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
        """
        # For now, treat source as a segment
        self.segments[key] = str(source_data)
    
    def list_components(self, component_type: str | None = None) -> dict:
        """
        List available components, optionally filtered by type.
        
        Args:
            component_type: Type to filter ('segmenters', 'embedders', etc.)
                          If None, returns all components
                          
        Returns:
            Dictionary of component types to lists of component names
            
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
            **component_params: Parameters to pass to components
            
        Returns:
            The assembled DAG pipeline
            
        >>> project = Project.create('test', backend='memory')
        >>> pipeline = project.create_pipeline(
        ...     'my_pipeline',
        ...     segmenter='lines',
        ...     embedder='simple',
        ... )
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
        dag = _assemble_pipeline_dag(
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
            
        >>> project = Project.create('test', backend='memory')
        >>> project.create_pipeline('simple', embedder='simple')
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
        
        >>> project = Project.create('test', backend='memory')
        >>> project.create_pipeline('pipe1', embedder='simple')
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
# Projects Manager (store of projects)
# ============================================================================

class Projects(MutableMapping[str, Project]):
    """
    A store of projects, following the "mall" pattern.
    
    This allows you to manage multiple projects with a dict-like interface.
    
    >>> projects = Projects()
    >>> projects['proj1'] = Project.create('proj1', backend='memory')
    >>> 'proj1' in projects
    True
    """
    
    def __init__(self, root_dir: str | None = None):
        """
        Initialize projects manager.
        
        Args:
            root_dir: Root directory for all projects
        """
        self.root_dir = root_dir or os.path.join(
            tempfile.gettempdir(), 'imbed_projects'
        )
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
                d for d in os.listdir(self.root_dir)
                if os.path.isdir(os.path.join(self.root_dir, d))
            }
        return iter(set(self._projects.keys()) | disk_projects)
    
    def __len__(self) -> int:
        return len(list(iter(self)))
    
    def create_project(self, project_id: str, **kwargs) -> Project:
        """
        Create a new project and add it to the manager.
        
        >>> projects = Projects()
        >>> proj = projects.create_project('new_proj', backend='memory')
        >>> proj.project_id
        'new_proj'
        """
        project = Project.create(project_id, root_dir=self.root_dir, **kwargs)
        self[project_id] = project
        return project


# ============================================================================
# Main Example and Tests
# ============================================================================

def example_usage():
    """
    Example showing typical usage patterns.
    """
    print("=" * 70)
    print("IMBED Framework Example Usage")
    print("=" * 70)
    
    # 1. Create a project
    print("\n1. Creating project...")
    project = Project.create('my_analysis', backend='memory')
    print(f"   Created project: {project.project_id}")
    
    # 2. List available components
    print("\n2. Available components:")
    components = project.list_components()
    for comp_type, comp_list in components.items():
        print(f"   {comp_type}: {', '.join(comp_list)}")
    
    # 3. Add some source data
    print("\n3. Adding source data...")
    project.add_source('doc1', 'This is the first document about AI.')
    project.add_source('doc2', 'Machine learning is fascinating.')
    project.add_source('doc3', 'Deep learning uses neural networks.')
    print(f"   Added {len(project.segments)} documents")
    
    # 4. Create a pipeline
    print("\n4. Creating pipeline...")
    pipeline = project.create_pipeline(
        'full_analysis',
        embedder='simple',
        planarizer='simple_2d',
        clusterer='simple_kmeans',
        n_clusters=2
    )
    print(f"   Created pipeline: 'full_analysis'")
    print(f"   Pipeline has {len(pipeline.graph)} nodes")
    
    # 5. Run the pipeline
    print("\n5. Running pipeline...")
    results = project.run_pipeline('full_analysis')
    print(f"   Results:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"     {key}: {len(value)} items")
    
    # 6. Access persisted data
    print("\n6. Persisted data:")
    print(f"   Segments: {len(project.segments)}")
    print(f"   Embeddings: {len(project.embeddings)}")
    print(f"   Planar embeddings: {len(project.planar_embeddings)}")
    print(f"   Clusters: {len(project.clusters)}")
    
    # 7. Show cluster assignments
    print("\n7. Cluster assignments:")
    for key, cluster_id in project.clusters.items():
        print(f"   {key}: cluster {cluster_id}")
    
    # 8. Project summary
    print("\n8. Project summary:")
    summary = project.summary()
    for key, value in summary.items():
        if key != 'available_components':
            print(f"   {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)


if __name__ == '__main__':
    # Run the example
    example_usage()
    
    # Run doctests
    print("\n\nRunning doctests...")
    import doctest
    results = doctest.testmod()
    if results.failed == 0:
        print(f"✓ All {results.attempted} doctests passed!")
    else:
        print(f"✗ {results.failed} of {results.attempted} doctests failed")
