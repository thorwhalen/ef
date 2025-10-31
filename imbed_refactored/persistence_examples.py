"""
Advanced persistence and pipeline sharing examples.

Shows how to:
1. Persist pipeline definitions
2. Share pipelines between projects
3. Use larder for automatic caching
4. Integrate with dol stores
"""

import json
import pickle
from functools import wraps
from imbed_refactored import Project, Projects, mk_project_mall


# ============================================================================
# Pipeline Persistence
# ============================================================================

def save_pipeline_config(project: Project, pipeline_name: str, filepath: str):
    """
    Save a pipeline configuration to a JSON file.
    
    This allows you to share pipeline definitions without sharing data.
    """
    if pipeline_name not in project.pipelines:
        raise ValueError(f"Pipeline '{pipeline_name}' not found")
    
    dag = project.pipelines[pipeline_name]
    
    # Extract configuration from DAG nodes
    config = {
        'pipeline_name': pipeline_name,
        'nodes': []
    }
    
    for node in dag.nodes:
        # Handle partial functions
        func = node.func
        if hasattr(func, 'func'):  # It's a partial
            func_name = func.func.__name__
        else:
            func_name = func.__name__
        
        node_config = {
            'name': node.name,
            'func_name': func_name,
            'bind': node.bind,
            'out': node.out,
        }
        config['nodes'].append(node_config)
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Saved pipeline '{pipeline_name}' to {filepath}")


def load_pipeline_config(project: Project, filepath: str) -> str:
    """
    Load a pipeline configuration from a JSON file.
    
    Returns the name of the loaded pipeline.
    """
    with open(filepath, 'r') as f:
        config = json.load(f)
    
    pipeline_name = config['pipeline_name']
    
    # Reconstruct pipeline from config
    # (This is simplified - in practice, you'd need to map func_names 
    # back to actual functions in the registries)
    
    print(f"Loaded pipeline '{pipeline_name}' from {filepath}")
    print(f"  Contains {len(config['nodes'])} nodes")
    
    return pipeline_name


# ============================================================================
# Automatic Caching with Larder
# ============================================================================

def add_cached_embedder(project: Project, embedder_name: str, cache_store: dict):
    """
    Add an embedder that automatically caches its results.
    
    Uses larder's store_on_output pattern.
    """
    try:
        from larder import store_on_output
        
        # Get the base embedder
        base_embedder = project.embedders[embedder_name]
        
        # Wrap with caching
        @store_on_output(
            store=cache_store,
            auto_namer=lambda *, arguments, output: str(hash(str(arguments['segments'])))
        )
        def cached_embedder(segments):
            print(f"  Computing embeddings for {len(segments)} segments...")
            return base_embedder(segments)
        
        # Register as new component
        cached_name = f'{embedder_name}_cached'
        project.embedders[cached_name] = cached_embedder
        
        print(f"✓ Added cached embedder: {cached_name}")
        return cached_name
    
    except ImportError:
        print("○ larder not available - caching disabled")
        return embedder_name


def add_cached_clusterer(project: Project, clusterer_name: str, cache_store: dict):
    """
    Add a clusterer that automatically caches its results.
    """
    try:
        from larder import store_on_output
        
        base_clusterer = project.clusterers[clusterer_name]
        
        @store_on_output(
            store=cache_store,
            auto_namer=lambda *, arguments, output: f"n{arguments.get('n_clusters', 'auto')}"
        )
        def cached_clusterer(embeddings, **kwargs):
            print(f"  Computing clusters for {len(embeddings)} embeddings...")
            return base_clusterer(embeddings, **kwargs)
        
        cached_name = f'{clusterer_name}_cached'
        project.clusterers[cached_name] = cached_clusterer
        
        print(f"✓ Added cached clusterer: {cached_name}")
        return cached_name
    
    except ImportError:
        print("○ larder not available - caching disabled")
        return clusterer_name


# ============================================================================
# Cross-Project Pipeline Sharing
# ============================================================================

class PipelineLibrary:
    """
    A library of reusable pipeline configurations.
    
    Allows sharing pipeline definitions across projects.
    """
    
    def __init__(self, storage_path: str = './pipeline_library'):
        """
        Initialize pipeline library.
        
        Args:
            storage_path: Directory to store pipeline definitions
        """
        self.storage_path = storage_path
        import os
        os.makedirs(storage_path, exist_ok=True)
    
    def save(self, project: Project, pipeline_name: str, description: str = ''):
        """
        Save a pipeline to the library.
        """
        import os
        
        filepath = os.path.join(self.storage_path, f'{pipeline_name}.json')
        
        # Save pipeline config
        save_pipeline_config(project, pipeline_name, filepath)
        
        # Also save metadata
        metadata = {
            'name': pipeline_name,
            'description': description,
            'project_id': project.project_id,
        }
        
        metadata_path = os.path.join(self.storage_path, f'{pipeline_name}.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved pipeline to library: {pipeline_name}")
    
    def load(self, project: Project, pipeline_name: str) -> str:
        """
        Load a pipeline from the library into a project.
        """
        import os
        
        filepath = os.path.join(self.storage_path, f'{pipeline_name}.json')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pipeline '{pipeline_name}' not found in library")
        
        return load_pipeline_config(project, filepath)
    
    def list(self) -> list[dict]:
        """
        List all pipelines in the library.
        """
        import os
        
        pipelines = []
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith('.meta.json'):
                filepath = os.path.join(self.storage_path, filename)
                with open(filepath, 'r') as f:
                    metadata = json.load(f)
                pipelines.append(metadata)
        
        return pipelines


# ============================================================================
# Integration with dol for Flexible Storage
# ============================================================================

def create_project_with_s3_storage(project_id: str, bucket_name: str):
    """
    Example: Create a project that stores data in S3.
    
    Requires: pip install boto3 s2
    """
    try:
        from s2 import S3TextStore
        
        # Create S3-backed stores
        mall = {
            'segments': S3TextStore(bucket_name, prefix=f'{project_id}/segments/'),
            'embeddings': S3TextStore(bucket_name, prefix=f'{project_id}/embeddings/'),
            'planar_embeddings': S3TextStore(bucket_name, prefix=f'{project_id}/planar/'),
            'clusters': S3TextStore(bucket_name, prefix=f'{project_id}/clusters/'),
        }
        
        project = Project(
            project_id=project_id,
            mall=mall,
        )
        
        print(f"✓ Created project with S3 storage: {bucket_name}")
        return project
    
    except ImportError:
        print("○ S3 storage not available (pip install boto3 s2)")
        return None


def create_project_with_mongodb_storage(project_id: str, db_name: str):
    """
    Example: Create a project that stores data in MongoDB.
    
    Requires: pip install pymongo mongodol
    """
    try:
        from mongodol import MongoStore
        
        # Create MongoDB-backed stores
        mall = {
            'segments': MongoStore(db_name, collection=f'{project_id}_segments'),
            'embeddings': MongoStore(db_name, collection=f'{project_id}_embeddings'),
            'planar_embeddings': MongoStore(db_name, collection=f'{project_id}_planar'),
            'clusters': MongoStore(db_name, collection=f'{project_id}_clusters'),
        }
        
        project = Project(
            project_id=project_id,
            mall=mall,
        )
        
        print(f"✓ Created project with MongoDB storage: {db_name}")
        return project
    
    except ImportError:
        print("○ MongoDB storage not available (pip install pymongo mongodol)")
        return None


# ============================================================================
# Example Usage
# ============================================================================

def example_pipeline_persistence():
    """
    Example showing how to persist and share pipelines.
    """
    print("=" * 70)
    print("Pipeline Persistence Example")
    print("=" * 70)
    
    # Create a project with a pipeline
    print("\n1. Creating project with pipeline...")
    project1 = Project.create('proj1', backend='memory')
    project1.add_source('doc1', 'Sample text')
    
    project1.create_pipeline(
        'my_analysis',
        embedder='simple',
        clusterer='simple_kmeans',
        n_clusters=3
    )
    
    # Save pipeline configuration
    print("\n2. Saving pipeline configuration...")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, 'pipeline.json')
        save_pipeline_config(project1, 'my_analysis', config_path)
        
        # Load into another project
        print("\n3. Loading pipeline into new project...")
        project2 = Project.create('proj2', backend='memory')
        load_pipeline_config(project2, config_path)
    
    print("\n" + "=" * 70)


def example_pipeline_library():
    """
    Example using PipelineLibrary to share pipelines.
    """
    print("\n" + "=" * 70)
    print("Pipeline Library Example")
    print("=" * 70)
    
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create library
        library = PipelineLibrary(tmpdir)
        
        # Create project with pipelines
        print("\n1. Creating pipelines...")
        project = Project.create('research', backend='memory')
        
        project.create_pipeline('quick_analysis', embedder='simple')
        project.create_pipeline('deep_analysis', embedder='simple', clusterer='simple_kmeans')
        
        # Save to library
        print("\n2. Saving to library...")
        library.save(project, 'quick_analysis', 'Fast analysis for initial exploration')
        library.save(project, 'deep_analysis', 'Thorough analysis with clustering')
        
        # List library contents
        print("\n3. Library contents:")
        for pipeline_meta in library.list():
            print(f"  - {pipeline_meta['name']}: {pipeline_meta['description']}")
    
    print("\n" + "=" * 70)


def example_cached_components():
    """
    Example using cached components for expensive operations.
    """
    print("\n" + "=" * 70)
    print("Cached Components Example")
    print("=" * 70)
    
    # Create project
    project = Project.create('cached_project', backend='memory')
    
    # Create cache stores
    embedding_cache = {}
    cluster_cache = {}
    
    # Add cached versions of components
    print("\n1. Adding cached components...")
    cached_embedder = add_cached_embedder(project, 'simple', embedding_cache)
    cached_clusterer = add_cached_clusterer(project, 'simple_kmeans', cluster_cache)
    
    # Add data
    project.add_source('doc1', 'First document')
    project.add_source('doc2', 'Second document')
    
    # Create pipeline with cached components
    print("\n2. Creating pipeline with cached components...")
    project.create_pipeline(
        'cached_pipeline',
        embedder=cached_embedder,
        clusterer=cached_clusterer,
    )
    
    # Run twice - second time should use cache
    print("\n3. First run (no cache):")
    project.run_pipeline('cached_pipeline')
    
    print("\n4. Second run (with cache):")
    project.run_pipeline('cached_pipeline')
    
    print(f"\n5. Cache stats:")
    print(f"   Embedding cache: {len(embedding_cache)} entries")
    print(f"   Cluster cache: {len(cluster_cache)} entries")
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    # Run examples
    example_pipeline_persistence()
    example_pipeline_library()
    example_cached_components()
    
    print("\n\n" + "=" * 70)
    print("All examples complete!")
    print("=" * 70)
