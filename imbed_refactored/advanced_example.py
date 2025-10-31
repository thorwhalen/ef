"""
Advanced imbed example with real integrations.

This example shows how to:
1. Add real OpenAI embedders
2. Use UMAP for planarization
3. Use scikit-learn for clustering  
4. Persist to files and databases
5. Visualize results

Dependencies (optional):
    pip install openai umap-learn scikit-learn plotly pandas
"""

from imbed_refactored import Project, ComponentRegistry, mk_project_mall
from functools import partial
import numpy as np


def add_real_components(project: Project):
    """Add real-world components to a project."""
    
    # ========================================================================
    # Real Embedders
    # ========================================================================
    
    try:
        import openai
        from openai import OpenAI
        
        @project.embedders.register('openai-small', dimension=1536, cost_per_1k=0.00002)
        def openai_small_embedder(segments: dict[str, str]) -> dict[str, list[float]]:
            """OpenAI text-embedding-3-small model."""
            client = OpenAI()
            results = {}
            
            # Batch process
            texts = list(segments.values())
            keys = list(segments.keys())
            
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            
            for key, embedding_obj in zip(keys, response.data):
                results[key] = embedding_obj.embedding
            
            return results
        
        @project.embedders.register('openai-large', dimension=3072, cost_per_1k=0.00013)
        def openai_large_embedder(segments: dict[str, str]) -> dict[str, list[float]]:
            """OpenAI text-embedding-3-large model."""
            client = OpenAI()
            results = {}
            
            texts = list(segments.values())
            keys = list(segments.keys())
            
            response = client.embeddings.create(
                input=texts,
                model="text-embedding-3-large"
            )
            
            for key, embedding_obj in zip(keys, response.data):
                results[key] = embedding_obj.embedding
            
            return results
        
        print("✓ Registered OpenAI embedders")
    
    except ImportError:
        print("○ OpenAI not available (pip install openai)")
    
    # ========================================================================
    # Real Planarizers
    # ========================================================================
    
    try:
        from umap import UMAP
        
        @project.planarizers.register('umap', n_components=2)
        def umap_planarizer(
            embeddings: dict[str, list[float]],
            *,
            n_neighbors: int = 15,
            min_dist: float = 0.1,
            metric: str = 'cosine'
        ) -> dict[str, tuple[float, float]]:
            """UMAP dimensionality reduction to 2D."""
            keys = list(embeddings.keys())
            vectors = np.array(list(embeddings.values()))
            
            reducer = UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                n_components=2,
                random_state=42
            )
            
            reduced = reducer.fit_transform(vectors)
            
            return {
                key: (float(reduced[i, 0]), float(reduced[i, 1]))
                for i, key in enumerate(keys)
            }
        
        print("✓ Registered UMAP planarizer")
    
    except ImportError:
        print("○ UMAP not available (pip install umap-learn)")
    
    # ========================================================================
    # Real Clusterers
    # ========================================================================
    
    try:
        from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
        
        @project.clusterers.register('kmeans')
        def kmeans_clusterer(
            embeddings: dict[str, list[float]],
            *,
            n_clusters: int = 5,
            random_state: int = 42
        ) -> dict[str, int]:
            """K-Means clustering."""
            keys = list(embeddings.keys())
            vectors = np.array(list(embeddings.values()))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
            labels = kmeans.fit_predict(vectors)
            
            return {key: int(labels[i]) for i, key in enumerate(keys)}
        
        @project.clusterers.register('dbscan')
        def dbscan_clusterer(
            embeddings: dict[str, list[float]],
            *,
            eps: float = 0.5,
            min_samples: int = 5
        ) -> dict[str, int]:
            """DBSCAN clustering (density-based)."""
            keys = list(embeddings.keys())
            vectors = np.array(list(embeddings.values()))
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(vectors)
            
            return {key: int(labels[i]) for i, key in enumerate(keys)}
        
        @project.clusterers.register('hierarchical')
        def hierarchical_clusterer(
            embeddings: dict[str, list[float]],
            *,
            n_clusters: int = 5,
            linkage: str = 'ward'
        ) -> dict[str, int]:
            """Hierarchical/Agglomerative clustering."""
            keys = list(embeddings.keys())
            vectors = np.array(list(embeddings.values()))
            
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=linkage
            )
            labels = clustering.fit_predict(vectors)
            
            return {key: int(labels[i]) for i, key in enumerate(keys)}
        
        print("✓ Registered sklearn clusterers")
    
    except ImportError:
        print("○ scikit-learn not available (pip install scikit-learn)")


def visualize_results(project: Project, pipeline_name: str):
    """Create an interactive visualization of results."""
    
    try:
        import plotly.graph_objects as go
        import pandas as pd
        
        # Get data
        segments = dict(project.segments)
        planar = dict(project.planar_embeddings)
        clusters = dict(project.clusters)
        
        # Create DataFrame
        df = pd.DataFrame({
            'key': list(segments.keys()),
            'text': list(segments.values()),
            'x': [planar[k][0] for k in segments.keys()],
            'y': [planar[k][1] for k in segments.keys()],
            'cluster': [clusters[k] for k in segments.keys()],
        })
        
        # Create scatter plot
        fig = go.Figure()
        
        for cluster_id in df['cluster'].unique():
            cluster_df = df[df['cluster'] == cluster_id]
            fig.add_trace(go.Scatter(
                x=cluster_df['x'],
                y=cluster_df['y'],
                mode='markers+text',
                name=f'Cluster {cluster_id}',
                text=cluster_df['key'],
                textposition='top center',
                marker=dict(size=10),
                hovertext=cluster_df['text'],
            ))
        
        fig.update_layout(
            title=f'Pipeline: {pipeline_name}',
            xaxis_title='Dimension 1',
            yaxis_title='Dimension 2',
            hovermode='closest'
        )
        
        fig.show()
        print("✓ Visualization created")
    
    except ImportError:
        print("○ Plotly not available (pip install plotly pandas)")


def example_with_real_data():
    """
    Example using real embedders and data.
    """
    print("=" * 70)
    print("Advanced Imbed Example with Real Components")
    print("=" * 70)
    
    # Create project
    project = Project.create('real_analysis', backend='memory')
    
    # Add real components
    print("\nAdding real components...")
    add_real_components(project)
    
    # Show available components
    print("\nAvailable components:")
    for comp_type, components in project.list_components().items():
        print(f"  {comp_type}: {', '.join(components)}")
    
    # Add sample data (replace with your data)
    sample_texts = {
        'ai_intro': 'Artificial intelligence is transforming technology.',
        'ml_basics': 'Machine learning models learn from data patterns.',
        'dl_neural': 'Deep learning uses neural networks with many layers.',
        'nlp_text': 'Natural language processing analyzes human language.',
        'cv_images': 'Computer vision helps machines understand images.',
        'rl_agents': 'Reinforcement learning trains agents through rewards.',
        'data_science': 'Data science extracts insights from data.',
        'big_data': 'Big data technologies handle massive datasets.',
    }
    
    print(f"\nAdding {len(sample_texts)} sample documents...")
    for key, text in sample_texts.items():
        project.add_source(key, text)
    
    # Try to use real embedder if available, otherwise fall back
    embedder = 'openai-small' if 'openai-small' in project.embedders else 'simple'
    planarizer = 'umap' if 'umap' in project.planarizers else 'simple_2d'
    clusterer = 'kmeans' if 'kmeans' in project.clusterers else 'simple_kmeans'
    
    print(f"\nCreating pipeline with:")
    print(f"  Embedder: {embedder}")
    print(f"  Planarizer: {planarizer}")
    print(f"  Clusterer: {clusterer}")
    
    # Create and run pipeline
    project.create_pipeline(
        'full_pipeline',
        embedder=embedder,
        planarizer=planarizer,
        clusterer=clusterer,
        n_clusters=3
    )
    
    print("\nRunning pipeline...")
    results = project.run_pipeline('full_pipeline')
    
    # Show results
    print("\nResults:")
    print(f"  Processed {len(results['embeddings'])} documents")
    print(f"  Created {len(results['planar_embeddings'])} 2D points")
    print(f"  Found {len(set(results['clusters'].values()))} clusters")
    
    print("\nCluster assignments:")
    for key, cluster_id in sorted(results['clusters'].items()):
        print(f"  {key:15s} → Cluster {cluster_id}")
    
    # Try visualization
    print("\nAttempting visualization...")
    visualize_results(project, 'full_pipeline')
    
    # Show summary
    print("\nProject summary:")
    summary = project.summary()
    for key, value in summary.items():
        if key != 'available_components':
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)
    
    return project


def example_compare_pipelines():
    """
    Example comparing different pipeline configurations.
    """
    print("\n" + "=" * 70)
    print("Comparing Multiple Pipeline Configurations")
    print("=" * 70)
    
    project = Project.create('comparison', backend='memory')
    add_real_components(project)
    
    # Add data
    texts = {
        f'doc{i}': f'Document {i} with some sample text content.'
        for i in range(20)
    }
    
    for key, text in texts.items():
        project.add_source(key, text)
    
    # Try different clustering approaches
    configs = [
        ('simple_3', {'clusterer': 'simple_kmeans', 'n_clusters': 3}),
        ('simple_5', {'clusterer': 'simple_kmeans', 'n_clusters': 5}),
    ]
    
    # Add sklearn-based configs if available
    if 'kmeans' in project.clusterers:
        configs.extend([
            ('kmeans_3', {'clusterer': 'kmeans', 'n_clusters': 3}),
            ('kmeans_5', {'clusterer': 'kmeans', 'n_clusters': 5}),
        ])
    
    if 'hierarchical' in project.clusterers:
        configs.append(
            ('hierarchical_4', {'clusterer': 'hierarchical', 'n_clusters': 4})
        )
    
    print(f"\nTesting {len(configs)} configurations...\n")
    
    results_summary = []
    
    for name, params in configs:
        # Ensure embedder is set
        params['embedder'] = params.get('embedder', 'simple')
        
        project.create_pipeline(name, **params)
        results = project.run_pipeline(name)
        
        n_clusters = len(set(results['clusters'].values()))
        
        print(f"{name:20s}: {n_clusters} clusters")
        results_summary.append((name, n_clusters, params))
    
    print("\n" + "=" * 70)
    print("Comparison complete!")
    print("=" * 70)
    
    return project, results_summary


if __name__ == '__main__':
    # Run advanced example
    project = example_with_real_data()
    
    # Run comparison
    # comparison_project, summary = example_compare_pipelines()
