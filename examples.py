"""
Simple examples showing ef usage patterns.
"""

from ef import Project

print("Example 1: Quickstart")
print("=" * 60)

# Create project and add data
project = Project.create('quickstart', backend='memory')
project.add_source('doc1', 'This is about AI and ML')
project.add_source('doc2', 'Deep learning is powerful')

# Create and run pipeline
project.create_pipeline('analysis', embedder='simple', clusterer='simple_kmeans')
results = project.run_pipeline('analysis')

print(f"Processed {len(results['embeddings'])} documents")
print(f"Clusters: {dict(project.clusters)}")
print()

print("Example 2: Component Discovery")
print("=" * 60)

# List available components
components = project.list_components()
print(f"Available embedders: {', '.join(components['embedders'])}")
print(f"Available clusterers: {', '.join(components['clusterers'])}")
print()

print("Example 3: Quick Embed")
print("=" * 60)

# Quick one-off embedding
embeddings = project.quick_embed("Quick test text")
print(f"Got embedding: {embeddings['main']}")
print()

print("Example 4: Custom Component")
print("=" * 60)


# Register custom embedder
@project.embedders.register('word_count', dimension=1)
def word_count_embedder(segments):
    """Count words in each segment."""
    return {key: [float(len(text.split()))] for key, text in segments.items()}


# Use it
project.create_pipeline('word_count_pipe', embedder='word_count')
results = project.run_pipeline('word_count_pipe')
print(f"Word counts: {dict(results['embeddings'])}")
print()

print("Example 5: Project Summary")
print("=" * 60)

summary = project.summary()
print(f"Project: {summary['project_id']}")
print(f"Data: {summary['n_segments']} segments, {summary['n_embeddings']} embeddings")
print(f"Pipelines: {', '.join(project.list_pipelines())}")
