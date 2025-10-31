#!/usr/bin/env python
"""
Demo script for ef (Embedding Flow).

This script demonstrates the core functionality of ef:
- Creating projects
- Adding data
- Listing components
- Creating pipelines
- Running pipelines
- Accessing results
"""

from ef import Project


def main():
    """Run ef demo."""

    print("=" * 70)
    print("ef (Embedding Flow) Demo")
    print("=" * 70)

    # 1. Create project
    print("\n1. Creating project...")
    project = Project.create('demo_project', backend='memory')
    print(f"   ✓ Created project: {project.project_id}")

    # 2. List available components
    print("\n2. Available components:")
    components = project.list_components()
    for comp_type, comp_list in components.items():
        print(f"   {comp_type:20s}: {', '.join(comp_list)}")

    # 3. Add source data
    print("\n3. Adding source data...")
    sample_texts = {
        'ai_intro': 'Artificial intelligence is transforming technology.',
        'ml_basics': 'Machine learning models learn from data patterns.',
        'dl_neural': 'Deep learning uses neural networks with many layers.',
    }

    for key, text in sample_texts.items():
        project.add_source(key, text)

    print(f"   ✓ Added {len(sample_texts)} documents")

    # 4. Create pipeline
    print("\n4. Creating pipeline...")
    pipeline = project.create_pipeline(
        'full_analysis',
        embedder='simple',
        planarizer='simple_2d',
        clusterer='simple_kmeans',
        n_clusters=2,
    )
    print(f"   ✓ Created pipeline: 'full_analysis'")
    print(f"   ✓ Pipeline has {len(pipeline.graph)} nodes")

    # 5. Run pipeline
    print("\n5. Running pipeline...")
    results = project.run_pipeline('full_analysis')

    print(f"   Results summary:")
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"     {key:20s}: {len(value)} items")
        else:
            print(f"     {key:20s}: {type(value).__name__}")

    # 6. Access persisted data
    print("\n6. Persisted data:")
    print(f"   Segments: {len(project.segments)}")
    print(f"   Embeddings: {len(project.embeddings)}")
    print(f"   Planar embeddings: {len(project.planar_embeddings)}")
    print(f"   Clusters: {len(project.clusters)}")

    # 7. Show cluster assignments
    print("\n7. Cluster assignments:")
    for key, cluster_id in project.clusters.items():
        text_preview = project.segments[key][:40] + '...'
        print(f"   {key:12s} → Cluster {cluster_id}: {text_preview}")

    # 8. Quick embed demo
    print("\n8. Quick embed demo (no pipeline):")
    quick_result = project.quick_embed('Test text for quick embedding')
    print(f"   ✓ Got embedding: {quick_result['main']}")

    # 9. Project summary
    print("\n9. Project summary:")
    summary = project.summary()
    for key, value in summary.items():
        if key != 'available_components':
            print(f"   {key:25s}: {value}")

    # 10. List pipelines
    print("\n10. Available pipelines:")
    for pipeline_name in project.list_pipelines():
        print(f"   - {pipeline_name}")

    print("\n" + "=" * 70)
    print("Demo complete! ✓")
    print("=" * 70)

    print("\nNext steps:")
    print("  • Install full dependencies: pip install ef[full]")
    print("  • Add production components: from ef.plugins import imbed")
    print("  • Create your own plugins")
    print("  • Explore the documentation")


if __name__ == '__main__':
    main()
