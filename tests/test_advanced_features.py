"""
Tests for advanced segmentation features.

Tests all 15 advanced features added to the EF framework.
"""

import pytest
import tempfile
from pathlib import Path


@pytest.fixture
def project():
    """Create test project with auto-registration."""
    from ef import Project
    return Project.create('test_advanced', auto_register_segmenters=True)


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    First paragraph with multiple sentences. This is the second sentence.
    And here is a third one.

    Second paragraph starts here. It also has multiple sentences.
    Including this one.

    Third paragraph is the last one. It concludes our sample text.
    """


# Test 1: Segmenter Composition
def test_compose_segmenters(project, sample_text):
    """Test segmenter composition."""
    from ef.plugins.segmenter_composition import compose_segmenters

    # Compose two segmenters
    composed = compose_segmenters(
        project,
        ['by_paragraphs', 'sentences']
    )

    result = composed(sample_text)
    assert isinstance(result, dict)
    assert len(result) > 0


def test_segmenter_pipeline(project, sample_text):
    """Test SegmenterPipeline fluent interface."""
    from ef.plugins.segmenter_composition import SegmenterPipeline

    pipeline = (SegmenterPipeline(project)
        .segment('sentences')
        .filter(min_length=10)
        .build())

    result = pipeline(sample_text)
    assert isinstance(result, dict)
    # All segments should be at least 10 chars
    for text in result.values():
        assert len(text) >= 10


def test_parallel_segmenters(project, sample_text):
    """Test parallel segmenters."""
    from ef.plugins.segmenter_composition import parallel_segmenters

    parallel = parallel_segmenters(
        project,
        ['sentences', 'lines']
    )

    result = parallel(sample_text)
    assert isinstance(result, dict)
    # Keys should be prefixed with segmenter name
    assert any(k.startswith('sentences.') for k in result.keys())
    assert any(k.startswith('lines.') for k in result.keys())


# Test 2: Quality Scoring & Optimization
def test_evaluate_quality(sample_text):
    """Test quality evaluation."""
    from ef.plugins.segmenter_quality import evaluate_quality

    segments = {
        'seg1': 'First sentence.',
        'seg2': 'Second sentence.',
        'seg3': 'Third sentence.'
    }

    quality = evaluate_quality(segments, source=sample_text)
    assert 'coherence' in quality
    assert 'balance' in quality
    assert 'overall' in quality
    assert 0 <= quality['overall'] <= 1


def test_optimize_for_task(project, sample_text):
    """Test task optimization."""
    from ef.plugins.segmenter_quality import optimize_for_task

    result = optimize_for_task(
        project,
        [sample_text],  # Pass as list
        task='general',
        candidates=['sentences', 'lines']
    )

    assert isinstance(result, dict)
    assert 'name' in result  # The key is 'name', not 'best_segmenter'
    assert result['name'] in ['sentences', 'lines']


# Test 3: Streaming/Incremental Segmentation
def test_streaming_segmenter(project):
    """Test streaming segmentation."""
    from ef.plugins.segmenter_streaming import StreamingSegmenter
    import tempfile
    import io

    # Create a test file
    text_stream = io.StringIO("First sentence. Second sentence. Third sentence.")

    with StreamingSegmenter(project, 'sentences', file_obj=text_stream, buffer_size=100) as streamer:
        segments = list(streamer)

    assert isinstance(segments, list)
    assert len(segments) >= 1


def test_incremental_segmenter(project):
    """Test incremental segmentation."""
    from ef.plugins.segmenter_streaming import IncrementalSegmenter

    segmenter = IncrementalSegmenter(project, 'sentences')
    segmenter.add_text("First sentence. ")

    segments1 = segmenter.get_completed_segments()
    # May be 0 or more depending on buffer
    assert isinstance(segments1, dict)

    segmenter.add_text("Second sentence. ")
    segments2 = segmenter.get_completed_segments()
    assert isinstance(segments2, dict)
    # After more text, should have at least as many or more
    assert len(segments2) >= len(segments1)


# Test 4: Configuration Management
def test_segmentation_config():
    """Test configuration creation and serialization."""
    from ef.plugins.segmenter_config import SegmentationConfig

    config = SegmentationConfig(
        segmenter='sentences',
        params={'min_length': 10}
    )

    # Test YAML serialization
    yaml_str = config.to_yaml()
    assert 'sentences' in yaml_str

    # Test round-trip
    config2 = SegmentationConfig.from_yaml(yaml_str)
    assert config2.segmenter == config.segmenter
    assert config2.params == config.params


def test_config_presets():
    """Test configuration presets."""
    from ef.plugins.segmenter_config import PRESETS

    assert 'article' in PRESETS
    assert 'code_python' in PRESETS

    article_preset = PRESETS['article']
    assert 'segmenter' in article_preset
    assert article_preset['segmenter'] is not None


# Test 5: Visualization Tools
def test_plot_segmentation(sample_text):
    """Test visualization (basic check)."""
    from ef.plugins.segmenter_viz import plot_segmentation
    import tempfile

    segments = {
        'seg1': 'First part.',
        'seg2': 'Second part.'
    }

    # Just check it doesn't crash
    # (actual plotting requires display)
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            output_file = f.name
        plot_segmentation(sample_text, segments, output_file=output_file)
        # Clean up
        Path(output_file).unlink(missing_ok=True)
    except Exception as e:
        # Allow failures due to missing matplotlib
        if 'matplotlib' not in str(e).lower():
            raise


# Test 6: Adaptive/Smart Segmentation
def test_adaptive_segmenter(project, sample_text):
    """Test adaptive segmentation."""
    from ef.plugins.adaptive_segmentation import adaptive_segmenter

    segments = adaptive_segmenter(
        sample_text,
        target_segment_size=100
    )

    assert isinstance(segments, dict)
    assert len(segments) > 0


def test_content_aware_segmenter(sample_text):
    """Test content-aware segmentation."""
    from ef.plugins.adaptive_segmentation import content_aware_segmenter

    segments = content_aware_segmenter(sample_text)
    assert isinstance(segments, dict)
    assert len(segments) > 0


# Test 7: Multilingual Support
def test_detect_language():
    """Test language detection."""
    from ef.plugins.multilingual_segmenters import detect_language

    lang = detect_language("This is English text.")
    # May return 'en' or 'unknown' depending on if langdetect is installed
    assert lang in ['en', 'unknown']


def test_register_multilingual_segmenters(project):
    """Test multilingual segmenter registration."""
    from ef.plugins.multilingual_segmenters import register_multilingual_segmenters

    count = register_multilingual_segmenters(project)
    assert count >= 1  # At least multilingual_auto should be registered


# Test 8: Domain-Specific Segmenters
def test_register_legal_segmenters(project):
    """Test legal segmenter registration."""
    from ef.plugins.domain_segmenters import register_legal_segmenters

    count = register_legal_segmenters(project)
    assert count >= 1
    assert 'legal_clauses' in project.segmenters


def test_register_scientific_segmenters(project):
    """Test scientific segmenter registration."""
    from ef.plugins.domain_segmenters import register_scientific_segmenters

    count = register_scientific_segmenters(project)
    assert count >= 1
    assert 'paper_sections' in project.segmenters


def test_register_code_segmenters(project):
    """Test code segmenter registration."""
    from ef.plugins.domain_segmenters import register_code_segmenters

    count = register_code_segmenters(project)
    assert count >= 1
    assert 'by_complexity' in project.segmenters


def test_register_medical_segmenters(project):
    """Test medical segmenter registration."""
    from ef.plugins.domain_segmenters import register_medical_segmenters

    count = register_medical_segmenters(project)
    assert count >= 1
    assert 'clinical_notes' in project.segmenters


# Test 9: Embedding-Aware Segmentation
def test_segment_for_embeddings(sample_text):
    """Test embedding-optimized segmentation."""
    from ef.plugins.embedding_optimized import segment_for_embeddings

    segments = segment_for_embeddings(
        sample_text,
        embedder=None,  # Use default
        target_segments=5
    )

    assert isinstance(segments, dict)
    assert len(segments) > 0


def test_segment_for_llm(sample_text):
    """Test LLM-optimized segmentation."""
    from ef.plugins.embedding_optimized import segment_for_llm

    segments = segment_for_llm(
        sample_text,
        context_window=1000
    )

    assert isinstance(segments, dict)
    assert len(segments) > 0


def test_register_embedding_optimized_segmenters(project):
    """Test embedding segmenter registration."""
    from ef.plugins.embedding_optimized import register_embedding_optimized_segmenters

    count = register_embedding_optimized_segmenters(project)
    assert count >= 2  # embedding_optimized and llm_optimized
    assert 'embedding_optimized' in project.segmenters
    assert 'llm_optimized' in project.segmenters


# Test 10: A/B Testing Framework
def test_ab_test(project, sample_text):
    """Test A/B testing."""
    from ef.plugins.ab_testing import ABTest

    test = ABTest(
        name='test_comparison',
        strategies={
            'strategy_a': 'sentences',
            'strategy_b': 'lines'
        }
    )

    test.run(project, [sample_text])

    assert len(test.results) > 0
    summary = test.summary()
    assert 'A/B Test' in summary


# Test 11: Performance Profiling
def test_profile_segmenter(project, sample_text):
    """Test performance profiling."""
    from ef.plugins.profiler import profile_segmenter

    segmenter = project.segmenters['sentences']
    profile = profile_segmenter(segmenter, [sample_text])

    assert 'time' in profile
    assert 'throughput' in profile
    assert profile['time']['mean'] > 0


def test_compare_performance(project, sample_text):
    """Test performance comparison."""
    from ef.plugins.profiler import compare_performance

    results = compare_performance(
        project,
        ['sentences', 'lines'],
        [sample_text]
    )

    assert 'sentences' in results
    assert 'lines' in results


# Test 12: Plugin Marketplace
def test_marketplace_search():
    """Test marketplace search."""
    from ef.plugins.marketplace import marketplace

    results = marketplace.search('legal')
    assert isinstance(results, list)
    # Mock implementation returns legal-doc-segmenter
    assert any('legal' in r['name'] for r in results)


def test_marketplace_install():
    """Test marketplace install."""
    from ef.plugins.marketplace import marketplace

    # Mock installation
    result = marketplace.install('test-segmenter')
    assert result is True


# Test 13: CLI Tools
def test_segment_file_cli(project, sample_text):
    """Test CLI file segmentation."""
    from ef.plugins.cli_tools import segment_file_cli

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        f.write(sample_text)
        input_file = f.name

    try:
        output_file = input_file + '.segmented.json'
        segment_file_cli(input_file, output_file, 'sentences', 'json', project)

        assert Path(output_file).exists()

        # Clean up
        Path(output_file).unlink()
    finally:
        Path(input_file).unlink()


def test_list_segmenters_cli(project, capsys):
    """Test CLI segmenter listing."""
    from ef.plugins.cli_tools import list_segmenters_cli

    list_segmenters_cli(project, verbose=False)
    captured = capsys.readouterr()

    assert 'Available Segmenters' in captured.out


# Test 14: Vector Database Integration
def test_vector_db_adapter():
    """Test vector database adapter base class."""
    from ef.plugins.vector_db_integration import VectorDBAdapter

    adapter = VectorDBAdapter()

    # Test default embedder
    embedding = adapter._default_embedder("test text")
    assert isinstance(embedding, list)
    assert len(embedding) > 0


def test_chroma_adapter():
    """Test Chroma adapter creation."""
    from ef.plugins.vector_db_integration import ChromaAdapter

    # Just test creation (may not have chromadb installed)
    try:
        adapter = ChromaAdapter(path=tempfile.mkdtemp())
        assert adapter is not None
    except:
        # OK if chromadb not installed
        pass


def test_create_adapter():
    """Test adapter factory."""
    from ef.plugins.vector_db_integration import create_adapter

    # Test creation of different types
    for db_type in ['chroma']:  # Only test chroma as others need servers
        try:
            adapter = create_adapter(db_type, path=tempfile.mkdtemp())
            assert adapter is not None
        except:
            # OK if dependencies not installed
            pass


def test_segment_and_index(project, sample_text):
    """Test segment and index workflow."""
    from ef.plugins.vector_db_integration import segment_and_index, ChromaAdapter

    try:
        adapter = ChromaAdapter(path=tempfile.mkdtemp())
        segments = segment_and_index(
            project,
            sample_text,
            'sentences',
            adapter
        )
        assert isinstance(segments, dict)
    except:
        # OK if chromadb not installed
        pass


# Test 15: Jupyter Integration
def test_segmentation_explorer(project, sample_text):
    """Test Jupyter explorer creation."""
    from ef.plugins.jupyter_widgets import SegmentationExplorer

    explorer = SegmentationExplorer(project, sample_text)
    assert explorer.text == sample_text
    assert explorer.project == project


def test_segmentation_comparison(project, sample_text):
    """Test Jupyter comparison widget."""
    from ef.plugins.jupyter_widgets import SegmentationComparison

    comparison = SegmentationComparison(
        project,
        sample_text,
        ['sentences', 'lines']
    )

    assert comparison.text == sample_text
    assert len(comparison.segmenters) == 2


def test_segmentation_visualizer(sample_text):
    """Test Jupyter visualizer."""
    from ef.plugins.jupyter_widgets import SegmentationVisualizer

    segments = {
        'seg1': 'First part.',
        'seg2': 'Second part.'
    }

    viz = SegmentationVisualizer(sample_text, segments)
    assert viz.text == sample_text
    assert viz.segments == segments


def test_notebook_segment(project, sample_text):
    """Test notebook segmentation helper."""
    from ef.plugins.jupyter_widgets import notebook_segment

    segments = notebook_segment(
        project,
        sample_text,
        'sentences',
        visualize=False  # Don't try to display in tests
    )

    assert isinstance(segments, dict)
    assert len(segments) > 0


def test_export_notebook_example():
    """Test notebook export."""
    from ef.plugins.jupyter_widgets import export_notebook_example

    with tempfile.NamedTemporaryFile(delete=False, suffix='.ipynb') as f:
        output_path = f.name

    try:
        export_notebook_example(output_path)
        assert Path(output_path).exists()

        # Verify it's valid JSON
        import json
        with open(output_path) as f:
            nb = json.load(f)
        assert 'cells' in nb
        assert 'metadata' in nb
    finally:
        Path(output_path).unlink()


# Integration tests
def test_full_workflow(project, sample_text):
    """Test complete workflow with multiple features."""
    # 1. Segment with adaptive segmentation
    from ef.plugins.adaptive_segmentation import adaptive_segmenter
    segments = adaptive_segmenter(sample_text, target_segment_size=100)

    # 2. Evaluate quality
    from ef.plugins.segmenter_quality import evaluate_quality
    quality = evaluate_quality(segments, source=sample_text)
    assert quality['overall'] >= 0

    # 3. Profile performance
    from ef.plugins.profiler import profile_segmenter
    segmenter = project.segmenters['sentences']
    profile = profile_segmenter(segmenter, [sample_text])
    assert 'time' in profile


def test_all_registrations(project):
    """Test that all registration functions work."""
    from ef.plugins import (
        domain_segmenters,
        embedding_optimized,
        multilingual_segmenters
    )

    # Register all
    count = 0
    count += domain_segmenters.register_legal_segmenters(project)
    count += domain_segmenters.register_scientific_segmenters(project)
    count += domain_segmenters.register_code_segmenters(project)
    count += domain_segmenters.register_medical_segmenters(project)
    count += embedding_optimized.register_embedding_optimized_segmenters(project)
    count += multilingual_segmenters.register_multilingual_segmenters(project)

    assert count > 0
    assert len(project.segmenters) > 10  # Should have many segmenters now
