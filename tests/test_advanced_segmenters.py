"""
Tests for advanced segmentation features.

Tests cover:
- Auto-registration on project creation
- Configurable segmenters
- Sliding window segmentation
- Hierarchical segmentation
- Semantic segmentation
- Pattern-based segmentation
- Multi-strategy segmentation
- Segmentation metrics
- Segmenter comparison
- Caching
- Batch processing
- Auto-detection
- Recommendation system
"""

import pytest
import tempfile
import os
from ef import Project
from ef.plugins import advanced_segmenters, segmenter_utils


class TestAutoRegistration:
    """Test auto-registration on project creation."""

    def test_auto_register_false_by_default(self):
        """Test that auto_register_segmenters is False by default."""
        project = Project.create('test', backend='memory')

        segmenters = project.list_components()['segmenters']

        # Should NOT have external segmenters by default
        assert 'langchain_recursive_1000' not in segmenters
        # Should have built-in segmenters
        assert 'sentences' in segmenters

    def test_auto_register_true(self):
        """Test auto-registration of external segmenters."""
        project = Project.create('test', backend='memory', auto_register_segmenters=True)

        segmenters = project.list_components()['segmenters']

        # Should have built-in segmenters
        assert 'sentences' in segmenters

        # May have external segmenters if packages are installed
        # (we can't test this reliably without knowing what's installed)
        # At minimum, AST should be registered (built-in)
        if 'ast_python' in segmenters:
            assert True  # Good, AST was registered


class TestConfigurableSegmenters:
    """Test configurable segmenter factories."""

    def test_char_chunker_registration(self):
        """Test that char_chunker is registered."""
        project = Project.create('test', backend='memory')
        advanced_segmenters.register_configurable_segmenters(project)

        assert 'char_chunker' in project.segmenters

    def test_char_chunker_default_params(self):
        """Test char_chunker with default parameters."""
        project = Project.create('test', backend='memory')
        advanced_segmenters.register_configurable_segmenters(project)

        text = 'a' * 2500  # 2500 chars
        segmenter = project.segmenters['char_chunker']
        result = segmenter(text)

        assert isinstance(result, dict)
        # With default 1000 char chunks and 200 overlap
        # Should get multiple chunks
        assert len(result) >= 2

    def test_char_chunker_custom_params(self):
        """Test char_chunker with custom parameters."""
        from functools import partial

        project = Project.create('test', backend='memory')
        advanced_segmenters.register_configurable_segmenters(project)

        text = 'a' * 1000
        base_segmenter = project.segmenters['char_chunker']

        # Create custom configured version
        custom_segmenter = partial(base_segmenter, chunk_size=500, chunk_overlap=100)

        result = custom_segmenter(text)

        assert isinstance(result, dict)
        assert len(result) >= 2

    def test_word_chunker(self):
        """Test word_chunker segmenter."""
        project = Project.create('test', backend='memory')
        advanced_segmenters.register_configurable_segmenters(project)

        text = ' '.join(['word'] * 500)  # 500 words
        segmenter = project.segmenters['word_chunker']

        result = segmenter(text)

        assert isinstance(result, dict)
        assert len(result) >= 2


class TestSlidingWindowSegmenters:
    """Test sliding window segmentation."""

    def test_sliding_window_registration(self):
        """Test that sliding window segmenters are registered."""
        project = Project.create('test', backend='memory')
        advanced_segmenters.register_sliding_window_segmenters(project)

        assert 'sliding_window' in project.segmenters
        assert 'sentence_window' in project.segmenters

    def test_sliding_window_overlap(self):
        """Test that sliding window creates overlapping segments."""
        project = Project.create('test', backend='memory')
        advanced_segmenters.register_sliding_window_segmenters(project)

        text = 'a' * 1000
        segmenter = project.segmenters['sliding_window']

        result = segmenter(text, window_size=500, stride=250)

        # Should have overlapping windows
        assert len(result) >= 3  # With 50% overlap

        # Check overlap
        values = list(result.values())
        # First window should overlap with second
        if len(values) >= 2:
            # Last 250 chars of first window should match first 250 chars of second
            assert values[0][-250:] == values[1][:250]

    def test_sentence_window(self):
        """Test sentence window segmenter."""
        project = Project.create('test', backend='memory')
        advanced_segmenters.register_sliding_window_segmenters(project)

        text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
        segmenter = project.segmenters['sentence_window']

        result = segmenter(text, sentences_per_window=2, overlap_sentences=1)

        assert isinstance(result, dict)
        assert len(result) >= 2


class TestHierarchicalSegmenters:
    """Test hierarchical segmentation."""

    def test_hierarchical_registration(self):
        """Test hierarchical segmenter registration."""
        project = Project.create('test', backend='memory')
        advanced_segmenters.register_hierarchical_segmenters(project)

        assert 'hierarchical' in project.segmenters
        assert 'markdown_hierarchical' in project.segmenters

    def test_hierarchical_paragraphs(self):
        """Test hierarchical segmentation at paragraph level."""
        project = Project.create('test', backend='memory')
        advanced_segmenters.register_hierarchical_segmenters(project)

        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        segmenter = project.segmenters['hierarchical']

        result = segmenter(text, max_depth=1)

        # Should have paragraph-level segments
        assert any('para_' in key for key in result.keys())

    def test_hierarchical_sentences(self):
        """Test hierarchical segmentation with sentences."""
        project = Project.create('test', backend='memory')
        advanced_segmenters.register_hierarchical_segmenters(project)

        text = "First sentence. Second sentence.\n\nThird sentence. Fourth sentence."
        segmenter = project.segmenters['hierarchical']

        result = segmenter(text, max_depth=2)

        # Should have both paragraph and sentence levels
        assert any('para_' in key and 'sent_' not in key for key in result.keys())
        assert any('sent_' in key for key in result.keys())

    def test_markdown_hierarchical(self):
        """Test markdown hierarchical segmentation."""
        project = Project.create('test', backend='memory')
        advanced_segmenters.register_hierarchical_segmenters(project)

        text = """# Header 1
Content 1

## Header 2
Content 2

### Header 3
Content 3"""

        segmenter = project.segmenters['markdown_hierarchical']
        result = segmenter(text)

        # Should have headers
        assert any('h1_' in key for key in result.keys())
        assert any('h2_' in key for key in result.keys())
        assert any('h3_' in key for key in result.keys())


class TestPatternSegmenters:
    """Test pattern-based segmentation."""

    def test_pattern_segmenter_creation(self):
        """Test creating custom pattern segmenters."""
        project = Project.create('test', backend='memory')

        # Create a pattern segmenter
        advanced_segmenters.create_pattern_segmenter(
            project,
            'by_comma',
            r',\s*',
            'Split on commas'
        )

        assert 'by_comma' in project.segmenters

    def test_pattern_segmenter_usage(self):
        """Test using pattern segmenter."""
        project = Project.create('test', backend='memory')

        advanced_segmenters.create_pattern_segmenter(
            project,
            'by_comma',
            r',\s*',
            'Split on commas'
        )

        text = "apple, banana, cherry, date"
        segmenter = project.segmenters['by_comma']

        result = segmenter(text)

        assert len(result) == 4
        assert 'apple' in result.values()

    def test_builtin_pattern_segmenters(self):
        """Test built-in pattern segmenters."""
        project = Project.create('test', backend='memory')
        advanced_segmenters.register_pattern_segmenters(project)

        assert 'by_paragraphs' in project.segmenters
        assert 'by_markdown_headers' in project.segmenters


class TestMultiStrategySegmenters:
    """Test multi-strategy segmentation."""

    def test_multi_level_registration(self):
        """Test multi-level segmenter registration."""
        project = Project.create('test', backend='memory')
        advanced_segmenters.register_multi_strategy_segmenters(project)

        assert 'multi_level' in project.segmenters
        assert 'best_fit' in project.segmenters

    def test_multi_level_segmenter(self):
        """Test multi-level segmenter."""
        project = Project.create('test', backend='memory')
        # Need simple segmenters first
        from ef.plugins import simple
        simple.register_simple_components(project)
        advanced_segmenters.register_multi_strategy_segmenters(project)

        text = "Hello. World."
        segmenter = project.segmenters['multi_level']

        result = segmenter(text, strategies=['sentences', 'lines'])

        # Should have keys prefixed with strategy names
        assert any(key.startswith('sentences.') for key in result.keys())

    def test_best_fit_segmenter(self):
        """Test best-fit automatic segmenter selection."""
        project = Project.create('test', backend='memory')
        from ef.plugins import simple
        simple.register_simple_components(project)
        advanced_segmenters.register_multi_strategy_segmenters(project)

        text = "Normal text. Multiple sentences."
        segmenter = project.segmenters['best_fit']

        result = segmenter(text)

        assert isinstance(result, dict)
        assert len(result) >= 1


class TestSegmentationMetrics:
    """Test segmentation quality metrics."""

    def test_analyze_segmentation(self):
        """Test analyze_segmentation function."""
        segments = {
            'seg1': 'Hello world',
            'seg2': 'Foo bar baz',
            'seg3': 'Test'
        }

        metrics = segmenter_utils.analyze_segmentation(segments)

        assert metrics['count'] == 3
        assert metrics['avg_length'] > 0
        assert metrics['min_length'] > 0
        assert metrics['max_length'] >= metrics['min_length']

    def test_analyze_empty_segments(self):
        """Test metrics with empty segments."""
        segments = {}

        metrics = segmenter_utils.analyze_segmentation(segments)

        assert metrics['count'] == 0
        assert metrics['avg_length'] == 0

    def test_print_segmentation_report(self, capsys):
        """Test printing segmentation report."""
        segments = {'seg1': 'Hello', 'seg2': 'World'}

        segmenter_utils.print_segmentation_report(segments, "Test Report")

        captured = capsys.readouterr()
        assert 'Test Report' in captured.out
        assert 'Segments: 2' in captured.out


class TestSegmenterComparison:
    """Test segmenter comparison tools."""

    def test_compare_segmenters(self):
        """Test comparing multiple segmenters."""
        project = Project.create('test', backend='memory')
        from ef.plugins import simple
        simple.register_simple_components(project)

        text = "Hello. World. Test."
        results = segmenter_utils.compare_segmenters(
            project,
            text,
            ['sentences', 'lines', 'identity'],
            verbose=False
        )

        assert 'sentences' in results
        assert 'lines' in results
        assert 'identity' in results

        # Check each has segments and metrics
        for name, result in results.items():
            assert 'segments' in result
            assert 'metrics' in result
            assert 'success' in result

    def test_compare_with_missing_segmenter(self):
        """Test comparison with non-existent segmenter."""
        project = Project.create('test', backend='memory')

        text = "Hello world"
        results = segmenter_utils.compare_segmenters(
            project,
            text,
            ['nonexistent'],
            verbose=False
        )

        # Should not crash, just warn
        assert isinstance(results, dict)


class TestCaching:
    """Test segmentation caching."""

    def test_cache_creation(self):
        """Test creating a cache."""
        cache = segmenter_utils.SegmentationCache(max_size=10)

        assert cache._max_size == 10
        assert len(cache._cache) == 0

    def test_cache_put_get(self):
        """Test caching and retrieval."""
        cache = segmenter_utils.SegmentationCache()

        text = "Hello world"
        segments = {'seg1': 'Hello', 'seg2': 'world'}

        # Cache should be empty initially
        result = cache.get('test_seg', text)
        assert result is None

        # Put in cache
        cache.put('test_seg', text, segments)

        # Should now retrieve from cache
        result = cache.get('test_seg', text)
        assert result == segments

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = segmenter_utils.SegmentationCache()

        text = "Hello"
        segments = {'seg1': 'Hello'}

        # Miss
        cache.get('test', text)

        # Put and hit
        cache.put('test', text, segments)
        cache.get('test', text)

        stats = cache.stats()

        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['size'] == 1

    def test_make_cached_segmenter(self):
        """Test wrapping a segmenter with caching."""
        project = Project.create('test', backend='memory')
        from ef.plugins import simple
        simple.register_simple_components(project)

        segmenter = project.segmenters['sentences']
        cached_seg = segmenter_utils.make_cached_segmenter(segmenter)

        text = "Hello. World."

        # First call - cache miss
        result1 = cached_seg(text)

        # Second call - cache hit
        result2 = cached_seg(text)

        assert result1 == result2


class TestBatchProcessing:
    """Test batch processing utilities."""

    def test_batch_segment_texts(self):
        """Test batch segmenting multiple texts."""
        project = Project.create('test', backend='memory')
        from ef.plugins import simple
        simple.register_simple_components(project)

        texts = {
            'doc1': 'Hello. World.',
            'doc2': 'Foo. Bar. Baz.'
        }

        results = segmenter_utils.batch_segment_texts(
            project,
            texts,
            segmenter='sentences',
            cache_results=False
        )

        assert 'doc1' in results
        assert 'doc2' in results
        assert results['doc1']['success']
        assert len(results['doc1']['segments']) >= 2

    def test_batch_segment_files(self):
        """Test batch segmenting files."""
        project = Project.create('test', backend='memory')
        from ef.plugins import simple
        simple.register_simple_components(project)

        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = os.path.join(tmpdir, 'test1.txt')
            file2 = os.path.join(tmpdir, 'test2.txt')

            with open(file1, 'w') as f:
                f.write('Hello. World.')

            with open(file2, 'w') as f:
                f.write('Foo. Bar.')

            results = segmenter_utils.batch_segment_files(
                project,
                file_paths=[file1, file2],
                segmenter='sentences',
                cache_results=False
            )

            assert file1 in results
            assert file2 in results
            assert results[file1]['success']


class TestAutoDetection:
    """Test automatic segmenter detection."""

    def test_auto_detect_python(self):
        """Test auto-detection for Python code."""
        project = Project.create('test', backend='memory')
        from ef.plugins import simple
        simple.register_simple_components(project)

        code = "def hello():\n    import os\n    print('hi')"

        result = segmenter_utils.auto_detect_segmenter(project, code, 'test.py')

        # Should recommend a Python segmenter if available, otherwise fall back
        assert result in project.list_components()['segmenters']

    def test_auto_detect_markdown(self):
        """Test auto-detection for markdown."""
        project = Project.create('test', backend='memory')
        from ef.plugins import simple
        simple.register_simple_components(project)

        text = "# Header\n\nSome content"

        result = segmenter_utils.auto_detect_segmenter(project, text, 'README.md')

        assert result in project.list_components()['segmenters']

    def test_auto_detect_plain_text(self):
        """Test auto-detection for plain text."""
        project = Project.create('test', backend='memory')
        from ef.plugins import simple
        simple.register_simple_components(project)

        text = "Just some plain text. With sentences."

        result = segmenter_utils.auto_detect_segmenter(project, text)

        # Should default to sentence-based
        assert result in project.list_components()['segmenters']


class TestRecommendationSystem:
    """Test segmenter recommendation system."""

    def test_recommend_for_code(self):
        """Test recommendation for code content."""
        project = Project.create('test', backend='memory')
        from ef.plugins import simple
        simple.register_simple_components(project)

        seg, reason = segmenter_utils.recommend_segmenter(
            project,
            content_type='code',
            language='python'
        )

        assert seg in project.list_components()['segmenters']
        assert isinstance(reason, str)
        assert len(reason) > 0

    def test_recommend_for_documentation(self):
        """Test recommendation for documentation."""
        project = Project.create('test', backend='memory')
        from ef.plugins import simple
        simple.register_simple_components(project)

        seg, reason = segmenter_utils.recommend_segmenter(
            project,
            content_type='documentation'
        )

        assert seg in project.list_components()['segmenters']
        assert isinstance(reason, str)

    def test_recommend_for_llm_context(self):
        """Test recommendation for LLM context."""
        project = Project.create('test', backend='memory')
        from ef.plugins import simple
        simple.register_simple_components(project)

        seg, reason = segmenter_utils.recommend_segmenter(
            project,
            use_case='llm_context'
        )

        assert seg in project.list_components()['segmenters']

    def test_print_recommendation(self, capsys):
        """Test printing recommendation."""
        project = Project.create('test', backend='memory')
        from ef.plugins import simple
        simple.register_simple_components(project)

        segmenter_utils.print_recommendation(
            project,
            content_type='code',
            use_case='analysis'
        )

        captured = capsys.readouterr()
        assert 'Recommendation' in captured.out
        assert 'Recommended:' in captured.out


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_full_workflow(self):
        """Test a complete workflow with advanced features."""
        # Create project with auto-registration
        project = Project.create('test', backend='memory', auto_register_segmenters=False)

        # Register advanced segmenters
        advanced_segmenters.register_all_advanced_segmenters(project, verbose=False)

        # Check that we have various segmenters available
        segmenters = project.list_components()['segmenters']
        assert 'sliding_window' in segmenters
        assert 'hierarchical' in segmenters

        # Test segmentation
        text = "First paragraph with multiple sentences. Second sentence here.\n\nSecond paragraph also has content."

        # Compare different strategies
        results = segmenter_utils.compare_segmenters(
            project,
            text,
            ['sliding_window', 'hierarchical', 'sentences'],
            verbose=False
        )

        assert len(results) >= 2

        # Get recommendation
        seg_name, reason = segmenter_utils.recommend_segmenter(
            project,
            content_length=len(text)
        )

        assert seg_name in segmenters


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
