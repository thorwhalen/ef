"""
Utilities for segmentation analysis and optimization.

This module provides:
- Segmentation quality metrics
- Segmenter comparison tools
- Caching for expensive segmentation
- Batch processing helpers
- Auto-detection of optimal segmenters
- Recommendation system
"""

from typing import Any, Optional, Callable
from collections.abc import Mapping
import hashlib
import glob as glob_module
import os
import statistics
import warnings
import re


# ============================================================================
# Segmentation Quality Metrics
# ============================================================================


def analyze_segmentation(segments: dict[str, str]) -> dict[str, Any]:
    """
    Analyze segmentation quality and return metrics.

    Metrics include:
    - Count: Number of segments
    - Length statistics: mean, std_dev, min, max
    - Distribution: Histogram of lengths

    Args:
        segments: Dict of segment key -> text

    Returns:
        Dict of metrics

    Example:
        >>> segments = {'seg1': 'Hello world', 'seg2': 'Foo bar baz'}
        >>> metrics = analyze_segmentation(segments)
        >>> metrics['count']
        2
    """
    if not segments:
        return {
            'count': 0,
            'avg_length': 0,
            'std_dev': 0,
            'min_length': 0,
            'max_length': 0,
            'total_chars': 0,
            'avg_words': 0,
        }

    lengths = [len(seg) for seg in segments.values()]
    word_counts = [len(seg.split()) for seg in segments.values()]

    metrics = {
        'count': len(segments),
        'total_chars': sum(lengths),
        'avg_length': statistics.mean(lengths),
        'std_dev': statistics.stdev(lengths) if len(lengths) > 1 else 0,
        'min_length': min(lengths),
        'max_length': max(lengths),
        'median_length': statistics.median(lengths),
        'avg_words': statistics.mean(word_counts),
        'total_words': sum(word_counts),
    }

    # Add length distribution (histogram)
    if lengths:
        # Create 5 bins
        min_len, max_len = min(lengths), max(lengths)
        if max_len > min_len:
            bin_size = (max_len - min_len) / 5
            bins = [min_len + i * bin_size for i in range(6)]
            distribution = {f'bin_{i}': 0 for i in range(5)}

            for length in lengths:
                for i in range(5):
                    if bins[i] <= length < bins[i + 1]:
                        distribution[f'bin_{i}'] += 1
                        break
                else:
                    # Handle max value
                    distribution['bin_4'] += 1

            metrics['distribution'] = distribution

    return metrics


def print_segmentation_report(segments: dict[str, str], name: str = "Segmentation") -> None:
    """
    Print a formatted report of segmentation metrics.

    Args:
        segments: Dict of segments
        name: Name for the report

    Example:
        >>> segments = {'s1': 'Hello', 's2': 'World'}
        >>> print_segmentation_report(segments, "My Segmentation")
    """
    metrics = analyze_segmentation(segments)

    print(f"\n{'='*60}")
    print(f"{name} Report")
    print("="*60)
    print(f"Segments: {metrics['count']}")
    print(f"Total characters: {metrics['total_chars']}")
    print(f"Total words: {metrics['total_words']}")
    print(f"\nLength Statistics:")
    print(f"  Mean: {metrics['avg_length']:.1f} chars ({metrics['avg_words']:.1f} words)")
    print(f"  Std Dev: {metrics['std_dev']:.1f}")
    print(f"  Min: {metrics['min_length']} chars")
    print(f"  Max: {metrics['max_length']} chars")
    print(f"  Median: {metrics['median_length']:.1f} chars")

    if 'distribution' in metrics:
        print(f"\nDistribution:")
        for bin_name, count in metrics['distribution'].items():
            print(f"  {bin_name}: {count} segments")

    print("="*60 + "\n")


# ============================================================================
# Segmenter Comparison
# ============================================================================


def compare_segmenters(
    project,
    source: Any,
    segmenter_names: list[str],
    verbose: bool = True
) -> dict[str, dict]:
    """
    Compare how different segmenters handle the same text.

    Args:
        project: Project instance
        source: Source text to segment
        segmenter_names: List of segmenter names to compare
        verbose: Whether to print comparison report

    Returns:
        Dict mapping segmenter names to their results and metrics

    Example:
        >>> results = compare_segmenters(
        ...     project,
        ...     "Some text. More text.",
        ...     ['sentences', 'lines', 'identity']
        ... )
    """
    results = {}

    for name in segmenter_names:
        if name not in project.segmenters:
            warnings.warn(f"Segmenter '{name}' not found, skipping")
            continue

        try:
            segmenter = project.segmenters[name]
            segments = segmenter(source)
            metrics = analyze_segmentation(segments)

            results[name] = {
                'segments': segments,
                'metrics': metrics,
                'success': True
            }
        except Exception as e:
            results[name] = {
                'segments': {},
                'metrics': {},
                'success': False,
                'error': str(e)
            }
            warnings.warn(f"Segmenter '{name}' failed: {e}")

    if verbose:
        print_comparison_report(results)

    return results


def print_comparison_report(results: dict[str, dict]) -> None:
    """
    Print a formatted comparison report.

    Args:
        results: Results from compare_segmenters()
    """
    print("\n" + "="*70)
    print("Segmenter Comparison Report")
    print("="*70)

    # Print summary table
    print(f"\n{'Segmenter':<25} {'Segments':<12} {'Avg Length':<15} {'Status':<10}")
    print("-"*70)

    for name, result in results.items():
        if result['success']:
            metrics = result['metrics']
            seg_count = metrics['count']
            avg_len = f"{metrics['avg_length']:.1f} chars"
            status = "✓ Success"
        else:
            seg_count = "N/A"
            avg_len = "N/A"
            status = "✗ Failed"

        print(f"{name:<25} {str(seg_count):<12} {avg_len:<15} {status:<10}")

    print("="*70 + "\n")


# ============================================================================
# Segmentation Caching
# ============================================================================


class SegmentationCache:
    """
    Cache for segmentation results.

    Caches based on content hash to avoid re-segmenting identical text.
    """

    def __init__(self, max_size: int = 1000):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached results
        """
        self._cache = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _hash_source(self, source: Any) -> str:
        """Create hash key from source."""
        if isinstance(source, dict):
            source_str = str(sorted(source.items()))
        else:
            source_str = str(source)

        return hashlib.md5(source_str.encode()).hexdigest()

    def get(self, segmenter_name: str, source: Any) -> Optional[dict[str, str]]:
        """
        Get cached result if available.

        Args:
            segmenter_name: Name of segmenter
            source: Source text

        Returns:
            Cached segments or None
        """
        cache_key = f"{segmenter_name}:{self._hash_source(source)}"

        if cache_key in self._cache:
            self._hits += 1
            return self._cache[cache_key]
        else:
            self._misses += 1
            return None

    def put(self, segmenter_name: str, source: Any, segments: dict[str, str]) -> None:
        """
        Store result in cache.

        Args:
            segmenter_name: Name of segmenter
            source: Source text
            segments: Segmentation result
        """
        # Implement simple LRU by removing oldest if at capacity
        if len(self._cache) >= self._max_size:
            # Remove first item (oldest)
            first_key = next(iter(self._cache))
            del self._cache[first_key]

        cache_key = f"{segmenter_name}:{self._hash_source(source)}"
        self._cache[cache_key] = segments

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0

        return {
            'size': len(self._cache),
            'max_size': self._max_size,
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
        }

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0


# Global cache instance
_global_cache = SegmentationCache()


def make_cached_segmenter(segmenter_func: Callable, cache: Optional[SegmentationCache] = None) -> Callable:
    """
    Wrap a segmenter with caching.

    Args:
        segmenter_func: Segmenter function to wrap
        cache: Cache instance (uses global cache if None)

    Returns:
        Cached version of the segmenter

    Example:
        >>> from ef import Project
        >>> project = Project.create('test')
        >>> cached_seg = make_cached_segmenter(project.segmenters['sentences'])
        >>> # First call computes
        >>> result1 = cached_seg("Hello. World.")
        >>> # Second call uses cache
        >>> result2 = cached_seg("Hello. World.")
    """
    if cache is None:
        cache = _global_cache

    def cached_segmenter(source: Any) -> dict[str, str]:
        """Cached segmenter wrapper."""
        segmenter_name = getattr(segmenter_func, '__name__', 'unknown')

        # Try to get from cache
        cached_result = cache.get(segmenter_name, source)
        if cached_result is not None:
            return cached_result

        # Compute and cache
        result = segmenter_func(source)
        cache.put(segmenter_name, source, result)

        return result

    # Preserve function metadata
    cached_segmenter.__name__ = f"cached_{getattr(segmenter_func, '__name__', 'segmenter')}"
    cached_segmenter.__doc__ = segmenter_func.__doc__

    return cached_segmenter


def get_cache_stats() -> dict[str, Any]:
    """Get statistics for the global cache."""
    return _global_cache.stats()


def clear_cache() -> None:
    """Clear the global segmentation cache."""
    _global_cache.clear()


# ============================================================================
# Batch Processing
# ============================================================================


def batch_segment_files(
    project,
    file_paths: Optional[list[str]] = None,
    pattern: Optional[str] = None,
    segmenter: str = 'auto',
    recursive: bool = True,
    cache_results: bool = True
) -> dict[str, dict[str, str]]:
    """
    Batch process multiple files with appropriate segmenters.

    Args:
        project: Project instance
        file_paths: List of file paths (mutually exclusive with pattern)
        pattern: Glob pattern for files (e.g., '**/*.py')
        segmenter: Segmenter name or 'auto' for auto-detection
        recursive: Whether to search recursively (for patterns)
        cache_results: Whether to cache segmentation results

    Returns:
        Dict mapping file paths to segmentation results

    Example:
        >>> results = batch_segment_files(
        ...     project,
        ...     pattern='**/*.py',
        ...     segmenter='ast_python'
        ... )
    """
    # Collect files
    if file_paths is None and pattern is None:
        raise ValueError("Must provide either file_paths or pattern")

    if file_paths is not None and pattern is not None:
        raise ValueError("Cannot provide both file_paths and pattern")

    if pattern is not None:
        file_paths = glob_module.glob(pattern, recursive=recursive)

    results = {}
    cache = SegmentationCache() if cache_results else None

    for filepath in file_paths:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Determine segmenter
            if segmenter == 'auto':
                seg_name = auto_detect_segmenter(project, content, filepath)
            else:
                seg_name = segmenter

            # Get segmenter function
            if seg_name not in project.segmenters:
                warnings.warn(f"Segmenter '{seg_name}' not found for {filepath}, using 'identity'")
                seg_name = 'identity'

            seg_func = project.segmenters[seg_name]

            # Apply caching if requested
            if cache_results:
                seg_func = make_cached_segmenter(seg_func, cache)

            # Segment
            segments = seg_func(content)
            results[filepath] = {
                'segments': segments,
                'segmenter': seg_name,
                'success': True
            }

        except Exception as e:
            results[filepath] = {
                'segments': {},
                'segmenter': seg_name if segmenter != 'auto' else 'unknown',
                'success': False,
                'error': str(e)
            }
            warnings.warn(f"Failed to process {filepath}: {e}")

    return results


def batch_segment_texts(
    project,
    texts: dict[str, str],
    segmenter: str = 'auto',
    cache_results: bool = True
) -> dict[str, dict[str, str]]:
    """
    Batch segment multiple texts.

    Args:
        project: Project instance
        texts: Dict mapping identifiers to text content
        segmenter: Segmenter name or 'auto'
        cache_results: Whether to cache results

    Returns:
        Dict mapping identifiers to segmentation results

    Example:
        >>> texts = {'doc1': 'Hello. World.', 'doc2': 'Foo. Bar.'}
        >>> results = batch_segment_texts(project, texts, segmenter='sentences')
    """
    results = {}
    cache = SegmentationCache() if cache_results else None

    for text_id, content in texts.items():
        try:
            # Determine segmenter
            if segmenter == 'auto':
                seg_name = auto_detect_segmenter(project, content)
            else:
                seg_name = segmenter

            # Get segmenter
            if seg_name not in project.segmenters:
                warnings.warn(f"Segmenter '{seg_name}' not found, using 'identity'")
                seg_name = 'identity'

            seg_func = project.segmenters[seg_name]

            # Apply caching
            if cache_results:
                seg_func = make_cached_segmenter(seg_func, cache)

            # Segment
            segments = seg_func(content)
            results[text_id] = {
                'segments': segments,
                'segmenter': seg_name,
                'success': True
            }

        except Exception as e:
            results[text_id] = {
                'segments': {},
                'segmenter': seg_name if segmenter != 'auto' else 'unknown',
                'success': False,
                'error': str(e)
            }

    return results


# ============================================================================
# Auto-Detection
# ============================================================================


def auto_detect_segmenter(
    project,
    content: str,
    filename: Optional[str] = None
) -> str:
    """
    Automatically detect the best segmenter for content.

    Detection based on:
    - File extension (if filename provided)
    - Content patterns (code, markdown, etc.)
    - Content length

    Args:
        project: Project instance
        content: Text content
        filename: Optional filename for extension-based detection

    Returns:
        Name of recommended segmenter

    Example:
        >>> segmenter = auto_detect_segmenter(project, "def foo(): pass", "code.py")
        >>> print(segmenter)
        'ast_python'
    """
    available = project.list_components()['segmenters']

    # File extension-based detection
    if filename:
        ext = os.path.splitext(filename)[1].lower()

        ext_map = {
            '.py': ['tree_sitter_python', 'ast_python'],
            '.js': ['tree_sitter_javascript', 'langchain_js'],
            '.ts': ['tree_sitter_typescript', 'langchain_ts'],
            '.java': ['tree_sitter_java', 'langchain_java'],
            '.cpp': ['tree_sitter_cpp', 'langchain_cpp'],
            '.cc': ['tree_sitter_cpp', 'langchain_cpp'],
            '.go': ['tree_sitter_go', 'langchain_go'],
            '.rs': ['tree_sitter_rust', 'langchain_rust'],
            '.md': ['langchain_markdown', 'markdown_hierarchical'],
            '.html': ['langchain_html'],
        }

        if ext in ext_map:
            # Return first available segmenter for this extension
            for seg in ext_map[ext]:
                if seg in available:
                    return seg

    # Content-based detection
    # Python code
    if re.search(r'\bdef\s+\w+\s*\(', content) and re.search(r'\bimport\s+\w+', content):
        for seg in ['tree_sitter_python', 'ast_python']:
            if seg in available:
                return seg

    # JavaScript/TypeScript
    if re.search(r'\bfunction\s+\w+\s*\(', content) or re.search(r'\bconst\s+\w+\s*=', content):
        for seg in ['tree_sitter_javascript', 'langchain_js']:
            if seg in available:
                return seg

    # Markdown headers
    if re.search(r'^#+\s+', content, re.MULTILINE):
        for seg in ['langchain_markdown', 'markdown_hierarchical']:
            if seg in available:
                return seg

    # Code blocks (markdown with code)
    if '```' in content:
        for seg in ['langchain_markdown']:
            if seg in available:
                return seg

    # Length-based defaults
    content_length = len(content)

    if content_length > 10000:
        # Long documents - use chunking
        for seg in ['langchain_recursive_2000', 'sliding_window', 'by_paragraphs']:
            if seg in available:
                return seg
    elif content_length > 2000:
        for seg in ['langchain_recursive_1000', 'by_paragraphs']:
            if seg in available:
                return seg

    # Default strategies
    for seg in ['spacy_sentences', 'nltk_sentences', 'sentences']:
        if seg in available:
            return seg

    # Final fallback
    return 'identity' if 'identity' in available else available[0] if available else 'identity'


# ============================================================================
# Recommendation System
# ============================================================================


def recommend_segmenter(
    project,
    content_type: Optional[str] = None,
    use_case: Optional[str] = None,
    content_length: Optional[int] = None,
    language: Optional[str] = None,
    **kwargs
) -> tuple[str, str]:
    """
    Recommend the best segmenter based on use case and content type.

    Args:
        project: Project instance
        content_type: Type of content ('code', 'documentation', 'article', 'chat', etc.)
        use_case: Intended use ('llm_context', 'semantic_search', 'analysis', etc.)
        content_length: Approximate content length
        language: Programming language (for code content)
        **kwargs: Additional context

    Returns:
        Tuple of (segmenter_name, reason)

    Example:
        >>> seg, reason = recommend_segmenter(
        ...     project,
        ...     content_type='code',
        ...     language='python'
        ... )
        >>> print(f"Use {seg}: {reason}")
        'Use tree_sitter_python: Best for Python code structure'
    """
    available = project.list_components()['segmenters']

    # Code segmentation
    if content_type == 'code':
        if language:
            lang_lower = language.lower()

            # Try tree-sitter first
            ts_name = f'tree_sitter_{lang_lower}'
            if ts_name in available:
                return ts_name, f'Best for {language} code structure analysis'

            # Try LangChain
            lc_name = f'langchain_{lang_lower}'
            if lc_name in available:
                return lc_name, f'Good for {language} code segmentation'

            # Python-specific fallback
            if lang_lower == 'python' and 'ast_python' in available:
                return 'ast_python', 'Built-in Python AST-based segmentation'

        return 'identity', 'No specialized code segmenter available for language'

    # LLM context preparation
    if use_case == 'llm_context':
        if 'tiktoken_gpt-4' in available:
            return 'tiktoken_gpt-4', 'Respects GPT-4 token limits (8K tokens)'
        if 'tiktoken_gpt-3_5' in available:
            return 'tiktoken_gpt-3_5', 'Respects GPT-3.5 token limits (4K tokens)'
        if 'char_chunker' in available:
            return 'char_chunker', 'Character-based chunking for LLM contexts'

    # Documentation
    if content_type == 'documentation':
        if 'langchain_markdown' in available:
            return 'langchain_markdown', 'Respects markdown document structure'
        if 'markdown_hierarchical' in available:
            return 'markdown_hierarchical', 'Hierarchical markdown segmentation'
        if 'by_paragraphs' in available:
            return 'by_paragraphs', 'Paragraph-based segmentation'

    # Semantic search
    if use_case == 'semantic_search':
        if 'semantic_similarity' in available:
            return 'semantic_similarity', 'Segments by semantic coherence'
        if 'sliding_window' in available:
            return 'sliding_window', 'Overlapping windows preserve context'
        if 'spacy_sentences' in available:
            return 'spacy_sentences', 'Sentence-level for semantic embedding'

    # Length-based recommendations
    if content_length:
        if content_length > 50000:
            if 'langchain_recursive_2000' in available:
                return 'langchain_recursive_2000', 'Large chunks for very long documents'
        elif content_length > 10000:
            if 'langchain_recursive_1000' in available:
                return 'langchain_recursive_1000', 'Medium chunks for long documents'
        elif content_length > 2000:
            if 'by_paragraphs' in available:
                return 'by_paragraphs', 'Paragraph-level for medium texts'
        else:
            if 'spacy_sentences' in available:
                return 'spacy_sentences', 'Sentence-level for short texts'

    # Analysis use case
    if use_case == 'analysis':
        if 'hierarchical' in available:
            return 'hierarchical', 'Multi-level analysis with hierarchical segments'
        if 'multi_level' in available:
            return 'multi_level', 'Multiple segmentation strategies for comparison'

    # Default recommendation
    if 'spacy_sentences' in available:
        return 'spacy_sentences', 'General-purpose NLP-based sentence segmentation'
    if 'sentences' in available:
        return 'sentences', 'Simple sentence-based segmentation'

    return 'identity', 'No specialized segmenter available, using identity'


def print_recommendation(
    project,
    content_type: Optional[str] = None,
    use_case: Optional[str] = None,
    **kwargs
) -> None:
    """
    Print a recommendation with explanation.

    Args:
        project: Project instance
        content_type: Type of content
        use_case: Intended use case
        **kwargs: Additional context
    """
    segmenter, reason = recommend_segmenter(
        project,
        content_type=content_type,
        use_case=use_case,
        **kwargs
    )

    print("\n" + "="*60)
    print("Segmenter Recommendation")
    print("="*60)
    print(f"Content Type: {content_type or 'Not specified'}")
    print(f"Use Case: {use_case or 'Not specified'}")
    print(f"\nRecommended: {segmenter}")
    print(f"Reason: {reason}")
    print("="*60 + "\n")
