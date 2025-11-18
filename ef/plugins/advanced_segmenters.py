"""
Advanced segmentation strategies for ef.

This module provides sophisticated segmentation approaches including:
- Configurable segmenter factories
- Sliding window segmentation
- Hierarchical segmentation
- Semantic/embedding-based segmentation
- Pattern-based segmentation
- Multi-strategy segmentation
"""

from typing import Any, Callable, Optional
from functools import partial
import re
import warnings


# ============================================================================
# Configurable Segmenter Factories
# ============================================================================


def register_configurable_segmenters(project) -> int:
    """
    Register configurable segmenter factories that can be customized at runtime.

    These segmenters accept parameters when used in pipelines, allowing
    flexible configuration without pre-registering every variant.

    Args:
        project: Project instance with segmenters registry

    Returns:
        Number of segmenters registered

    Example:
        >>> from ef import Project
        >>> project = Project.create('test', backend='memory')
        >>> register_configurable_segmenters(project)
        >>> # Use with functools.partial to configure
        >>> from functools import partial
        >>> custom_chunker = partial(project.segmenters['char_chunker'], chunk_size=500)
    """
    count = 0

    @project.segmenters.register(
        'char_chunker',
        package='ef.plugins.advanced',
        description='Configurable character-based chunking'
    )
    def char_chunker(source: Any, chunk_size: int = 1000, chunk_overlap: int = 200) -> dict[str, str]:
        """
        Chunk text by character count with configurable size and overlap.

        Args:
            source: Text or dict of text
            chunk_size: Maximum characters per chunk
            chunk_overlap: Overlap between consecutive chunks

        Returns:
            Dict of segments
        """
        if isinstance(source, dict):
            source = '\n\n'.join(source.values())

        segments = {}
        start = 0
        chunk_num = 0

        while start < len(source):
            end = start + chunk_size
            chunk = source[start:end]

            if chunk.strip():
                segments[f'chunk_{chunk_num}'] = chunk
                chunk_num += 1

            # Move start forward by (chunk_size - overlap)
            start = start + chunk_size - chunk_overlap

            # Prevent infinite loop
            if chunk_size <= chunk_overlap:
                break

        return segments if segments else {'main': source}

    count += 1

    @project.segmenters.register(
        'word_chunker',
        package='ef.plugins.advanced',
        description='Configurable word-based chunking'
    )
    def word_chunker(source: Any, words_per_chunk: int = 200, chunk_overlap: int = 50) -> dict[str, str]:
        """
        Chunk text by word count with configurable size and overlap.

        Args:
            source: Text or dict of text
            words_per_chunk: Maximum words per chunk
            chunk_overlap: Overlap in words between consecutive chunks

        Returns:
            Dict of segments
        """
        if isinstance(source, dict):
            source = '\n\n'.join(source.values())

        words = source.split()
        segments = {}
        start = 0
        chunk_num = 0

        while start < len(words):
            end = start + words_per_chunk
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)

            if chunk_text.strip():
                segments[f'chunk_{chunk_num}'] = chunk_text
                chunk_num += 1

            # Move start forward
            start = start + words_per_chunk - chunk_overlap

            if words_per_chunk <= chunk_overlap:
                break

        return segments if segments else {'main': source}

    count += 1

    print(f"✓ Registered {count} configurable segmenters")
    return count


# ============================================================================
# Sliding Window Segmenters
# ============================================================================


def register_sliding_window_segmenters(project) -> int:
    """
    Register sliding window segmenters with overlapping segments.

    Sliding windows are essential for:
    - Maintaining context across segments
    - Preventing information loss at boundaries
    - Creating robust embeddings

    Args:
        project: Project instance with segmenters registry

    Returns:
        Number of segmenters registered
    """
    count = 0

    @project.segmenters.register(
        'sliding_window',
        package='ef.plugins.advanced',
        description='Sliding window with 50% overlap',
        window_size=500,
        stride=250
    )
    def sliding_window(source: Any, window_size: int = 500, stride: int = 250) -> dict[str, str]:
        """
        Create overlapping segments using a sliding window.

        Args:
            source: Text or dict of text
            window_size: Size of each window in characters
            stride: Step size between windows

        Returns:
            Dict of overlapping segments
        """
        if isinstance(source, dict):
            source = '\n\n'.join(source.values())

        segments = {}
        position = 0
        window_num = 0

        while position < len(source):
            window = source[position:position + window_size]

            if window.strip():
                segments[f'window_{window_num}_pos_{position}'] = window
                window_num += 1

            position += stride

            # Stop if we've covered the text
            if position >= len(source) and window_num > 0:
                break

        return segments if segments else {'main': source}

    count += 1

    @project.segmenters.register(
        'sentence_window',
        package='ef.plugins.advanced',
        description='Sliding window of sentences',
        sentences_per_window=3,
        overlap_sentences=1
    )
    def sentence_window(
        source: Any,
        sentences_per_window: int = 3,
        overlap_sentences: int = 1
    ) -> dict[str, str]:
        """
        Create overlapping windows of sentences.

        Args:
            source: Text or dict of text
            sentences_per_window: Number of sentences per window
            overlap_sentences: Number of sentences to overlap

        Returns:
            Dict of sentence windows
        """
        if isinstance(source, dict):
            source = '\n\n'.join(source.values())

        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', source)
        sentences = [s.strip() for s in sentences if s.strip()]

        segments = {}
        position = 0
        window_num = 0

        while position < len(sentences):
            window_sents = sentences[position:position + sentences_per_window]
            window_text = '. '.join(window_sents) + '.'

            segments[f'sent_window_{window_num}'] = window_text
            window_num += 1

            # Move forward by (window_size - overlap)
            position += sentences_per_window - overlap_sentences

            if position >= len(sentences):
                break

        return segments if segments else {'main': source}

    count += 1

    print(f"✓ Registered {count} sliding window segmenters")
    return count


# ============================================================================
# Hierarchical Segmentation
# ============================================================================


def register_hierarchical_segmenters(project) -> int:
    """
    Register hierarchical segmentation strategies.

    Hierarchical segmentation creates nested levels:
    - Document level
    - Section/Chapter level
    - Paragraph level
    - Sentence level

    Uses dot notation for keys: 'doc.section.para.sent'

    Args:
        project: Project instance with segmenters registry

    Returns:
        Number of segmenters registered
    """
    count = 0

    @project.segmenters.register(
        'hierarchical',
        package='ef.plugins.advanced',
        description='Multi-level hierarchical segmentation (paragraphs → sentences)'
    )
    def hierarchical_segmenter(source: Any, max_depth: int = 2) -> dict[str, str]:
        """
        Create hierarchical segments with parent-child relationships.

        Args:
            source: Text or dict of text
            max_depth: Maximum nesting depth (1=paragraphs, 2=sentences)

        Returns:
            Dict with hierarchical keys (e.g., 'para_0.sent_1')
        """
        if isinstance(source, dict):
            source = '\n\n'.join(source.values())

        segments = {}

        # Level 1: Paragraphs (split on double newlines)
        paragraphs = re.split(r'\n\s*\n', source)

        for p_idx, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            para_key = f'para_{p_idx}'

            if max_depth >= 1:
                segments[para_key] = para

            # Level 2: Sentences within paragraphs
            if max_depth >= 2:
                sentences = re.split(r'[.!?]+', para)
                for s_idx, sent in enumerate(sentences):
                    sent = sent.strip()
                    if sent:
                        sent_key = f'{para_key}.sent_{s_idx}'
                        segments[sent_key] = sent

        return segments if segments else {'main': source}

    count += 1

    @project.segmenters.register(
        'markdown_hierarchical',
        package='ef.plugins.advanced',
        description='Hierarchical segmentation respecting markdown headers'
    )
    def markdown_hierarchical(source: Any) -> dict[str, str]:
        """
        Segment markdown by header hierarchy.

        Creates keys like: 'h1_0', 'h1_0.h2_1', 'h1_0.h2_1.content'

        Args:
            source: Markdown text or dict

        Returns:
            Dict with hierarchical markdown segments
        """
        if isinstance(source, dict):
            source = '\n\n'.join(source.values())

        segments = {}
        lines = source.split('\n')

        current_path = []
        current_content = []
        header_counts = {}

        for line in lines:
            # Check if line is a header
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if header_match:
                # Save previous content
                if current_content and current_path:
                    content_key = '.'.join(current_path) + '.content'
                    segments[content_key] = '\n'.join(current_content).strip()
                    current_content = []

                # Determine header level
                level = len(header_match.group(1))
                title = header_match.group(2)

                # Adjust path based on level
                current_path = current_path[:level - 1]

                # Create header key
                header_type = f'h{level}'
                header_count = header_counts.get(header_type, 0)
                header_counts[header_type] = header_count + 1

                header_key = f'{header_type}_{header_count}'
                current_path.append(header_key)

                # Store header
                full_key = '.'.join(current_path)
                segments[full_key] = title
            else:
                # Accumulate content
                current_content.append(line)

        # Save final content
        if current_content and current_path:
            content_key = '.'.join(current_path) + '.content'
            segments[content_key] = '\n'.join(current_content).strip()
        elif current_content:
            segments['content'] = '\n'.join(current_content).strip()

        return segments if segments else {'main': source}

    count += 1

    print(f"✓ Registered {count} hierarchical segmenters")
    return count


# ============================================================================
# Semantic/Smart Segmentation
# ============================================================================


def register_semantic_segmenters(project) -> int:
    """
    Register semantic segmentation strategies.

    These use embeddings or statistical methods to find natural breakpoints
    based on semantic similarity rather than just syntax.

    Args:
        project: Project instance with segmenters registry

    Returns:
        Number of segmenters registered
    """
    count = 0

    @project.segmenters.register(
        'semantic_similarity',
        package='ef.plugins.advanced',
        description='Segment based on semantic similarity between sentences (requires sentence-transformers)'
    )
    def semantic_similarity_segmenter(
        source: Any,
        similarity_threshold: float = 0.5,
        min_segment_sentences: int = 2
    ) -> dict[str, str]:
        """
        Segment text by finding semantic breakpoints.

        Breaks when cosine similarity between consecutive sentences
        drops below threshold.

        Args:
            source: Text or dict of text
            similarity_threshold: Similarity threshold for breaking (0-1)
            min_segment_sentences: Minimum sentences per segment

        Returns:
            Dict of semantically coherent segments
        """
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            warnings.warn(
                "sentence-transformers not available. Falling back to simple sentence segmentation."
            )
            # Fallback to simple sentence splitting
            if isinstance(source, dict):
                source = '\n\n'.join(source.values())
            sentences = re.split(r'[.!?]+', source)
            return {f'sent_{i}': s.strip() for i, s in enumerate(sentences) if s.strip()}

        if isinstance(source, dict):
            source = '\n\n'.join(source.values())

        # Split into sentences
        sentences = re.split(r'[.!?]+', source)
        sentences = [s.strip() for s in sentences if s.strip()]

        if len(sentences) <= 1:
            return {'main': source}

        # Encode sentences
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Small, fast model
        embeddings = model.encode(sentences)

        # Find breakpoints based on similarity
        segments = {}
        current_segment = [sentences[0]]
        segment_num = 0

        for i in range(1, len(sentences)):
            # Calculate similarity with previous sentence
            sim = cosine_similarity(
                embeddings[i - 1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]

            if sim < similarity_threshold and len(current_segment) >= min_segment_sentences:
                # Create segment
                segment_text = '. '.join(current_segment) + '.'
                segments[f'segment_{segment_num}'] = segment_text
                segment_num += 1
                current_segment = [sentences[i]]
            else:
                current_segment.append(sentences[i])

        # Add final segment
        if current_segment:
            segment_text = '. '.join(current_segment) + '.'
            segments[f'segment_{segment_num}'] = segment_text

        return segments if segments else {'main': source}

    count += 1

    print(f"✓ Registered {count} semantic segmenters")
    return count


# ============================================================================
# Pattern-Based Segmenters
# ============================================================================


def create_pattern_segmenter(
    project,
    name: str,
    pattern: str,
    description: Optional[str] = None,
    keep_delimiter: bool = False
) -> Callable:
    """
    Create and register a custom pattern-based segmenter.

    Args:
        project: Project instance
        name: Name for the segmenter
        pattern: Regex pattern to split on
        description: Description for metadata
        keep_delimiter: Whether to keep the delimiter in results

    Returns:
        The registered segmenter function

    Example:
        >>> create_pattern_segmenter(
        ...     project,
        ...     'by_double_newline',
        ...     r'\\n\\n+',
        ...     'Split on paragraph breaks'
        ... )
    """
    @project.segmenters.register(
        name,
        package='ef.plugins.advanced',
        pattern=pattern,
        description=description or f'Pattern-based segmenter: {pattern}'
    )
    def pattern_segmenter(source: Any) -> dict[str, str]:
        """Segment text using regex pattern."""
        if isinstance(source, dict):
            source = '\n\n'.join(source.values())

        if keep_delimiter:
            parts = re.split(f'({pattern})', source)
            # Recombine parts with delimiters
            segments = {}
            i = 0
            for idx in range(0, len(parts) - 1, 2):
                text = parts[idx] + (parts[idx + 1] if idx + 1 < len(parts) else '')
                if text.strip():
                    segments[f'{name}_{i}'] = text.strip()
                    i += 1
        else:
            parts = re.split(pattern, source)
            segments = {
                f'{name}_{i}': part.strip()
                for i, part in enumerate(parts)
                if part.strip()
            }

        return segments if segments else {'main': source}

    return pattern_segmenter


def register_pattern_segmenters(project) -> int:
    """
    Register common pattern-based segmenters.

    Args:
        project: Project instance

    Returns:
        Number of segmenters registered
    """
    count = 0

    # Paragraph breaks
    create_pattern_segmenter(
        project,
        'by_paragraphs',
        r'\n\s*\n',
        'Split on paragraph breaks (double newline)'
    )
    count += 1

    # Section headers (markdown)
    create_pattern_segmenter(
        project,
        'by_markdown_headers',
        r'\n#+\s+',
        'Split on markdown headers',
        keep_delimiter=True
    )
    count += 1

    # Code blocks
    create_pattern_segmenter(
        project,
        'by_code_blocks',
        r'```[a-z]*\n.*?\n```',
        'Split around code blocks'
    )
    count += 1

    print(f"✓ Registered {count} pattern-based segmenters")
    return count


# ============================================================================
# Multi-Strategy Segmentation
# ============================================================================


def register_multi_strategy_segmenters(project) -> int:
    """
    Register multi-strategy segmenters that combine multiple approaches.

    Args:
        project: Project instance

    Returns:
        Number of segmenters registered
    """
    count = 0

    @project.segmenters.register(
        'multi_level',
        package='ef.plugins.advanced',
        description='Apply multiple segmentation strategies and return all levels'
    )
    def multi_level_segmenter(
        source: Any,
        strategies: Optional[list[str]] = None
    ) -> dict[str, str]:
        """
        Apply multiple segmentation strategies.

        Args:
            source: Text or dict of text
            strategies: List of segmenter names to apply (default: ['sentences', 'lines'])

        Returns:
            Dict with keys prefixed by strategy name
        """
        if strategies is None:
            strategies = ['sentences', 'lines']

        results = {}

        for strategy in strategies:
            if strategy in project.segmenters:
                try:
                    segments = project.segmenters[strategy](source)
                    # Prefix keys with strategy name
                    for key, value in segments.items():
                        prefixed_key = f'{strategy}.{key}'
                        results[prefixed_key] = value
                except Exception as e:
                    warnings.warn(f"Strategy '{strategy}' failed: {e}")

        return results if results else {'main': str(source)}

    count += 1

    @project.segmenters.register(
        'best_fit',
        package='ef.plugins.advanced',
        description='Automatically choose best segmentation based on content'
    )
    def best_fit_segmenter(source: Any) -> dict[str, str]:
        """
        Automatically select the best segmentation strategy.

        Analyzes content and chooses appropriate strategy:
        - Code: Use AST or tree-sitter
        - Markdown: Use markdown-aware
        - Long text: Use paragraphs
        - Short text: Use sentences

        Args:
            source: Text or dict of text

        Returns:
            Segments from best-fit strategy
        """
        if isinstance(source, dict):
            source_text = '\n\n'.join(source.values())
        else:
            source_text = source

        # Detect content type
        has_python_code = bool(re.search(r'\bdef\s+\w+\s*\(', source_text))
        has_markdown = bool(re.search(r'^#+\s+', source_text, re.MULTILINE))
        has_paragraphs = source_text.count('\n\n') > 2
        is_long = len(source_text) > 1000

        # Choose strategy
        if has_python_code and 'ast_python' in project.segmenters:
            strategy = 'ast_python'
        elif has_markdown and 'markdown_hierarchical' in project.segmenters:
            strategy = 'markdown_hierarchical'
        elif has_paragraphs:
            strategy = 'by_paragraphs' if 'by_paragraphs' in project.segmenters else 'sentences'
        elif is_long:
            strategy = 'sliding_window' if 'sliding_window' in project.segmenters else 'sentences'
        else:
            strategy = 'sentences'

        if strategy in project.segmenters:
            return project.segmenters[strategy](source)
        else:
            # Fallback
            return {'main': source_text}

    count += 1

    print(f"✓ Registered {count} multi-strategy segmenters")
    return count


# ============================================================================
# Registration Function
# ============================================================================


def register_all_advanced_segmenters(project, verbose: bool = True) -> dict[str, int]:
    """
    Register all advanced segmenters.

    Args:
        project: Project instance
        verbose: Whether to print registration status

    Returns:
        Dict mapping category to number registered
    """
    if verbose:
        print("\n" + "="*60)
        print("Registering advanced segmenters...")
        print("="*60 + "\n")

    results = {}
    results['configurable'] = register_configurable_segmenters(project)
    results['sliding_window'] = register_sliding_window_segmenters(project)
    results['hierarchical'] = register_hierarchical_segmenters(project)
    results['semantic'] = register_semantic_segmenters(project)
    results['pattern'] = register_pattern_segmenters(project)
    results['multi_strategy'] = register_multi_strategy_segmenters(project)

    if verbose:
        total = sum(results.values())
        print(f"\n{'='*60}")
        print(f"Total: {total} advanced segmenters registered")
        print("="*60 + "\n")

    return results
