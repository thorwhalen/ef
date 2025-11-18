"""
Adaptive and smart segmentation that adjusts based on content.

This module provides segmenters that dynamically adapt their
behavior based on content characteristics.
"""

from typing import Any, Callable, Optional
import statistics
import warnings


def adaptive_segmenter(
    source: Any,
    target_segment_size: int = 500,
    quality_threshold: float = 0.8,
    max_iterations: int = 5
) -> dict[str, str]:
    """
    Adaptively segment to meet quality criteria.

    Starts with target size, measures coherence, adjusts boundaries
    to maximize coherence while respecting size constraints.

    Args:
        source: Text to segment
        target_segment_size: Target size per segment
        quality_threshold: Minimum quality score (0-1)
        max_iterations: Maximum refinement iterations

    Returns:
        Optimized segments
    """
    if isinstance(source, dict):
        source = '\n\n'.join(source.values())

    # Initial segmentation by target size
    segments = {}
    pos = 0
    idx = 0

    while pos < len(source):
        end = min(pos + target_segment_size, len(source))

        # Try to find natural boundary near target end
        boundary = find_natural_boundary(source, end, window=100)

        # Ensure we always advance (prevent infinite loop)
        if boundary <= pos:
            boundary = min(pos + 1, len(source))

        segment = source[pos:boundary]
        if segment.strip():
            segments[f'seg_{idx}'] = segment
            idx += 1

        pos = boundary

    # Refine boundaries for better coherence
    for iteration in range(max_iterations):
        from ef.plugins.segmenter_quality import evaluate_quality

        quality = evaluate_quality(segments, source=source)

        if quality['overall'] >= quality_threshold:
            break

        # Try to improve by adjusting boundaries
        segments = refine_boundaries(source, segments, quality)

    return segments


def find_natural_boundary(text: str, position: int, window: int = 100) -> int:
    """
    Find natural boundary (sentence end, paragraph) near position.

    Args:
        text: Full text
        position: Target position
        window: Search window size

    Returns:
        Position of natural boundary
    """
    start = max(0, position - window // 2)
    end = min(len(text), position + window // 2)

    search_area = text[start:end]

    # Look for paragraph breaks first
    para_break = search_area.rfind('\n\n')
    if para_break >= 0:
        return start + para_break + 2

    # Then sentence endings
    for terminator in ['. ', '! ', '? ']:
        sent_end = search_area.rfind(terminator)
        if sent_end >= 0:
            return start + sent_end + 2

    # Fall back to target position
    return position


def refine_boundaries(
    source: str,
    segments: dict[str, str],
    quality: dict
) -> dict[str, str]:
    """
    Refine segment boundaries based on quality metrics.

    Args:
        source: Original text
        segments: Current segments
        quality: Quality metrics

    Returns:
        Refined segments
    """
    # If balance is low, try to even out sizes
    if quality.get('balance', 1.0) < 0.5:
        lengths = [len(s) for s in segments.values()]
        target_len = statistics.mean(lengths)

        # Rebuild with more consistent sizes
        refined = {}
        pos = 0
        idx = 0

        while pos < len(source):
            end = min(pos + int(target_len), len(source))
            boundary = find_natural_boundary(source, end)

            # Ensure we always advance (prevent infinite loop)
            if boundary <= pos:
                boundary = min(pos + 1, len(source))

            segment = source[pos:boundary]
            if segment.strip():
                refined[f'seg_{idx}'] = segment
                idx += 1

            pos = boundary

        return refined

    return segments


def content_aware_segmenter(
    source: Any,
    detect_sections: bool = True,
    preserve_context: bool = True
) -> dict[str, str]:
    """
    Segment based on content structure detection.

    Detects:
    - Headers and sections
    - List items
    - Code blocks
    - Quotes

    Args:
        source: Text to segment
        detect_sections: Whether to detect section headers
        preserve_context: Whether to preserve contextual groupings

    Returns:
        Structurally-aware segments
    """
    if isinstance(source, dict):
        source = '\n\n'.join(source.values())

    segments = {}
    lines = source.split('\n')

    current_section = []
    section_idx = 0
    in_code_block = False

    for line in lines:
        # Detect code blocks
        if line.strip().startswith('```'):
            in_code_block = not in_code_block
            current_section.append(line)
            continue

        # Detect section headers
        if detect_sections and not in_code_block:
            if line.startswith('#') or (line and line[0].isupper() and ':' in line):
                # New section
                if current_section:
                    segments[f'section_{section_idx}'] = '\n'.join(current_section)
                    section_idx += 1
                    current_section = []

        current_section.append(line)

        # Break on double newline if not preserving context
        if not preserve_context and line == '':
            if current_section:
                segments[f'section_{section_idx}'] = '\n'.join(current_section)
                section_idx += 1
                current_section = []

    # Add final section
    if current_section:
        segments[f'section_{section_idx}'] = '\n'.join(current_section)

    return segments


def variable_size_segmenter(
    source: Any,
    min_size: int = 100,
    max_size: int = 2000,
    prefer_complete_sentences: bool = True
) -> dict[str, str]:
    """
    Create variable-sized segments based on natural boundaries.

    Args:
        source: Text to segment
        min_size: Minimum segment size
        max_size: Maximum segment size
        prefer_complete_sentences: Try to end at sentence boundaries

    Returns:
        Variable-sized segments
    """
    if isinstance(source, dict):
        source = '\n\n'.join(source.values())

    segments = {}
    pos = 0
    idx = 0

    while pos < len(source):
        # Start with max size
        end = min(pos + max_size, len(source))

        # Find natural boundary
        if prefer_complete_sentences:
            boundary = find_natural_boundary(source, end, window=min(200, max_size // 2))
        else:
            boundary = end

        # Ensure minimum size
        if boundary - pos < min_size and pos + min_size < len(source):
            boundary = pos + min_size

        segment = source[pos:boundary]
        if segment.strip():
            segments[f'seg_{idx}'] = segment
            idx += 1

        pos = boundary

    return segments


def register_adaptive_segmenters(project) -> int:
    """
    Register adaptive segmentation strategies.

    Args:
        project: Project instance

    Returns:
        Number of segmenters registered
    """
    count = 0

    @project.segmenters.register(
        'adaptive',
        package='ef.plugins.adaptive',
        description='Adaptive segmentation with quality optimization'
    )
    def adaptive(source: Any, **kwargs) -> dict[str, str]:
        return adaptive_segmenter(source, **kwargs)
    count += 1

    @project.segmenters.register(
        'content_aware',
        package='ef.plugins.adaptive',
        description='Content structure-aware segmentation'
    )
    def content_aware(source: Any, **kwargs) -> dict[str, str]:
        return content_aware_segmenter(source, **kwargs)
    count += 1

    @project.segmenters.register(
        'variable_size',
        package='ef.plugins.adaptive',
        description='Variable-sized segments with natural boundaries'
    )
    def variable_size(source: Any, **kwargs) -> dict[str, str]:
        return variable_size_segmenter(source, **kwargs)
    count += 1

    print(f"✓ Registered {count} adaptive segmenters")
    return count
