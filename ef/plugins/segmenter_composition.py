"""
Segmenter composition and chaining utilities.

This module provides tools to combine multiple segmenters into sophisticated
multi-stage segmentation strategies.
"""

from typing import Any, Callable, Optional, Union
import warnings


def compose_segmenters(
    project,
    segmenter_names: list[str],
    names: Optional[list[str]] = None,
    flatten: bool = False
) -> Callable:
    """
    Compose multiple segmenters in sequence.

    Args:
        project: Project instance
        segmenter_names: List of segmenter names to apply in order
        names: Optional list of names for each level (for key prefixing)
        flatten: If True, flatten hierarchical keys into single-level dict

    Returns:
        Composed segmenter function

    Example:
        >>> composed = compose_segmenters(
        ...     project,
        ...     ['by_paragraphs', 'sentences'],
        ...     names=['para', 'sent']
        ... )
        >>> segments = composed(text)
        >>> # Returns: {'para_0.sent_0': '...', 'para_0.sent_1': '...', ...}
    """
    if names is None:
        names = [f'level_{i}' for i in range(len(segmenter_names))]

    if len(names) != len(segmenter_names):
        raise ValueError("names and segmenter_names must have same length")

    def composed_segmenter(source: Any) -> dict[str, str]:
        """Apply segmenters in sequence."""
        # Start with source as first-level segments
        if isinstance(source, str):
            current_segments = {'root': source}
        else:
            current_segments = source

        # Apply each segmenter in sequence
        for i, seg_name in enumerate(segmenter_names):
            if seg_name not in project.segmenters:
                warnings.warn(f"Segmenter '{seg_name}' not found, skipping")
                continue

            segmenter = project.segmenters[seg_name]
            next_segments = {}

            # Apply segmenter to each current segment
            for key, text in current_segments.items():
                try:
                    sub_segments = segmenter(text)

                    # Create hierarchical keys
                    for sub_key, sub_text in sub_segments.items():
                        if i == 0 and key == 'root':
                            # First level, skip 'root' prefix
                            new_key = f'{names[i]}_{sub_key}'
                        else:
                            # Subsequent levels, maintain hierarchy
                            new_key = f'{key}.{names[i]}_{sub_key}'

                        next_segments[new_key] = sub_text

                except Exception as e:
                    warnings.warn(f"Error applying {seg_name} to '{key}': {e}")
                    # Keep original on error
                    next_segments[key] = text

            current_segments = next_segments

        # Flatten if requested
        if flatten:
            current_segments = {
                k.split('.')[-1]: v for k, v in current_segments.items()
            }

        return current_segments

    return composed_segmenter


def parallel_segmenters(
    project,
    segmenter_names: list[str],
    merge_strategy: str = 'union'
) -> Callable:
    """
    Apply multiple segmenters in parallel and merge results.

    Args:
        project: Project instance
        segmenter_names: List of segmenter names to apply
        merge_strategy: How to merge results ('union', 'intersection', 'vote')

    Returns:
        Merged segmenter function

    Example:
        >>> parallel = parallel_segmenters(
        ...     project,
        ...     ['sentences', 'by_paragraphs'],
        ...     merge_strategy='union'
        ... )
    """
    def parallel_segmenter(source: Any) -> dict[str, str]:
        """Apply multiple segmenters and merge."""
        all_results = {}

        for seg_name in segmenter_names:
            if seg_name not in project.segmenters:
                warnings.warn(f"Segmenter '{seg_name}' not found, skipping")
                continue

            segmenter = project.segmenters[seg_name]

            try:
                segments = segmenter(source)

                # Prefix keys with segmenter name
                for key, value in segments.items():
                    prefixed_key = f'{seg_name}.{key}'
                    all_results[prefixed_key] = value

            except Exception as e:
                warnings.warn(f"Error with {seg_name}: {e}")

        return all_results

    return parallel_segmenter


def conditional_segmenter(
    project,
    condition: Callable[[str], bool],
    true_segmenter: str,
    false_segmenter: str
) -> Callable:
    """
    Choose segmenter based on condition.

    Args:
        project: Project instance
        condition: Function that takes source and returns bool
        true_segmenter: Segmenter to use if condition is True
        false_segmenter: Segmenter to use if condition is False

    Returns:
        Conditional segmenter function

    Example:
        >>> # Use code segmenter for code, sentence segmenter for text
        >>> conditional = conditional_segmenter(
        ...     project,
        ...     condition=lambda s: 'def ' in s and 'import ' in s,
        ...     true_segmenter='ast_python',
        ...     false_segmenter='sentences'
        ... )
    """
    def cond_segmenter(source: Any) -> dict[str, str]:
        """Apply segmenter based on condition."""
        if isinstance(source, dict):
            source_text = '\n\n'.join(source.values())
        else:
            source_text = source

        # Evaluate condition
        try:
            use_true = condition(source_text)
        except Exception as e:
            warnings.warn(f"Condition evaluation failed: {e}, using false branch")
            use_true = False

        # Select segmenter
        seg_name = true_segmenter if use_true else false_segmenter

        if seg_name not in project.segmenters:
            warnings.warn(f"Segmenter '{seg_name}' not found, using identity")
            return {'main': source_text}

        return project.segmenters[seg_name](source)

    return cond_segmenter


def filter_segments(
    segmenter: Callable,
    min_length: Optional[int] = None,
    max_length: Optional[int] = None,
    pattern: Optional[str] = None,
    custom_filter: Optional[Callable[[str], bool]] = None
) -> Callable:
    """
    Wrap a segmenter with filtering logic.

    Args:
        segmenter: Base segmenter function
        min_length: Minimum segment length (characters)
        max_length: Maximum segment length (characters)
        pattern: Regex pattern that segments must match
        custom_filter: Custom filter function

    Returns:
        Filtered segmenter function

    Example:
        >>> # Only keep segments between 50-500 chars
        >>> filtered = filter_segments(
        ...     project.segmenters['sentences'],
        ...     min_length=50,
        ...     max_length=500
        ... )
    """
    import re

    def filtered_segmenter(source: Any) -> dict[str, str]:
        """Apply segmenter and filter results."""
        segments = segmenter(source)
        filtered = {}

        for key, text in segments.items():
            # Apply filters
            if min_length and len(text) < min_length:
                continue

            if max_length and len(text) > max_length:
                continue

            if pattern and not re.search(pattern, text):
                continue

            if custom_filter and not custom_filter(text):
                continue

            filtered[key] = text

        return filtered

    return filtered_segmenter


def transform_segments(
    segmenter: Callable,
    transform: Callable[[str], str]
) -> Callable:
    """
    Apply transformation to segment outputs.

    Args:
        segmenter: Base segmenter function
        transform: Transformation function (segment -> segment)

    Returns:
        Transformed segmenter function

    Example:
        >>> # Lowercase all segments
        >>> lowercased = transform_segments(
        ...     project.segmenters['sentences'],
        ...     transform=str.lower
        ... )
        >>>
        >>> # Remove URLs
        >>> import re
        >>> no_urls = transform_segments(
        ...     project.segmenters['sentences'],
        ...     transform=lambda s: re.sub(r'https?://\S+', '', s)
        ... )
    """
    def transformed_segmenter(source: Any) -> dict[str, str]:
        """Apply segmenter and transform results."""
        segments = segmenter(source)

        transformed = {}
        for key, text in segments.items():
            try:
                transformed[key] = transform(text)
            except Exception as e:
                warnings.warn(f"Transform failed for '{key}': {e}")
                transformed[key] = text

        return transformed

    return transformed_segmenter


def deduplicate_segments(segmenter: Callable, strategy: str = 'exact') -> Callable:
    """
    Remove duplicate segments.

    Args:
        segmenter: Base segmenter function
        strategy: Deduplication strategy ('exact', 'fuzzy', 'semantic')

    Returns:
        Deduplicated segmenter function
    """
    def deduped_segmenter(source: Any) -> dict[str, str]:
        """Apply segmenter and deduplicate."""
        segments = segmenter(source)

        if strategy == 'exact':
            # Keep first occurrence of each unique text
            seen = set()
            deduped = {}

            for key, text in segments.items():
                if text not in seen:
                    seen.add(text)
                    deduped[key] = text

            return deduped

        elif strategy == 'fuzzy':
            # Use fuzzy matching (requires fuzzywuzzy)
            try:
                from fuzzywuzzy import fuzz

                deduped = {}
                texts = []

                for key, text in segments.items():
                    is_duplicate = False

                    for existing_text in texts:
                        similarity = fuzz.ratio(text, existing_text)
                        if similarity > 90:  # 90% similar
                            is_duplicate = True
                            break

                    if not is_duplicate:
                        deduped[key] = text
                        texts.append(text)

                return deduped

            except ImportError:
                warnings.warn("fuzzywuzzy not available, falling back to exact dedup")
                return deduped_segmenter(source)

        else:
            # Unknown strategy, return as-is
            return segments

    return deduped_segmenter


class SegmenterPipeline:
    """
    Build complex segmentation pipelines with fluent interface.

    Example:
        >>> pipeline = (SegmenterPipeline(project)
        ...     .segment('by_paragraphs')
        ...     .then('sentences')
        ...     .filter(min_length=50)
        ...     .transform(str.lower)
        ...     .deduplicate()
        ...     .build())
        >>>
        >>> segments = pipeline(text)
    """

    def __init__(self, project):
        """Initialize pipeline."""
        self.project = project
        self._stages = []

    def segment(self, segmenter_name: str, name: Optional[str] = None):
        """Add a segmentation stage."""
        self._stages.append({
            'type': 'segment',
            'name': segmenter_name,
            'level_name': name or segmenter_name
        })
        return self

    def then(self, segmenter_name: str, name: Optional[str] = None):
        """Add another segmentation stage (alias for segment)."""
        return self.segment(segmenter_name, name)

    def filter(self, **filter_args):
        """Add filtering stage."""
        self._stages.append({
            'type': 'filter',
            'args': filter_args
        })
        return self

    def transform(self, transform_func: Callable):
        """Add transformation stage."""
        self._stages.append({
            'type': 'transform',
            'func': transform_func
        })
        return self

    def deduplicate(self, strategy: str = 'exact'):
        """Add deduplication stage."""
        self._stages.append({
            'type': 'deduplicate',
            'strategy': strategy
        })
        return self

    def build(self) -> Callable:
        """Build the final pipeline function."""
        def pipeline(source: Any) -> dict[str, str]:
            """Execute the pipeline."""
            result = source

            # Apply each stage
            for stage in self._stages:
                if stage['type'] == 'segment':
                    # For first segment stage, apply directly
                    if isinstance(result, (str, dict)) and not isinstance(result, dict) or \
                       (isinstance(result, dict) and all(isinstance(v, str) for v in result.values())):
                        # Source or simple dict
                        seg_name = stage['name']

                        if seg_name not in self.project.segmenters:
                            warnings.warn(f"Segmenter '{seg_name}' not found")
                            continue

                        result = self.project.segmenters[seg_name](result)
                    else:
                        # Already segmented, apply to each segment
                        next_result = {}
                        seg_name = stage['name']

                        if seg_name not in self.project.segmenters:
                            warnings.warn(f"Segmenter '{seg_name}' not found")
                            continue

                        segmenter = self.project.segmenters[seg_name]

                        for key, text in result.items():
                            sub_segs = segmenter(text)
                            for sub_key, sub_text in sub_segs.items():
                                next_result[f'{key}.{sub_key}'] = sub_text

                        result = next_result

                elif stage['type'] == 'filter':
                    # Apply filters
                    filtered = {}

                    for key, text in result.items():
                        keep = True

                        if 'min_length' in stage['args'] and len(text) < stage['args']['min_length']:
                            keep = False

                        if 'max_length' in stage['args'] and len(text) > stage['args']['max_length']:
                            keep = False

                        if keep:
                            filtered[key] = text

                    result = filtered

                elif stage['type'] == 'transform':
                    # Apply transformation
                    transformed = {}

                    for key, text in result.items():
                        try:
                            transformed[key] = stage['func'](text)
                        except Exception as e:
                            warnings.warn(f"Transform failed for '{key}': {e}")
                            transformed[key] = text

                    result = transformed

                elif stage['type'] == 'deduplicate':
                    # Deduplicate
                    seen = set()
                    deduped = {}

                    for key, text in result.items():
                        if text not in seen:
                            seen.add(text)
                            deduped[key] = text

                    result = deduped

            return result

        return pipeline


def register_composed_segmenter(
    project,
    name: str,
    segmenter_names: list[str],
    **compose_kwargs
) -> None:
    """
    Register a composed segmenter with the project.

    Args:
        project: Project instance
        name: Name for the composed segmenter
        segmenter_names: List of segmenters to compose
        **compose_kwargs: Additional arguments for compose_segmenters

    Example:
        >>> register_composed_segmenter(
        ...     project,
        ...     'para_sentences',
        ...     ['by_paragraphs', 'sentences'],
        ...     names=['para', 'sent']
        ... )
        >>>
        >>> segments = project.segmenters['para_sentences'](text)
    """
    composed = compose_segmenters(project, segmenter_names, **compose_kwargs)

    project.segmenters.register(
        name,
        package='ef.plugins.composition',
        description=f'Composed segmenter: {" → ".join(segmenter_names)}',
        components=segmenter_names
    )(composed)
