"""
Segmentation quality scoring and optimization.

This module provides tools to evaluate segmentation quality and
automatically optimize segmenter selection.
"""

from typing import Any, Callable, Optional, Union
import statistics
import warnings


def evaluate_coherence(segments: dict[str, str], method: str = 'simple') -> float:
    """
    Evaluate semantic coherence within segments.

    Args:
        segments: Dict of segments
        method: Evaluation method ('simple', 'embedding', 'perplexity')

    Returns:
        Coherence score (0-1, higher is better)
    """
    if not segments:
        return 0.0

    if method == 'simple':
        # Simple heuristic: segments with similar sentence lengths are more coherent
        sentences_per_segment = []

        for text in segments.values():
            # Count sentences (rough approximation)
            sent_count = text.count('.') + text.count('!') + text.count('?')
            sentences_per_segment.append(max(1, sent_count))

        if len(sentences_per_segment) < 2:
            return 1.0

        # Low variance = high coherence
        mean_sents = statistics.mean(sentences_per_segment)
        variance = statistics.variance(sentences_per_segment)

        # Normalize to 0-1
        coherence = 1.0 / (1.0 + variance / max(1, mean_sents))
        return coherence

    elif method == 'embedding':
        # Use embeddings to measure coherence
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np

            model = SentenceTransformer('all-MiniLM-L6-v2')

            coherence_scores = []

            for text in segments.values():
                # Split into sentences
                sentences = text.split('.')
                sentences = [s.strip() for s in sentences if s.strip()]

                if len(sentences) < 2:
                    coherence_scores.append(1.0)
                    continue

                # Embed sentences
                embeddings = model.encode(sentences)

                # Calculate pairwise cosine similarities
                similarities = []
                for i in range(len(embeddings) - 1):
                    sim = np.dot(embeddings[i], embeddings[i + 1]) / \
                          (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1]))
                    similarities.append(sim)

                # Average similarity = coherence
                coherence_scores.append(np.mean(similarities))

            return float(np.mean(coherence_scores))

        except ImportError:
            warnings.warn("sentence-transformers not available, using simple method")
            return evaluate_coherence(segments, method='simple')

    return 0.5  # Default


def evaluate_coverage(segments: dict[str, str], source: str) -> float:
    """
    Evaluate how much of the source is covered by segments.

    Args:
        segments: Dict of segments
        source: Original source text

    Returns:
        Coverage score (0-1)
    """
    if not segments or not source:
        return 0.0

    # Reconstruct text from segments
    reconstructed = ' '.join(segments.values())

    # Calculate character coverage
    coverage = min(1.0, len(reconstructed) / len(source))

    return coverage


def evaluate_balance(segments: dict[str, str]) -> float:
    """
    Evaluate balance of segment sizes.

    Args:
        segments: Dict of segments

    Returns:
        Balance score (0-1, higher means more balanced)
    """
    if not segments:
        return 0.0

    lengths = [len(s) for s in segments.values()]

    if len(lengths) < 2:
        return 1.0

    mean_len = statistics.mean(lengths)
    std_dev = statistics.stdev(lengths)

    # Coefficient of variation
    cv = std_dev / mean_len if mean_len > 0 else float('inf')

    # Convert to 0-1 score (lower CV = higher balance)
    balance = 1.0 / (1.0 + cv)

    return balance


def evaluate_boundary_quality(segments: dict[str, str]) -> float:
    """
    Evaluate quality of segment boundaries.

    Good boundaries:
    - At sentence endings
    - At paragraph breaks
    - Not mid-word

    Args:
        segments: Dict of segments

    Returns:
        Boundary quality score (0-1)
    """
    if not segments:
        return 0.0

    good_boundaries = 0
    total_boundaries = len(segments)

    for text in segments.values():
        # Check if segment ends well
        text_stripped = text.strip()

        if not text_stripped:
            continue

        # Good ending: sentence terminator or paragraph break
        if text_stripped[-1] in '.!?\n':
            good_boundaries += 1
        # Acceptable: complete word
        elif text_stripped[-1].isspace() or text_stripped[-1].isalnum():
            good_boundaries += 0.5

    return good_boundaries / max(1, total_boundaries)


def evaluate_quality(
    segments: dict[str, str],
    source: Optional[str] = None,
    criteria: Optional[list[str]] = None,
    weights: Optional[dict[str, float]] = None
) -> dict[str, float]:
    """
    Comprehensive quality evaluation.

    Args:
        segments: Dict of segments
        source: Optional source text (for coverage evaluation)
        criteria: List of criteria to evaluate
        weights: Optional weights for each criterion

    Returns:
        Dict of scores for each criterion plus overall score

    Example:
        >>> quality = evaluate_quality(
        ...     segments,
        ...     source=text,
        ...     criteria=['coherence', 'coverage', 'balance', 'boundary_quality']
        ... )
        >>> print(f"Overall: {quality['overall']:.2f}")
    """
    if criteria is None:
        criteria = ['coherence', 'balance', 'boundary_quality']
        if source:
            criteria.append('coverage')

    if weights is None:
        # Default equal weights
        weights = {c: 1.0 for c in criteria}

    scores = {}

    if 'coherence' in criteria:
        scores['coherence'] = evaluate_coherence(segments)

    if 'coverage' in criteria and source:
        scores['coverage'] = evaluate_coverage(segments, source)

    if 'balance' in criteria:
        scores['balance'] = evaluate_balance(segments)

    if 'boundary_quality' in criteria:
        scores['boundary_quality'] = evaluate_boundary_quality(segments)

    # Calculate weighted overall score
    weighted_sum = sum(scores[c] * weights.get(c, 1.0) for c in scores)
    weight_total = sum(weights.get(c, 1.0) for c in scores)

    scores['overall'] = weighted_sum / weight_total if weight_total > 0 else 0.0

    return scores


def optimize_for_task(
    project,
    texts: list[str],
    task: str = 'general',
    metric: str = 'overall',
    candidates: Optional[list[str]] = None
) -> dict[str, Any]:
    """
    Find optimal segmenter for a specific task.

    Args:
        project: Project instance
        texts: Validation texts to test on
        task: Task type ('semantic_search', 'summarization', 'qa', 'general')
        metric: Metric to optimize ('overall', 'coherence', 'balance', etc.)
        candidates: Optional list of segmenter names to test

    Returns:
        Dict with best segmenter name, score, and full results

    Example:
        >>> best = optimize_for_task(
        ...     project,
        ...     texts=validation_texts,
        ...     task='semantic_search',
        ...     metric='coherence'
        ... )
        >>> print(f"Best: {best['name']} with score {best['score']:.2f}")
    """
    if candidates is None:
        # Use available segmenters
        candidates = project.list_components()['segmenters']

    # Task-specific criteria and weights
    task_configs = {
        'semantic_search': {
            'criteria': ['coherence', 'balance'],
            'weights': {'coherence': 2.0, 'balance': 1.0}
        },
        'summarization': {
            'criteria': ['coverage', 'coherence'],
            'weights': {'coverage': 2.0, 'coherence': 1.0}
        },
        'qa': {
            'criteria': ['coherence', 'boundary_quality'],
            'weights': {'coherence': 2.0, 'boundary_quality': 1.0}
        },
        'general': {
            'criteria': ['coherence', 'balance', 'boundary_quality'],
            'weights': {'coherence': 1.0, 'balance': 1.0, 'boundary_quality': 1.0}
        }
    }

    config = task_configs.get(task, task_configs['general'])

    results = {}
    best_score = -1
    best_name = None

    for seg_name in candidates:
        if seg_name not in project.segmenters:
            continue

        segmenter = project.segmenters[seg_name]
        scores_list = []

        try:
            for text in texts:
                segments = segmenter(text)

                quality = evaluate_quality(
                    segments,
                    source=text,
                    criteria=config['criteria'],
                    weights=config['weights']
                )

                scores_list.append(quality[metric])

            # Average score across texts
            avg_score = statistics.mean(scores_list)

            results[seg_name] = {
                'score': avg_score,
                'scores': scores_list
            }

            if avg_score > best_score:
                best_score = avg_score
                best_name = seg_name

        except Exception as e:
            warnings.warn(f"Error evaluating {seg_name}: {e}")
            results[seg_name] = {'score': 0.0, 'error': str(e)}

    return {
        'name': best_name,
        'score': best_score,
        'all_results': results,
        'task': task,
        'metric': metric
    }


def suggest_improvements(
    segments: dict[str, str],
    source: Optional[str] = None
) -> list[str]:
    """
    Suggest improvements to segmentation.

    Args:
        segments: Current segments
        source: Optional source text

    Returns:
        List of improvement suggestions

    Example:
        >>> suggestions = suggest_improvements(segments, source=text)
        >>> for suggestion in suggestions:
        ...     print(f"- {suggestion}")
    """
    suggestions = []

    # Evaluate quality
    quality = evaluate_quality(segments, source=source)

    # Coherence issues
    if quality.get('coherence', 0) < 0.5:
        suggestions.append(
            "Low coherence detected. Consider using semantic_similarity segmenter "
            "or increasing chunk size to maintain topic cohesion."
        )

    # Balance issues
    if quality.get('balance', 0) < 0.5:
        suggestions.append(
            "Unbalanced segment sizes. Consider using sliding_window with overlap "
            "or configurable char_chunker for more consistent sizes."
        )

    # Boundary issues
    if quality.get('boundary_quality', 0) < 0.5:
        suggestions.append(
            "Poor boundary quality. Consider using sentence-based or paragraph-based "
            "segmentation for cleaner boundaries."
        )

    # Coverage issues
    if quality.get('coverage', 0) < 0.9:
        suggestions.append(
            f"Coverage is only {quality['coverage']*100:.1f}%. "
            "Some content may be lost during segmentation."
        )

    # Segment count
    if len(segments) < 3:
        suggestions.append(
            "Very few segments. Consider using finer-grained segmentation "
            "for better granularity."
        )
    elif len(segments) > 100:
        suggestions.append(
            "Many segments. Consider using coarser segmentation or "
            "hierarchical segmentation to reduce complexity."
        )

    # Length variance
    lengths = [len(s) for s in segments.values()]
    if lengths and max(lengths) / min(lengths) > 10:
        suggestions.append(
            "High length variance. Consider using fixed-size chunking "
            "or filtering very short/long segments."
        )

    if not suggestions:
        suggestions.append("Segmentation quality looks good! No major issues detected.")

    return suggestions


def benchmark_segmenters(
    project,
    corpus: list[str],
    segmenters: Optional[list[str]] = None,
    metrics: Optional[list[str]] = None
) -> dict[str, dict]:
    """
    Benchmark multiple segmenters on a corpus.

    Args:
        project: Project instance
        corpus: List of texts to test on
        segmenters: List of segmenter names (None = all available)
        metrics: Metrics to evaluate

    Returns:
        Benchmark results

    Example:
        >>> results = benchmark_segmenters(
        ...     project,
        ...     corpus=test_texts,
        ...     segmenters=['sentences', 'sliding_window', 'hierarchical']
        ... )
    """
    if segmenters is None:
        segmenters = project.list_components()['segmenters']

    if metrics is None:
        metrics = ['coherence', 'balance', 'boundary_quality']

    results = {}

    for seg_name in segmenters:
        if seg_name not in project.segmenters:
            continue

        segmenter = project.segmenters[seg_name]
        all_scores = {m: [] for m in metrics}
        all_scores['overall'] = []

        try:
            for text in corpus:
                segments = segmenter(text)

                quality = evaluate_quality(segments, source=text, criteria=metrics)

                for metric in metrics:
                    if metric in quality:
                        all_scores[metric].append(quality[metric])

                all_scores['overall'].append(quality['overall'])

            # Calculate aggregates
            results[seg_name] = {
                metric: {
                    'mean': statistics.mean(scores),
                    'std': statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    'min': min(scores),
                    'max': max(scores)
                }
                for metric, scores in all_scores.items()
                if scores
            }

        except Exception as e:
            warnings.warn(f"Error benchmarking {seg_name}: {e}")
            results[seg_name] = {'error': str(e)}

    return results
