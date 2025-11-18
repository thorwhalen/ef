"""
Performance profiling for segmenters.

Measure time, memory, and throughput of segmentation strategies.
"""

import time
import statistics


def profile_segmenter(
    segmenter,
    texts: list[str],
    metrics: list[str] = None
) -> dict:
    """
    Profile segmenter performance.
    
    Args:
        segmenter: Segmenter function
        texts: Test texts
        metrics: Metrics to measure
    
    Returns:
        Profile results
    """
    if metrics is None:
        metrics = ['time', 'throughput']
    
    times = []
    throughputs = []
    
    for text in texts:
        start = time.time()
        try:
            segments = segmenter(text)
            elapsed = time.time() - start
            
            times.append(elapsed)
            throughputs.append(len(text) / max(0.001, elapsed))
        except Exception:
            pass
    
    results = {}
    
    if 'time' in metrics and times:
        results['time'] = {
            'mean': statistics.mean(times),
            'std': statistics.stdev(times) if len(times) > 1 else 0,
            'min': min(times),
            'max': max(times)
        }
    
    if 'throughput' in metrics and throughputs:
        results['throughput'] = {
            'mean': statistics.mean(throughputs),
            'unit': 'chars/sec'
        }
    
    return results


def compare_performance(
    project,
    segmenters: list[str],
    corpus: list[str]
) -> dict:
    """Compare performance of multiple segmenters."""
    results = {}
    
    for seg_name in segmenters:
        if seg_name not in project.segmenters:
            continue
        
        segmenter = project.segmenters[seg_name]
        results[seg_name] = profile_segmenter(segmenter, corpus)
    
    return results
