"""
A/B testing framework for segmentation strategies.

Compare segmentation strategies on real tasks with statistical rigor.
"""

import statistics
from typing import Any, Callable, Optional


class ABTest:
    """A/B test for segmentation strategies."""
    
    def __init__(
        self,
        name: str,
        strategies: dict[str, str],
        task: str = 'general',
        test_queries: Optional[list] = None,
        ground_truth: Optional[dict] = None
    ):
        """Initialize A/B test."""
        self.name = name
        self.strategies = strategies
        self.task = task
        self.test_queries = test_queries or []
        self.ground_truth = ground_truth or {}
        self.results = {}
    
    def run(self, project, corpus: list[str]) -> 'ABTest':
        """Run the A/B test."""
        from ef.plugins.segmenter_quality import evaluate_quality
        
        for strategy_name, segmenter_name in self.strategies.items():
            if segmenter_name not in project.segmenters:
                continue
            
            segmenter = project.segmenters[segmenter_name]
            scores = []
            
            for text in corpus:
                try:
                    segments = segmenter(text)
                    quality = evaluate_quality(segments, source=text)
                    scores.append(quality['overall'])
                except Exception as e:
                    pass
            
            if scores:
                self.results[strategy_name] = {
                    'mean': statistics.mean(scores),
                    'std': statistics.stdev(scores) if len(scores) > 1 else 0,
                    'scores': scores
                }
        
        return self
    
    def summary(self) -> str:
        """Get summary of results."""
        lines = [f"\nA/B Test: {self.name}", "=" * 60]
        
        for strategy, result in sorted(self.results.items(), 
                                      key=lambda x: x[1]['mean'], 
                                      reverse=True):
            lines.append(f"{strategy}: Score = {result['mean']:.3f} (+/- {result['std']:.3f})")
        
        return '\n'.join(lines)
    
    def significance_test(self, alpha: float = 0.05) -> dict:
        """Test statistical significance."""
        try:
            from scipy import stats
            
            if len(self.results) < 2:
                return {'error': 'Need at least 2 strategies'}
            
            strategies = list(self.results.keys())
            scores1 = self.results[strategies[0]]['scores']
            scores2 = self.results[strategies[1]]['scores']
            
            t_stat, p_value = stats.ttest_ind(scores1, scores2)
            
            return {
                'significant': p_value < alpha,
                'p_value': p_value,
                't_statistic': t_stat,
                'alpha': alpha
            }
        except ImportError:
            return {'error': 'scipy not available'}
