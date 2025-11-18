# EF Advanced Segmentation Features

This document provides an overview of the 15 advanced segmentation features added to the EF framework.

## Quick Start

```python
from ef import Project

# Create project with auto-registration of all segmenters
project = Project.create('myproject', auto_register_segmenters=True)
```

## Feature Overview

### 1. Segmenter Composition (`ef.plugins.segmenter_composition`)

Chain and combine multiple segmenters for sophisticated multi-stage segmentation.

```python
from ef.plugins.segmenter_composition import compose_segmenters, SegmenterPipeline

# Compose segmenters in sequence
composed = compose_segmenters(project, ['by_paragraphs', 'sentences'])
result = composed(text)

# Or use fluent interface
pipeline = (SegmenterPipeline(project)
    .segment('sentences')
    .filter(min_length=50)
    .deduplicate()
    .build())
```

**Key Functions:**
- `compose_segmenters()` - Apply segmenters in sequence
- `parallel_segmenters()` - Apply multiple in parallel
- `conditional_segmenter()` - Choose based on condition
- `SegmenterPipeline` - Fluent builder interface

---

### 2. Quality Scoring & Optimization (`ef.plugins.segmenter_quality`)

Evaluate and optimize segmentation quality with multiple metrics.

```python
from ef.plugins.segmenter_quality import evaluate_quality, optimize_for_task

# Evaluate segmentation quality
quality = evaluate_quality(segments, source=text)
# Returns: {'coherence': 0.85, 'balance': 0.92, 'coverage': 1.0, 'overall': 0.89}

# Find best segmenter for a task
best = optimize_for_task(project, [text1, text2], task='semantic_search')
```

**Metrics:**
- Coherence: Semantic consistency within segments
- Balance: Size distribution balance
- Coverage: Source text coverage
- Boundary Quality: Natural boundary detection

---

### 3. Streaming/Incremental Segmentation (`ef.plugins.segmenter_streaming`)

Process large files and streaming data without loading everything into memory.

```python
from ef.plugins.segmenter_streaming import StreamingSegmenter, IncrementalSegmenter

# Stream large files
with StreamingSegmenter(project, 'sentences', file_path='huge.txt') as streamer:
    for key, segment in streamer:
        process(segment)

# Incremental segmentation
incremental = IncrementalSegmenter(project, 'sentences')
incremental.add_text("First part. ")
incremental.add_text("Second part. ")
segments = incremental.finalize()
```

---

### 4. Configuration Management (`ef.plugins.segmenter_config`)

Save and load segmentation configurations with YAML/JSON support.

```python
from ef.plugins.segmenter_config import SegmentationConfig, PRESETS

# Create configuration
config = SegmentationConfig(
    segmenter='sentences',
    params={'min_length': 10},
    preprocessors=['strip_whitespace'],
    postprocessors=['deduplicate']
)

# Save/load
config.save('config.yaml')
config2 = SegmentationConfig.load('config.yaml')

# Use presets
article_config = PRESETS['article']  # Optimized for articles
code_config = PRESETS['code_python']  # Optimized for Python code
```

---

### 5. Visualization Tools (`ef.plugins.segmenter_viz`)

Visualize segmentation results with matplotlib, plotly, and interactive dashboards.

```python
from ef.plugins.segmenter_viz import plot_segmentation, compare_visual, interactive_explorer

# Plot segment boundaries
plot_segmentation(text, segments, output_file='viz.png')

# Compare multiple strategies
compare_visual(text, {'strategy_a': seg1, 'strategy_b': seg2})

# Launch interactive explorer (Dash web UI)
interactive_explorer(text, project)
```

---

### 6. Adaptive/Smart Segmentation (`ef.plugins.adaptive_segmentation`)

Dynamically adjust segmentation based on content characteristics.

```python
from ef.plugins.adaptive_segmentation import adaptive_segmenter, content_aware_segmenter

# Adaptive segmentation with quality optimization
segments = adaptive_segmenter(text, target_segment_size=500, quality_threshold=0.8)

# Content-aware segmentation (detects structure)
segments = content_aware_segmenter(text, detect_sections=True)
```

---

### 7. Multilingual Support (`ef.plugins.multilingual_segmenters`)

Specialized segmenters for different languages.

```python
from ef.plugins.multilingual_segmenters import register_multilingual_segmenters, detect_language

# Register multilingual segmenters
register_multilingual_segmenters(project)

# Auto-detect language and segment
lang = detect_language(text)  # Returns 'en', 'zh', 'ja', 'ar', etc.
segments = project.segmenters['multilingual_auto'](text)

# Language-specific segmenters
segments_cn = project.segmenters['chinese_jieba'](chinese_text)
segments_jp = project.segmenters['japanese_tinysegmenter'](japanese_text)
```

---

### 8. Domain-Specific Segmenters (`ef.plugins.domain_segmenters`)

Specialized segmenters for legal, scientific, medical, and code documents.

```python
from ef.plugins import domain_segmenters

# Register domain-specific segmenters
domain_segmenters.register_legal_segmenters(project)
domain_segmenters.register_scientific_segmenters(project)
domain_segmenters.register_code_segmenters(project)
domain_segmenters.register_medical_segmenters(project)

# Use domain segmenters
legal_segs = project.segmenters['legal_clauses'](contract_text)
paper_segs = project.segmenters['paper_sections'](research_paper)
code_segs = project.segmenters['by_complexity'](python_code)
medical_segs = project.segmenters['clinical_notes'](patient_notes)
```

---

### 9. Embedding-Aware Segmentation (`ef.plugins.embedding_optimized`)

Optimize segments for embedding quality and LLM context windows.

```python
from ef.plugins.embedding_optimized import segment_for_embeddings, segment_for_llm

# Optimize for embedding quality
segments = segment_for_embeddings(text, embedder=my_embedder, target_segments=20)

# Optimize for LLM token limits
segments = segment_for_llm(
    text,
    model='gpt-4',
    context_window=8000,
    preserve_context=True
)
```

---

### 10. A/B Testing Framework (`ef.plugins.ab_testing`)

Compare segmentation strategies with statistical rigor.

```python
from ef.plugins.ab_testing import ABTest

# Create A/B test
test = ABTest(
    name='Strategy Comparison',
    strategies={
        'baseline': 'sentences',
        'experimental': 'semantic_adaptive'
    }
)

# Run test
test.run(project, corpus=[text1, text2, text3])

# Get results
print(test.summary())
sig_test = test.significance_test()
print(f"Significant: {sig_test['significant']}, p-value: {sig_test['p_value']}")
```

---

### 11. Performance Profiling (`ef.plugins.profiler`)

Measure time, memory, and throughput of segmentation strategies.

```python
from ef.plugins.profiler import profile_segmenter, compare_performance

# Profile a segmenter
segmenter = project.segmenters['sentences']
profile = profile_segmenter(segmenter, [text1, text2, text3])
# Returns: {'time': {'mean': 0.05, 'std': 0.01}, 'throughput': {'mean': 50000, 'unit': 'chars/sec'}}

# Compare multiple segmenters
results = compare_performance(project, ['sentences', 'paragraphs', 'sliding_window'], corpus)
```

---

### 12. Plugin Marketplace (`ef.plugins.marketplace`)

Browse, install, and publish community segmenters.

```python
from ef.plugins.marketplace import marketplace

# Search for segmenters
results = marketplace.search('legal')

# Install a segmenter
marketplace.install('legal-doc-segmenter', version='latest')

# Publish your own
marketplace.publish('my-segmenter', my_func, metadata={
    'description': 'Custom segmenter',
    'author': 'me'
})
```

---

### 13. CLI Tools (`ef.plugins.cli_tools`)

Command-line interface for segmentation operations.

```bash
# Segment a file
python -m ef.plugins.cli_tools segment input.txt -s sentences -f json

# List available segmenters
python -m ef.plugins.cli_tools list -v

# Benchmark segmenters
python -m ef.plugins.cli_tools benchmark input.txt -s sentences paragraphs

# Compare segmenters visually
python -m ef.plugins.cli_tools compare input.txt -s sentences paragraphs -o report.html

# Auto-select best segmenter
python -m ef.plugins.cli_tools auto input.txt -t scientific

# Batch process directory
python -m ef.plugins.cli_tools batch ./docs/ -o ./output/ -s sentences
```

Or use programmatically:

```python
from ef.plugins.cli_tools import segment_file_cli, benchmark_segmenters_cli

segment_file_cli('input.txt', 'output.json', 'sentences', 'json', project)
benchmark_segmenters_cli('test.txt', ['sentences', 'paragraphs'], project)
```

---

### 14. Vector Database Integration (`ef.plugins.vector_db_integration`)

Integrate with Pinecone, Weaviate, Chroma, Qdrant and other vector databases.

```python
from ef.plugins.vector_db_integration import create_adapter, segment_and_index

# Create vector DB adapter
chroma = create_adapter('chroma', path='./db')
pinecone = create_adapter('pinecone', api_key='...', environment='us-west1')
weaviate = create_adapter('weaviate', url='http://localhost:8080')
qdrant = create_adapter('qdrant', url='localhost', port=6333)

# Segment and index in one step
segments = segment_and_index(
    project,
    text,
    segmenter='sentences',
    vector_db=chroma,
    metadata={'source': 'article.txt'}
)

# Semantic search
from ef.plugins.vector_db_integration import semantic_search_segments
results = semantic_search_segments('query text', chroma, top_k=5)
```

---

### 15. Jupyter Integration (`ef.plugins.jupyter_widgets`)

Interactive widgets for exploring segmentation in Jupyter notebooks.

```python
from ef.plugins.jupyter_widgets import (
    notebook_explore,
    notebook_compare,
    notebook_segment,
    SegmentationVisualizer
)

# Interactive explorer with dropdown
notebook_explore(project, text)

# Compare multiple segmenters
results = notebook_compare(project, text, ['sentences', 'paragraphs', 'semantic'])

# Segment with visualization
segments = notebook_segment(project, text, 'sentences', visualize=True)

# Custom visualization
viz = SegmentationVisualizer(text, segments)
viz.show(style='highlight')  # or 'boundaries' or 'interactive'

# Export example notebook
from ef.plugins.jupyter_widgets import export_notebook_example
export_notebook_example('demo.ipynb')
```

---

## Installation

### Core Framework
```bash
pip install ef  # Or install from source
```

### Optional Dependencies

For full functionality, install optional packages:

```bash
# LangChain segmenters
pip install langchain langchain-text-splitters

# Code segmentation
pip install tree-sitter tree-sitter-languages

# NLP segmenters
pip install spacy nltk segtok
python -m spacy download en_core_web_sm

# Token-based segmentation
pip install tiktoken

# Multilingual support
pip install jieba tinysegmenter langdetect

# Semantic segmentation
pip install sentence-transformers scikit-learn

# Visualization
pip install matplotlib plotly dash

# Vector databases
pip install chromadb pinecone-client weaviate-client qdrant-client

# Jupyter widgets
pip install ipywidgets

# Statistical testing
pip install scipy

# Configuration
pip install pyyaml
```

---

## Testing

All 15 features have comprehensive test coverage:

```bash
# Run all advanced feature tests (39 tests)
pytest tests/test_advanced_features.py -v

# Run specific test
pytest tests/test_advanced_features.py::test_compose_segmenters -v
```

---

## Architecture

The advanced features are designed with:

1. **Modularity**: Each feature is a separate module that can be used independently
2. **Composability**: Features work together (e.g., visualize quality scores, profile composed segmenters)
3. **Graceful Degradation**: Optional dependencies degrade gracefully with informative messages
4. **Consistency**: All features follow the same segmenter interface: `(source: Any) -> dict[str, str]`
5. **Extensibility**: Easy to add custom segmenters and extend existing features

---

## Examples

### Example 1: Complete Workflow

```python
from ef import Project
from ef.plugins import segmenter_composition, segmenter_quality, profiler

# Create project
project = Project.create('analysis', auto_register_segmenters=True)

# Define text
text = "..." # Your text here

# Compose segmenters
pipeline = segmenter_composition.compose_segmenters(project, ['by_paragraphs', 'sentences'])
segments = pipeline(text)

# Evaluate quality
quality = segmenter_quality.evaluate_quality(segments, source=text)
print(f"Quality score: {quality['overall']:.2f}")

# Profile performance
profile = profiler.profile_segmenter(pipeline, [text])
print(f"Average time: {profile['time']['mean']:.4f}s")
```

### Example 2: Find Best Segmenter for Task

```python
from ef import Project
from ef.plugins.segmenter_quality import optimize_for_task

project = Project.create('optimize', auto_register_segmenters=True)

# Prepare validation corpus
corpus = [text1, text2, text3]

# Find best segmenter
result = optimize_for_task(
    project,
    corpus,
    task='semantic_search',
    candidates=['sentences', 'sliding_window_500', 'semantic_adaptive']
)

print(f"Best segmenter: {result['name']} (score: {result['score']:.3f})")
```

### Example 3: Vector DB RAG Pipeline

```python
from ef import Project
from ef.plugins.vector_db_integration import create_adapter, segment_and_index, semantic_search_segments

project = Project.create('rag', auto_register_segmenters=True)

# Create vector DB
chroma = create_adapter('chroma', path='./knowledge_base')

# Index documents
for doc in documents:
    segment_and_index(
        project,
        doc['text'],
        segmenter='llm_optimized',  # Optimized for LLM context
        vector_db=chroma,
        metadata={'doc_id': doc['id'], 'title': doc['title']}
    )

# Query
query = "How does feature X work?"
results = semantic_search_segments(query, chroma, top_k=3)
for r in results:
    print(f"Score: {r['score']:.3f} | {r['text'][:100]}...")
```

---

## Contributing

To add a new advanced feature:

1. Create module in `ef/plugins/your_feature.py`
2. Follow the segmenter interface: `(source: Any) -> dict[str, str]`
3. Add registration function if applicable
4. Add tests in `tests/test_advanced_features.py`
5. Update `ef/plugins/__init__.py` to export your module
6. Document in this file

---

## License

Same as EF framework license.

---

## Support

- GitHub Issues: https://github.com/thorwhalen/ef/issues
- Documentation: See individual module docstrings
- Examples: `tests/test_advanced_features.py`
