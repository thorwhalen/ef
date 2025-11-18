# Advanced Segmentation Features

This guide covers the advanced segmentation features in the `ef` framework, including configurable segmenters, sliding windows, hierarchical segmentation, semantic segmentation, utilities, and optimization tools.

## Table of Contents

- [Auto-Registration](#auto-registration)
- [Configurable Segmenters](#configurable-segmenters)
- [Sliding Window Segmentation](#sliding-window-segmentation)
- [Hierarchical Segmentation](#hierarchical-segmentation)
- [Semantic Segmentation](#semantic-segmentation)
- [Pattern-Based Segmentation](#pattern-based-segmentation)
- [Multi-Strategy Segmentation](#multi-strategy-segmentation)
- [Segmentation Utilities](#segmentation-utilities)
  - [Quality Metrics](#quality-metrics)
  - [Comparison Tools](#comparison-tools)
  - [Caching](#caching)
  - [Batch Processing](#batch-processing)
  - [Auto-Detection](#auto-detection)
  - [Recommendation System](#recommendation-system)

## Auto-Registration

Automatically register all available external segmenters on project creation:

```python
from ef import Project

# Enable auto-registration
project = Project.create('my_project', backend='memory', auto_register_segmenters=True)

# All available external segmenters (LangChain, spaCy, NLTK, etc.) are now registered
segmenters = project.list_components()['segmenters']
print(f"Available: {len(segmenters)} segmenters")
```

**When to use:**
- Starting a new project and want all tools available
- Prototyping and experimenting with different segmenters
- Don't want to manually call registration functions

**When not to use:**
- Need fine-grained control over which packages are loaded
- Want to minimize startup time
- Working with limited dependencies

## Configurable Segmenters

Segmenters that can be configured at runtime without pre-registering every variant:

```python
from ef import Project
from ef.plugins import advanced_segmenters
from functools import partial

project = Project.create('test', backend='memory')
advanced_segmenters.register_configurable_segmenters(project)

# Get base segmenter
char_chunker = project.segmenters['char_chunker']

# Create custom configurations
small_chunks = partial(char_chunker, chunk_size=500, chunk_overlap=100)
large_chunks = partial(char_chunker, chunk_size=2000, chunk_overlap=400)

# Use in pipelines
text = open('long_document.txt').read()
small_segments = small_chunks(text)
large_segments = large_chunks(text)
```

### Available Configurable Segmenters

#### 1. `char_chunker`

Segment by character count with overlap:

```python
segments = char_chunker(
    source="Your text here...",
    chunk_size=1000,      # Max characters per chunk
    chunk_overlap=200     # Overlap between chunks
)
```

**Parameters:**
- `chunk_size`: Maximum characters per chunk (default: 1000)
- `chunk_overlap`: Overlap in characters (default: 200)

**Use cases:**
- Fixed-size chunks for processing pipelines
- Consistent segment sizes for embeddings
- LLM input preparation

#### 2. `word_chunker`

Segment by word count with overlap:

```python
segments = word_chunker(
    source="Your text here...",
    words_per_chunk=200,   # Max words per chunk
    chunk_overlap=50       # Overlap in words
)
```

**Parameters:**
- `words_per_chunk`: Maximum words per chunk (default: 200)
- `chunk_overlap`: Overlap in words (default: 50)

**Use cases:**
- Natural language processing
- Word-based analysis
- Consistent vocabulary size per segment

## Sliding Window Segmentation

Create overlapping segments to preserve context across boundaries:

```python
from ef import Project
from ef.plugins import advanced_segmenters

project = Project.create('test', backend='memory')
advanced_segmenters.register_sliding_window_segmenters(project)

text = open('article.txt').read()
```

### Available Sliding Window Segmenters

#### 1. `sliding_window`

Character-based sliding window:

```python
segments = project.segmenters['sliding_window'](
    text,
    window_size=500,  # Window size in characters
    stride=250        # Step size (50% overlap with these values)
)

# Example output:
# {
#   'window_0_pos_0': 'First 500 characters...',
#   'window_1_pos_250': 'Characters 250-750 (overlaps with window_0)...',
#   'window_2_pos_500': 'Characters 500-1000...',
#   ...
# }
```

**Use cases:**
- Preserving context for embeddings
- Preventing information loss at chunk boundaries
- Creating robust search indices

**Example with visualization:**

```
Text:     |------------- 1000 chars ------------|
Window 0: [=====500=====]
Window 1:       [=====500=====]
Window 2:             [=====500=====]
          ^     ^     ^
          0    250   500  (positions)
```

#### 2. `sentence_window`

Sentence-based sliding window:

```python
segments = project.segmenters['sentence_window'](
    text,
    sentences_per_window=3,  # Sentences per window
    overlap_sentences=1      # Overlapping sentences
)

# Example output:
# {
#   'sent_window_0': 'Sentence 1. Sentence 2. Sentence 3.',
#   'sent_window_1': 'Sentence 3. Sentence 4. Sentence 5.',  # Overlaps
#   ...
# }
```

**Use cases:**
- Semantic coherence across windows
- QA systems with context
- Document summarization

## Hierarchical Segmentation

Create nested, multi-level segments:

```python
from ef import Project
from ef.plugins import advanced_segmenters

project = Project.create('test', backend='memory')
advanced_segmenters.register_hierarchical_segmenters(project)
```

### Available Hierarchical Segmenters

#### 1. `hierarchical`

Multi-level paragraph and sentence segmentation:

```python
text = """
First paragraph with multiple sentences. Second sentence here.

Second paragraph. More content. Even more.

Third paragraph.
"""

segments = project.segmenters['hierarchical'](text, max_depth=2)

# Output structure:
# {
#   'para_0': 'First paragraph with multiple sentences. Second sentence here.',
#   'para_0.sent_0': 'First paragraph with multiple sentences',
#   'para_0.sent_1': 'Second sentence here',
#   'para_1': 'Second paragraph. More content. Even more.',
#   'para_1.sent_0': 'Second paragraph',
#   'para_1.sent_1': 'More content',
#   'para_1.sent_2': 'Even more',
#   'para_2': 'Third paragraph',
# }
```

**Parameters:**
- `max_depth`: Nesting depth (1=paragraphs, 2=sentences, default: 2)

**Use cases:**
- Multi-granularity analysis
- Document structure preservation
- Hierarchical embeddings

#### 2. `markdown_hierarchical`

Markdown-aware hierarchical segmentation:

```python
markdown = """
# Main Header

Introduction paragraph.

## Section 1

Section content here.

### Subsection 1.1

Subsection content.

## Section 2

More content.
"""

segments = project.segmenters['markdown_hierarchical'](markdown)

# Output structure:
# {
#   'h1_0': 'Main Header',
#   'h1_0.content': 'Introduction paragraph.',
#   'h1_0.h2_0': 'Section 1',
#   'h1_0.h2_0.content': 'Section content here.',
#   'h1_0.h2_0.h3_0': 'Subsection 1.1',
#   'h1_0.h2_0.h3_0.content': 'Subsection content.',
#   'h1_0.h2_1': 'Section 2',
#   'h1_0.h2_1.content': 'More content.',
# }
```

**Use cases:**
- Documentation processing
- README analysis
- Blog post segmentation

## Semantic Segmentation

Segment based on meaning rather than just syntax:

```python
from ef import Project
from ef.plugins import advanced_segmenters

project = Project.create('test', backend='memory')
advanced_segmenters.register_semantic_segmenters(project)
```

### `semantic_similarity`

Break text at points where semantic similarity drops:

```python
text = """
Machine learning is fascinating. It combines statistics and computer science.
The weather today is nice. I enjoy sunny days.
Python is a great programming language. It has many libraries.
"""

segments = project.segmenters['semantic_similarity'](
    text,
    similarity_threshold=0.5,  # Break when similarity < 0.5
    min_segment_sentences=2     # Minimum sentences per segment
)

# Will group semantically related sentences together:
# {
#   'segment_0': 'Machine learning is fascinating. It combines statistics and computer science.',
#   'segment_1': 'The weather today is nice. I enjoy sunny days.',
#   'segment_2': 'Python is a great programming language. It has many libraries.',
# }
```

**Requirements:**
- `sentence-transformers` package
- `scikit-learn` package

**Parameters:**
- `similarity_threshold`: Cosine similarity threshold for breaking (0-1)
- `min_segment_sentences`: Minimum sentences per segment

**Use cases:**
- Topic segmentation
- Creating coherent chunks
- Semantic search preparation

**Performance note:** Requires sentence embedding computation; may be slow for large texts.

## Pattern-Based Segmentation

Create custom segmenters using regex patterns:

```python
from ef import Project
from ef.plugins import advanced_segmenters

project = Project.create('test', backend='memory')

# Create custom pattern segmenters
advanced_segmenters.create_pattern_segmenter(
    project,
    name='by_double_newline',
    pattern=r'\n\n+',
    description='Split on paragraph breaks'
)

advanced_segmenters.create_pattern_segmenter(
    project,
    name='by_semicolon',
    pattern=r';\s*',
    description='Split on semicolons'
)

# Use them
text = "Part 1;\nPart 2;\nPart 3"
segments = project.segmenters['by_semicolon'](text)
```

### Built-in Pattern Segmenters

```python
advanced_segmenters.register_pattern_segmenters(project)

# Now available:
# - 'by_paragraphs': Split on double newlines
# - 'by_markdown_headers': Split on markdown headers
# - 'by_code_blocks': Split around code blocks
```

**Examples:**

```python
# Paragraph splitting
text = "Para 1.\n\nPara 2.\n\nPara 3."
segments = project.segmenters['by_paragraphs'](text)

# Markdown header splitting
markdown = "# H1\nContent\n## H2\nMore"
segments = project.segmenters['by_markdown_headers'](markdown)
```

## Multi-Strategy Segmentation

Apply multiple segmentation strategies:

```python
from ef import Project
from ef.plugins import advanced_segmenters

project = Project.create('test', backend='memory')
advanced_segmenters.register_multi_strategy_segmenters(project)
```

### Available Multi-Strategy Segmenters

#### 1. `multi_level`

Apply multiple segmenters and return all results:

```python
text = "Hello. World. Test."

segments = project.segmenters['multi_level'](
    text,
    strategies=['sentences', 'lines', 'identity']
)

# Output:
# {
#   'sentences.sent_0': 'Hello',
#   'sentences.sent_1': 'World',
#   'sentences.sent_2': 'Test',
#   'lines.line_0': 'Hello. World. Test.',
#   'identity.main': 'Hello. World. Test.',
# }
```

**Use cases:**
- Comparing segmentation strategies
- Multi-granularity analysis
- Ensemble approaches

#### 2. `best_fit`

Automatically choose the best strategy based on content:

```python
# Python code
code = "def hello():\n    print('hi')"
segments = project.segmenters['best_fit'](code)
# Uses AST or tree-sitter if available

# Markdown
markdown = "# Header\n\nContent"
segments = project.segmenters['best_fit'](markdown)
# Uses markdown segmenter if available

# Plain text
text = "Regular sentences. More text."
segments = project.segmenters['best_fit'](text)
# Uses sentence segmenter
```

## Segmentation Utilities

### Quality Metrics

Analyze segmentation quality:

```python
from ef.plugins import segmenter_utils

segments = {
    'seg1': 'Short',
    'seg2': 'Medium length text here',
    'seg3': 'Very long segment with lots of content and multiple sentences.'
}

metrics = segmenter_utils.analyze_segmentation(segments)

print(metrics)
# {
#   'count': 3,
#   'total_chars': 103,
#   'avg_length': 34.3,
#   'std_dev': 25.8,
#   'min_length': 5,
#   'max_length': 69,
#   'median_length': 24,
#   'avg_words': 7.0,
#   'total_words': 21,
#   'distribution': {...}
# }

# Print formatted report
segmenter_utils.print_segmentation_report(segments, "My Segmentation")
```

### Comparison Tools

Compare different segmentation strategies:

```python
from ef.plugins import segmenter_utils

text = "Your document here. Multiple sentences. Various content."

results = segmenter_utils.compare_segmenters(
    project,
    text,
    segmenter_names=['sentences', 'sliding_window', 'hierarchical'],
    verbose=True  # Prints comparison report
)

# Access individual results
for name, result in results.items():
    if result['success']:
        metrics = result['metrics']
        print(f"{name}: {metrics['count']} segments, avg {metrics['avg_length']:.1f} chars")
```

**Output example:**

```
======================================================================
Segmenter Comparison Report
======================================================================

Segmenter                 Segments     Avg Length      Status
----------------------------------------------------------------------
sentences                 3            18.3 chars      ✓ Success
sliding_window            5            35.0 chars      ✓ Success
hierarchical              4            22.5 chars      ✓ Success
======================================================================
```

### Caching

Cache segmentation results to avoid re-computation:

```python
from ef.plugins import segmenter_utils

# Create a cache
cache = segmenter_utils.SegmentationCache(max_size=1000)

# Wrap a segmenter with caching
segmenter = project.segmenters['sentences']
cached_segmenter = segmenter_utils.make_cached_segmenter(segmenter, cache)

# First call - computes and caches
text = "Hello. World."
result1 = cached_segmenter(text)

# Second call - retrieves from cache (much faster)
result2 = cached_segmenter(text)

# Check cache statistics
stats = cache.stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")
print(f"Hits: {stats['hits']}, Misses: {stats['misses']}")
```

**Global cache:**

```python
# Use global cache automatically
from ef.plugins import segmenter_utils

segmenter = project.segmenters['sentences']
cached = segmenter_utils.make_cached_segmenter(segmenter)  # Uses global cache

# Check global stats
stats = segmenter_utils.get_cache_stats()

# Clear global cache
segmenter_utils.clear_cache()
```

### Batch Processing

Process multiple texts or files efficiently:

```python
from ef.plugins import segmenter_utils

# Batch process texts
texts = {
    'doc1': 'First document. Multiple sentences.',
    'doc2': 'Second document. More content.',
    'doc3': 'Third document here.'
}

results = segmenter_utils.batch_segment_texts(
    project,
    texts,
    segmenter='sentences',
    cache_results=True
)

for doc_id, result in results.items():
    if result['success']:
        print(f"{doc_id}: {len(result['segments'])} segments")
```

**Batch process files:**

```python
# Process all Python files in a directory
results = segmenter_utils.batch_segment_files(
    project,
    pattern='**/*.py',
    segmenter='auto',  # Auto-detect best segmenter
    recursive=True,
    cache_results=True
)

# Or specific files
results = segmenter_utils.batch_segment_files(
    project,
    file_paths=['file1.py', 'file2.md', 'file3.txt'],
    segmenter='auto'
)
```

### Auto-Detection

Automatically detect the best segmenter for content:

```python
from ef.plugins import segmenter_utils

# Detect from content and filename
code = "def hello():\n    import os\n    print('hi')"
segmenter_name = segmenter_utils.auto_detect_segmenter(
    project,
    content=code,
    filename='script.py'
)
print(segmenter_name)  # 'ast_python' or 'tree_sitter_python'

# Detect from content alone
markdown = "# Header\n\nContent"
segmenter_name = segmenter_utils.auto_detect_segmenter(
    project,
    content=markdown
)
print(segmenter_name)  # 'langchain_markdown' or 'markdown_hierarchical'
```

**Detection logic:**
1. File extension (if filename provided)
2. Content patterns (code syntax, markdown, etc.)
3. Content length
4. Default to general-purpose segmenter

### Recommendation System

Get recommendations based on use case:

```python
from ef.plugins import segmenter_utils

# Recommend for code
seg, reason = segmenter_utils.recommend_segmenter(
    project,
    content_type='code',
    language='python'
)
print(f"Use {seg}: {reason}")
# "Use tree_sitter_python: Best for Python code structure analysis"

# Recommend for LLM context
seg, reason = segmenter_utils.recommend_segmenter(
    project,
    use_case='llm_context'
)
print(f"Use {seg}: {reason}")
# "Use tiktoken_gpt-4: Respects GPT-4 token limits (8K tokens)"

# Recommend for documentation
seg, reason = segmenter_utils.recommend_segmenter(
    project,
    content_type='documentation'
)
# "Use langchain_markdown: Respects markdown document structure"

# Recommend based on length
seg, reason = segmenter_utils.recommend_segmenter(
    project,
    content_length=50000
)
# "Use langchain_recursive_2000: Large chunks for very long documents"

# Print formatted recommendation
segmenter_utils.print_recommendation(
    project,
    content_type='code',
    use_case='semantic_search'
)
```

## Complete Example Workflows

### Workflow 1: Processing a Large Document

```python
from ef import Project
from ef.plugins import advanced_segmenters, segmenter_utils

# Create project
project = Project.create('document_analysis', backend='memory')
advanced_segmenters.register_all_advanced_segmenters(project)

# Load document
with open('large_document.txt') as f:
    text = f.read()

# Get recommendation
seg_name, reason = segmenter_utils.recommend_segmenter(
    project,
    content_length=len(text),
    use_case='semantic_search'
)
print(f"Recommended: {seg_name} - {reason}")

# Segment the document
segmenter = project.segmenters[seg_name]
segments = segmenter(text)

# Analyze quality
metrics = segmenter_utils.analyze_segmentation(segments)
print(f"Created {metrics['count']} segments")
print(f"Average length: {metrics['avg_length']:.1f} characters")

# Create pipeline and embed
project.segments.update(segments)
project.create_pipeline('embed', embedder='simple')
results = project.run_pipeline('embed', persist=True)
```

### Workflow 2: Code Repository Analysis

```python
from ef import Project
from ef.plugins import advanced_segmenters, segmenter_utils

project = Project.create('code_analysis', backend='memory')
advanced_segmenters.register_all_advanced_segmenters(project)

# Batch process all Python files
results = segmenter_utils.batch_segment_files(
    project,
    pattern='**/*.py',
    segmenter='ast_python',
    recursive=True,
    cache_results=True
)

# Analyze each file
for filepath, result in results.items():
    if result['success']:
        segments = result['segments']
        metrics = segmenter_utils.analyze_segmentation(segments)

        print(f"\n{filepath}:")
        print(f"  Functions/Classes: {metrics['count']}")
        print(f"  Avg size: {metrics['avg_length']:.0f} chars")
```

### Workflow 3: Comparing Strategies

```python
from ef import Project
from ef.plugins import advanced_segmenters, segmenter_utils

project = Project.create('comparison', backend='memory')
advanced_segmenters.register_all_advanced_segmenters(project)

# Load test document
with open('test_doc.txt') as f:
    text = f.read()

# Compare different strategies
strategies = [
    'sentences',
    'sliding_window',
    'hierarchical',
    'by_paragraphs'
]

results = segmenter_utils.compare_segmenters(
    project,
    text,
    strategies,
    verbose=True
)

# Choose best based on metrics
best_seg = None
best_score = float('inf')

for name, result in results.items():
    if result['success']:
        metrics = result['metrics']
        # Prefer moderate segment count with low variance
        score = abs(metrics['count'] - 10) + metrics['std_dev']

        if score < best_score:
            best_score = score
            best_seg = name

print(f"\nBest segmenter: {best_seg}")
```

## Performance Tips

1. **Use caching** for repeated segmentation of the same text
2. **Batch process** when dealing with multiple documents
3. **Choose appropriate granularity** - finer segmentation = more processing
4. **Prefer built-in segmenters** for speed (AST over tree-sitter)
5. **Avoid semantic segmentation** for large texts (requires embeddings)
6. **Use auto-detection** to avoid manual configuration

## Troubleshooting

### Semantic Segmenter Not Working

```python
# Install required packages:
pip install sentence-transformers scikit-learn
```

### Cache Not Improving Performance

```python
# Check cache stats
stats = segmenter_utils.get_cache_stats()
print(f"Hit rate: {stats['hit_rate']:.1%}")

# If hit rate is low, you may be segmenting different texts
# Cache is content-based, so identical text will hit
```

### Pattern Segmenter Not Splitting

```python
# Debug the pattern
import re
pattern = r'\n\n+'
text = "Para 1\n\nPara 2"

# Test pattern directly
parts = re.split(pattern, text)
print(parts)  # ['Para 1', 'Para 2']
```

## See Also

- [Main Segmenter Documentation](SEGMENTER_REGISTRATION.md)
- [ef README](../README.md)
- [API Reference](API.md)
