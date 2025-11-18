# Segmenter Registration Guide

This guide explains how to register and use external segmenters in the `ef` framework.

## Overview

The `ef` framework provides a flexible registration system for text segmenters from various third-party packages. All registrations are **conditional** - if a package is not installed, its segmenters are simply skipped.

## Quick Start

```python
from ef import Project
from ef.plugins import segmenter_registry

# Create a project
project = Project.create('my_project', backend='memory')

# Register all available segmenters (conditional on package installation)
segmenter_registry.register_all_segmenters(project)

# List available segmenters
print(project.list_components()['segmenters'])

# Use a segmenter in a pipeline
project.add_source('doc1', 'Your text here. Multiple sentences.')
project.create_pipeline('analysis', segmenter='spacy_sentences', embedder='simple')
results = project.run_pipeline('analysis')
```

## Supported Packages

The registration system supports the following packages:

### 1. LangChain Text Splitters

**Package:** `langchain-text-splitters`
**Installation:** `pip install langchain-text-splitters`

LangChain provides sophisticated text splitting strategies:

- **Character-based splitters** (`langchain_char_500`, `langchain_char_1000`, `langchain_char_2000`)
  - Split text by character count with configurable chunk size and overlap
  - Good for general text when you need consistent chunk sizes

- **Recursive character splitters** (`langchain_recursive_500`, `langchain_recursive_1000`, `langchain_recursive_2000`)
  - Try multiple separators hierarchically (paragraph → sentence → word → character)
  - Better semantic coherence than simple character splitting

- **Markdown splitter** (`langchain_markdown`)
  - Respects markdown structure (headers, code blocks, lists)
  - Ideal for documentation and markdown files

- **Code splitters**
  - `langchain_python` - Python code (respects functions, classes)
  - `langchain_js` - JavaScript code
  - `langchain_ts` - TypeScript code
  - `langchain_java` - Java code
  - `langchain_cpp` - C++ code
  - `langchain_go` - Go code
  - `langchain_rust` - Rust code
  - `langchain_html` - HTML markup

**Example:**
```python
from ef import Project
from ef.plugins import segmenter_registry

project = Project.create('text_analysis')
segmenter_registry.register_langchain_segmenters(project)

# Use recursive splitter for better semantic coherence
project.add_source('article', open('article.txt').read())
project.create_pipeline('analyze', segmenter='langchain_recursive_1000', embedder='simple')
results = project.run_pipeline('analyze')
print(results['segments'])
```

### 2. Tree-sitter (Code Parsing)

**Package:** `tree-sitter-languages`
**Installation:** `pip install tree-sitter-languages`

Tree-sitter provides industrial-grade parsing for many programming languages:

- **Python segmenter** (`tree_sitter_python`)
  - Segments Python code by function and class definitions
  - Preserves complete function/class bodies including docstrings

- **Multi-language support** (`tree_sitter_javascript`, `tree_sitter_typescript`, `tree_sitter_java`, etc.)
  - Segments code by top-level definitions
  - Respects language syntax structure

**Example:**
```python
from ef import Project
from ef.plugins import segmenter_registry

project = Project.create('code_analysis')
segmenter_registry.register_tree_sitter_segmenters(project)

# Segment Python code by functions
code = '''
def hello():
    """Say hello."""
    print("Hello")

class Greeter:
    def greet(self, name):
        return f"Hello, {name}"
'''

project.add_source('module', code)
project.create_pipeline('analyze', segmenter='tree_sitter_python', embedder='simple')
results = project.run_pipeline('analyze')

# Results will have separate segments for the function and class
for key, segment in results['segments'].items():
    print(f"{key}:\n{segment}\n")
```

### 3. Python AST (Built-in)

**Package:** Built-in (no installation required)
**Installation:** N/A (uses Python's `ast` module)

Python's Abstract Syntax Tree module for Python code analysis:

- **Function-level segmenter** (`ast_python`)
  - Segments by function and class definitions
  - No external dependencies required

- **Statement-level segmenter** (`ast_python_statements`)
  - Segments by top-level statements
  - Good for analyzing code structure

**Example:**
```python
from ef import Project
from ef.plugins import segmenter_registry

project = Project.create('python_analysis')
segmenter_registry.register_ast_segmenters(project)

# Segment Python code by top-level statements
code = '''
import os
import sys

DEBUG = True

def main():
    print("Hello")

if __name__ == '__main__':
    main()
'''

project.add_source('script', code)
project.create_pipeline('analyze', segmenter='ast_python_statements', embedder='simple')
results = project.run_pipeline('analyze')

# Results will have separate segments for each top-level statement
for key, segment in results['segments'].items():
    print(f"{key}:\n{segment}\n")
```

### 4. spaCy (NLP)

**Package:** `spacy`
**Installation:** `pip install spacy` and optionally `python -m spacy download en_core_web_sm`

Industrial-strength NLP library:

- **Sentence segmenter** (`spacy_sentences`)
  - Advanced sentence boundary detection
  - Handles abbreviations, URLs, and edge cases better than regex

- **Token segmenter** (`spacy_tokens`)
  - Intelligent word tokenization
  - Handles contractions, hyphenation, etc.

**Example:**
```python
from ef import Project
from ef.plugins import segmenter_registry

project = Project.create('nlp_analysis')
segmenter_registry.register_spacy_segmenters(project)

# Segment text into sentences
text = "Dr. Smith works at U.S. Labs. He's researching A.I. systems!"
project.add_source('bio', text)
project.create_pipeline('analyze', segmenter='spacy_sentences', embedder='simple')
results = project.run_pipeline('analyze')

# spaCy handles abbreviations correctly
for key, segment in results['segments'].items():
    print(f"{key}: {segment}")
```

### 5. NLTK (Natural Language Toolkit)

**Package:** `nltk`
**Installation:** `pip install nltk`

Classic NLP library with robust tokenization:

- **Sentence tokenizer** (`nltk_sentences`)
  - Punkt sentence tokenizer
  - Well-tested on many languages

- **Word tokenizer** (`nltk_words`)
  - Penn Treebank tokenization
  - Standard in NLP research

**Example:**
```python
from ef import Project
from ef.plugins import segmenter_registry

project = Project.create('text_processing')
segmenter_registry.register_nltk_segmenters(project)

# Segment into sentences
text = """Natural language processing is fascinating.
It combines linguistics and computer science."""

project.add_source('text', text)
project.create_pipeline('analyze', segmenter='nltk_sentences', embedder='simple')
results = project.run_pipeline('analyze')
```

### 6. tiktoken (OpenAI Tokenizer)

**Package:** `tiktoken`
**Installation:** `pip install tiktoken`

OpenAI's tokenizer for token-count-based segmentation:

- **GPT-4 segmenter** (`tiktoken_gpt-4`)
  - Segments text into 8000-token chunks
  - Useful for preparing data for GPT-4 API

- **GPT-3.5 segmenter** (`tiktoken_gpt-3_5`)
  - Segments text into 4000-token chunks
  - Matches GPT-3.5 context window

**Example:**
```python
from ef import Project
from ef.plugins import segmenter_registry

project = Project.create('llm_prep')
segmenter_registry.register_tiktoken_segmenters(project)

# Segment long document for GPT-4 processing
with open('long_document.txt') as f:
    text = f.read()

project.add_source('doc', text)
project.create_pipeline('prep', segmenter='tiktoken_gpt-4', embedder='simple')
results = project.run_pipeline('prep')

# Each segment will be ≤ 8000 tokens
print(f"Split into {len(results['segments'])} chunks")
```

### 7. segtok (Simple Segmentation)

**Package:** `segtok`
**Installation:** `pip install segtok`

Fast, simple sentence and word segmentation:

- **Sentence segmenter** (`segtok_sentences`)
  - Fast sentence splitting
  - Good for simple cases

- **Word tokenizer** (`segtok_words`)
  - Simple word tokenization

**Example:**
```python
from ef import Project
from ef.plugins import segmenter_registry

project = Project.create('simple_processing')
segmenter_registry.register_segtok_segmenters(project)

project.add_source('text', 'Simple text. Fast processing.')
project.create_pipeline('analyze', segmenter='segtok_sentences', embedder='simple')
results = project.run_pipeline('analyze')
```

## Registration Functions

You can register segmenters from specific packages or all at once:

### Register All Segmenters

```python
from ef import Project
from ef.plugins import segmenter_registry

project = Project.create('my_project')

# Register all available segmenters (prints status for each package)
results = segmenter_registry.register_all_segmenters(project)

# results is a dict: {'langchain': 15, 'spacy': 2, 'nltk': 2, ...}
print(f"Registered {sum(results.values())} total segmenters")
```

### Register Specific Packages

```python
from ef import Project
from ef.plugins import segmenter_registry

project = Project.create('my_project')

# Register only LangChain segmenters
segmenter_registry.register_langchain_segmenters(project)

# Register only code segmenters
segmenter_registry.register_tree_sitter_segmenters(project)
segmenter_registry.register_ast_segmenters(project)

# Register only NLP segmenters
segmenter_registry.register_spacy_segmenters(project)
segmenter_registry.register_nltk_segmenters(project)
```

## Custom Segmenters

You can still register custom segmenters using the decorator pattern:

```python
from ef import Project

project = Project.create('my_project')

@project.segmenters.register('my_custom', max_length=100)
def custom_segmenter(source):
    """Segment text by fixed character length."""
    if isinstance(source, dict):
        source = '\n\n'.join(source.values())

    segments = {}
    for i in range(0, len(source), 100):
        segments[f'chunk_{i//100}'] = source[i:i+100]

    return segments

# Use it
project.add_source('text', 'A' * 250)
project.create_pipeline('test', segmenter='my_custom', embedder='simple')
results = project.run_pipeline('test')
```

## Choosing a Segmenter

Here's a guide for choosing the right segmenter:

| Use Case | Recommended Segmenter | Reason |
|----------|----------------------|--------|
| General text, semantic coherence | `langchain_recursive_1000` | Tries multiple separators for better splits |
| Fixed-size chunks for LLMs | `tiktoken_gpt-4` | Respects token limits |
| Markdown documentation | `langchain_markdown` | Respects document structure |
| Python code analysis | `tree_sitter_python` or `ast_python` | Segments by functions/classes |
| Multi-language code | `tree_sitter_<lang>` | Language-aware parsing |
| NLP sentence analysis | `spacy_sentences` | Best sentence boundary detection |
| Simple sentence splitting | `nltk_sentences` or `segtok_sentences` | Fast, reliable |
| Academic/research NLP | `nltk_sentences` | Standard in research |

## Inspecting Segmenter Metadata

Each segmenter is registered with metadata:

```python
from ef import Project
from ef.plugins import segmenter_registry

project = Project.create('my_project')
segmenter_registry.register_all_segmenters(project)

# Get metadata for a segmenter
metadata = project.segmenters.get_metadata('langchain_recursive_1000')
print(metadata)
# Output: {'package': 'langchain-text-splitters',
#          'description': 'Recursive character splitter with 1000 char chunks...',
#          'chunk_size': 1000, 'chunk_overlap': 200}

# List all segmenters
all_segmenters = project.list_components()['segmenters']
print(f"Available: {', '.join(all_segmenters)}")
```

## Error Handling

All registrations are conditional and fail gracefully:

```python
from ef import Project
from ef.plugins import segmenter_registry

project = Project.create('my_project')

# This won't crash if packages aren't installed
results = segmenter_registry.register_all_segmenters(project, verbose=True)

# Output example:
# ○ LangChain text splitters not available: No module named 'langchain_text_splitters'
# ✓ Registered 2 AST-based segmenters
# ✓ Registered 2 spaCy segmenters
# ...
```

Symbols:
- `✓` Package available, segmenters registered
- `○` Package not available, skipped

## Performance Considerations

- **Fastest:** `segtok`, built-in regex segmenters (`lines`, `sentences`)
- **Fast:** `NLTK`, `spaCy` (after model loading)
- **Moderate:** `LangChain`, `ast`
- **Slower:** `tree-sitter` (comprehensive parsing)
- **Token-based:** `tiktoken` (depends on text length)

For large-scale processing, consider:
1. Using simpler segmenters when appropriate
2. Caching segmentation results
3. Processing in batches

## Installation Quick Reference

```bash
# Install all supported packages
pip install langchain-text-splitters  # LangChain splitters
pip install tree-sitter-languages      # Tree-sitter for code
pip install spacy                      # spaCy NLP
python -m spacy download en_core_web_sm  # spaCy model
pip install nltk                       # NLTK
pip install tiktoken                   # OpenAI tokenizer
pip install segtok                     # Simple segmentation

# Or install just what you need for your use case
```

## Examples

### Example 1: Documentation Processing

```python
from ef import Project
from ef.plugins import segmenter_registry

project = Project.create('docs')
segmenter_registry.register_langchain_segmenters(project)

# Process markdown documentation
with open('README.md') as f:
    readme = f.read()

project.add_source('readme', readme)
project.create_pipeline('doc_analysis',
                       segmenter='langchain_markdown',
                       embedder='simple')

results = project.run_pipeline('doc_analysis')
print(f"Split README into {len(results['segments'])} sections")
```

### Example 2: Code Repository Analysis

```python
from ef import Project
from ef.plugins import segmenter_registry
import glob

project = Project.create('codebase')
segmenter_registry.register_tree_sitter_segmenters(project)
segmenter_registry.register_ast_segmenters(project)

# Process all Python files
for filepath in glob.glob('**/*.py', recursive=True):
    with open(filepath) as f:
        code = f.read()

    source_key = filepath.replace('/', '_')
    project.add_source(source_key, code)

    # Segment by functions
    pipeline_name = f'analyze_{source_key}'
    project.create_pipeline(pipeline_name,
                           segmenter='tree_sitter_python',
                           embedder='simple')

    results = project.run_pipeline(pipeline_name)

    # Now you have each function as a separate segment
    for seg_key, segment in results['segments'].items():
        if 'function_definition' in seg_key:
            print(f"Found function in {filepath}: {seg_key}")
```

### Example 3: Multi-strategy Comparison

```python
from ef import Project
from ef.plugins import segmenter_registry

project = Project.create('comparison')
segmenter_registry.register_all_segmenters(project)

text = "Your text here. Multiple sentences. Several paragraphs."
project.add_source('text', text)

# Try different segmentation strategies
strategies = ['spacy_sentences', 'nltk_sentences', 'langchain_recursive_500']

for strategy in strategies:
    if strategy in project.list_components()['segmenters']:
        project.create_pipeline(f'test_{strategy}',
                               segmenter=strategy,
                               embedder='simple')

        results = project.run_pipeline(f'test_{strategy}')
        print(f"{strategy}: {len(results['segments'])} segments")
```

## Troubleshooting

### Import Errors

If you see import errors, install the missing package:

```bash
pip install <package-name>
```

### spaCy Model Not Found

If spaCy segmenters don't work:

```bash
python -m spacy download en_core_web_sm
```

### NLTK Data Not Found

If NLTK segmenters don't work:

```python
import nltk
nltk.download('punkt')
```

### Checking Available Segmenters

```python
from ef import Project
from ef.plugins import segmenter_registry

project = Project.create('test')
segmenter_registry.register_all_segmenters(project, verbose=True)

# See what's actually registered
available = project.list_components()['segmenters']
print(f"Available segmenters: {available}")
```

## Contributing

To add support for a new segmentation package:

1. Add a new registration function to `ef/plugins/segmenter_registry.py`
2. Follow the naming pattern: `register_<package>_segmenters(project) -> int`
3. Wrap the package's API to match ef's interface: `(source: Any) -> dict[str, str]`
4. Use try/except for conditional imports
5. Register with descriptive metadata
6. Add to `register_all_segmenters()`
7. Update this documentation

## See Also

- [Main README](../README.md) - Overview of ef framework
- [API Documentation](API.md) - Full API reference
- [LangChain Text Splitters](https://python.langchain.com/docs/concepts/text_splitters/) - LangChain documentation
- [Tree-sitter](https://tree-sitter.github.io/tree-sitter/) - Tree-sitter documentation
- [spaCy](https://spacy.io/) - spaCy documentation
