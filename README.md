# ef (Embedding Flow)

**Lightweight framework for embedding pipelines**

ef is a simple, composable framework for building and running embedding pipelines. It provides:
- ✅ Works out-of-the-box (zero configuration, built-in components)
- ✅ Component registries as mapping stores (easy discovery)
- ✅ Automatic pipeline composition (via DAG)
- ✅ Flexible storage backends (memory, files, custom)
- ✅ Plugin system (add production components when needed)

## Installation

```bash
# Basic installation (works immediately with built-in components)
pip install ef

# With full functionality (dol, meshed, larder)
pip install ef[full]

# With imbed integration (production components)
pip install ef[imbed]
```

## Quick Start

```python
from ef import Project

# Create project (works immediately!)
project = Project.create('my_project', backend='memory')

# Add data
project.add_source('doc1', 'First document about AI')
project.add_source('doc2', 'Second document about ML')

# List available components
print(project.list_components())
# {
#   'embedders': ['simple', 'char_counts'],
#   'planarizers': ['simple_2d', 'normalize_2d'],
#   'clusterers': ['simple_kmeans', 'threshold'],
#   'segmenters': ['identity', 'lines', 'sentences']
# }

# Create pipeline
project.create_pipeline(
    'analysis',
    embedder='simple',
    planarizer='simple_2d',
    clusterer='simple_kmeans',
    n_clusters=2
)

# Run pipeline (persists all results automatically)
results = project.run_pipeline('analysis')

# Access persisted data
print(f"Segments: {len(project.segments)}")
print(f"Embeddings: {len(project.embeddings)}")
print(f"Clusters: {dict(project.clusters)}")

# Get project summary
print(project.summary())
```

## Core Concepts

### 1. Component Registries (Mapping Stores)

Components are stored in registries that behave like dictionaries:

```python
# Access components like a dict
embedder = project.embedders['simple']
vectors = embedder({'text1': 'Sample text'})

# List all components
print(list(project.embedders.keys()))

# Get component metadata
meta = project.embedders.get_metadata('simple')
```

### 2. Mall Pattern (Store of Stores)

Each project has a "mall" - separate stores for each data type:

```python
# Access different stores
project.segments['doc1'] = 'text'
project.embeddings['doc1'] = [1.0, 2.0, 3.0]
project.clusters['doc1'] = 0

# All stores use MutableMapping interface
for key, value in project.embeddings.items():
    print(f"{key}: {value}")
```

### 3. Pipeline Assembly

Pipelines are assembled automatically from components:

```python
# Create pipeline by naming components
project.create_pipeline(
    'my_pipeline',
    segmenter='lines',      # Optional: split text
    embedder='simple',      # Required: embed segments
    planarizer='simple_2d', # Optional: reduce dimensions
    clusterer='simple_kmeans',  # Optional: cluster
    n_clusters=5  # Pass parameters to components
)

# Run with automatic persistence
results = project.run_pipeline('my_pipeline')
```

### 4. Flexible Storage

Choose storage backend based on needs:

```python
# In-memory (fast, temporary)
project = Project.create('test', backend='memory')

# File-based (persistent)
project = Project.create('prod', backend='files', root_dir='/data')

# Custom (bring your own store)
from ef.storage import mk_project_mall
mall = mk_project_mall('custom', backend='files')
mall['embeddings'] = MyCustomStore()  # Any MutableMapping
project = Project('custom', mall=mall)
```

## Plugin System

### Built-in Components (Always Available)

ef comes with simple implementations that work without dependencies:

```python
# Automatically registered on import
from ef import Project
project = Project.create('test')

# Has built-in components:
# - Embedders: simple, char_counts
# - Planarizers: simple_2d, normalize_2d
# - Clusterers: simple_kmeans, threshold
# - Segmenters: identity, lines, sentences
```

### Adding Production Components

Use plugins to add real ML implementations:

```python
from ef import Project
from ef.plugins import imbed  # Requires: pip install ef[imbed]

project = Project.create('production')
imbed.register(project)

# Now has production components:
# - OpenAI embedders
# - UMAP planarization
# - scikit-learn clustering
# - And more...

print(list(project.embedders.keys()))
# ['simple', 'char_counts', 'openai-small', 'openai-large', ...]
```

### External Segmenter Registration

ef provides conditional registration for advanced segmenters from popular packages:

```python
from ef import Project
from ef.plugins import segmenter_registry

project = Project.create('my_project')

# Register all available segmenters (conditional on package installation)
segmenter_registry.register_all_segmenters(project)

# Now you can use advanced segmenters in pipelines
project.create_pipeline(
    'analysis',
    segmenter='langchain_recursive_1000',  # LangChain smart chunking
    embedder='simple'
)
```

**Supported packages** (all optional, automatically skipped if not installed):

- **LangChain** (`langchain-text-splitters`) - Character, recursive, markdown, code splitters
- **Tree-sitter** (`tree-sitter-languages`) - Multi-language code parsing
- **AST** (built-in) - Python code segmentation
- **spaCy** (`spacy`) - NLP sentence/token segmentation
- **NLTK** (`nltk`) - Classic NLP tokenization
- **tiktoken** (`tiktoken`) - OpenAI tokenizer for token-based chunking
- **segtok** (`segtok`) - Fast sentence/word segmentation

**Example with LangChain markdown splitter:**

```python
project.add_source('docs', open('README.md').read())
project.create_pipeline('doc_pipeline',
                       segmenter='langchain_markdown',
                       embedder='simple')
results = project.run_pipeline('doc_pipeline')
```

**Example with Python code segmentation:**

```python
code = '''
def hello():
    print("Hello")

class Greeter:
    def greet(self):
        return "Hi"
'''

project.add_source('module', code)
project.create_pipeline('code_pipeline',
                       segmenter='tree_sitter_python',  # or 'ast_python'
                       embedder='simple')
results = project.run_pipeline('code_pipeline')
# Each function/class becomes a separate segment
```

See [docs/SEGMENTER_REGISTRATION.md](docs/SEGMENTER_REGISTRATION.md) for complete documentation on all available segmenters, installation instructions, and usage examples.

### Writing Your Own Plugin

```python
# my_plugin.py
def register(project):
    """Add custom components to project."""

    @project.embedders.register('my_embedder', dimension=768)
    def my_embedder(segments):
        # Your implementation
        return {key: compute(text) for key, text in segments.items()}

# Use it
from ef import Project
import my_plugin

project = Project.create('custom')
my_plugin.register(project)
```

## Advanced Usage

### Multiple Projects

```python
from ef import Projects

# Manage multiple projects
projects = Projects(root_dir='/data')

# Create projects
proj1 = projects.create_project('research', backend='files')
proj2 = projects.create_project('production', backend='files')

# Access later
existing = projects['research']

# List all
print(list(projects.keys()))
```

### Quick Embed (No Pipeline)

```python
# For one-off embeddings
embeddings = project.quick_embed(
    'Some text to embed',
    embedder='simple'
)
```

### Custom Components

```python
# Register your own component
@project.embedders.register('custom', dimension=512)
def custom_embedder(segments):
    return {k: my_model(v) for k, v in segments.items()}

# Use in pipeline
project.create_pipeline('custom_pipe', embedder='custom')
```

## Architecture

ef follows Option 1 from the design plan:

```
┌─────────────────────────────────────┐
│  ef (lightweight interface layer)   │
│  - ComponentRegistry                │
│  - Project/Projects                 │
│  - Mall pattern                     │
│  - Pipeline assembly                │
│  - Built-in toy components          │
└──────────────┬──────────────────────┘
               │ imports (optional)
               ↓
┌─────────────────────────────────────┐
│  imbed (heavy implementation)       │
│  - Real embedders (OpenAI, etc.)    │
│  - Real planarizers (UMAP)          │
│  - Real clusterers (sklearn)        │
│  - Dataset classes                  │
│  - All utilities                    │
└─────────────────────────────────────┘
```

## Design Principles

1. **Works immediately**: Built-in components require no setup
2. **Mapping everywhere**: All stores use `MutableMapping` interface
3. **Composable**: Mix and match components easily
4. **Discoverable**: `.list_components()`, `.list_pipelines()`
5. **Flexible**: Swap storage backends without code changes
6. **Extensible**: Plugin system for adding functionality
7. **Progressive enhancement**: Start simple, add complexity as needed

## Dependencies

**Required (minimal):**
- Python 3.10+
- numpy

**Optional (recommended):**
- `dol>=0.2.38` - Better storage abstraction
- `meshed>=0.1.20` - Automatic DAG composition
- `larder>=0.1.6` - Automatic caching

**Plugin dependencies:**
- `imbed>=0.1` - Production ML components

Install optional dependencies:
```bash
pip install ef[full]     # Install dol, meshed, larder
pip install ef[imbed]    # Install imbed + full dependencies
```

## Development

```bash
# Clone repository
git clone https://github.com/thorwhalen/ef.git
cd ef

# Install in development mode
pip install -e .

# Run tests (if available)
pytest
```

## Examples

See the `imbed_refactored/` directory for detailed examples:
- `imbed_refactored.py` - Core patterns and complete demo
- `advanced_example.py` - Real ML integrations (OpenAI, UMAP, sklearn)
- `persistence_examples.py` - Pipeline sharing and caching

## Comparison with imbed

| Feature | ef | imbed |
|---------|----|----|
| **Purpose** | Lightweight interface framework | Production ML implementations |
| **Dependencies** | numpy (+ optional) | openai, umap, sklearn, datasets, etc. |
| **Out-of-box** | ✓ Works immediately | Requires configuration |
| **Components** | Toy implementations | Production implementations |
| **Use case** | Prototyping, learning, interfaces | Production ML pipelines |

**Use together:**
```python
from ef import Project
from ef.plugins import imbed

project = Project.create('best_of_both')
imbed.register(project)  # Add production power to clean interface
```

## License

MIT

## Contributing

Contributions welcome! Please:
1. Write tests for new features
2. Follow existing code style
3. Update documentation
4. Submit PRs to main branch

## Links

- **GitHub**: https://github.com/thorwhalen/ef
- **imbed**: https://github.com/thorwhalen/imbed (production components)
- **dol**: https://github.com/i2mint/dol (storage layer)
- **meshed**: https://github.com/i2mint/meshed (DAG composition) -- Tools for workflows involving semantic embeddings
