# Writing ef Plugins

ef plugins extend the framework with additional components (embedders, planarizers, clusterers, segmenters).

## Built-in Plugins

### simple (Built-in)

Toy implementations that work out-of-the-box without dependencies.

```python
from ef import Project
from ef.plugins import simple

project = Project.create('test', backend='memory')
simple.register_simple_components(project)

# Now has toy implementations
print(project.embedders.keys())
# ['simple', 'char_counts']
```

**Components:**
- **Segmenters**: `identity`, `lines`, `sentences`
- **Embedders**: `simple` (char/word/punct counts), `char_counts` (26D letter frequencies)
- **Planarizers**: `simple_2d`, `normalize_2d`
- **Clusterers**: `simple_kmeans`, `threshold`

### imbed (Optional)

Bridge to production imbed package with real ML implementations.

```python
from ef import Project
from ef.plugins import imbed

project = Project.create('production')
imbed.register(project)

# Now has production components
print(project.embedders.keys())
# ['openai-small', 'openai-large', ...]
```

**Requires:** `pip install imbed` or `pip install ef[imbed]`

## Writing Your Own Plugin

Create a new plugin module in `ef/plugins/`:

```python
# ef/plugins/my_plugin.py

def register(project):
    """Register all components from this plugin."""
    _register_embedders(project)
    _register_clusterers(project)


def _register_embedders(project):
    """Add custom embedders."""
    
    @project.embedders.register('my_embedder', dimension=768)
    def my_embedder(segments):
        """My custom embedding implementation."""
        # Your code here
        return {key: compute_embedding(text) for key, text in segments.items()}


def _register_clusterers(project):
    """Add custom clusterers."""
    
    @project.clusterers.register('my_clusterer')
    def my_clusterer(embeddings, *, n_clusters=5):
        """My custom clustering implementation."""
        # Your code here
        return {key: assign_cluster(vec) for key, vec in embeddings.items()}
```

Then use it:

```python
from ef import Project
from ef.plugins import my_plugin

project = Project.create('custom')
my_plugin.register(project)
```

## Plugin Guidelines

1. **Registration Function**: Provide a `register(project)` function
2. **Decorators**: Use `@project.{component_type}.register(name)` to add components
3. **Metadata**: Include metadata in registration (e.g., `dimension=768`)
4. **Signatures**: Follow standard signatures:
   - Segmenter: `(source: str) -> dict[str, str]`
   - Embedder: `(segments: dict[str, str]) -> dict[str, list[float]]`
   - Planarizer: `(embeddings: dict[str, list[float]]) -> dict[str, tuple[float, float]]`
   - Clusterer: `(embeddings: dict[str, list[float]], **kwargs) -> dict[str, int]`
5. **Dependencies**: Import heavy dependencies inside functions (lazy loading)
6. **Error Handling**: Provide clear error messages if dependencies missing

## Example: Sentence Transformers Plugin

```python
# ef/plugins/sentence_transformers_plugin.py

def register(project):
    """Register sentence-transformers models."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers required: "
            "pip install sentence-transformers"
        )
    
    @project.embedders.register('all-MiniLM-L6-v2', dimension=384)
    def minilm_embedder(segments):
        """Efficient sentence embedder."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        keys = list(segments.keys())
        texts = list(segments.values())
        
        embeddings = model.encode(texts)
        
        return {key: emb.tolist() for key, emb in zip(keys, embeddings)}
```

## Distribution

To distribute your plugin:

1. **Standalone Package**:
   ```bash
   # my_ef_plugin/
   # ├── setup.py
   # └── my_ef_plugin.py
   
   pip install my-ef-plugin
   ```

2. **Include in ef.plugins** (for official plugins):
   - Add to `ef/plugins/`
   - Add to `ef/plugins/__init__.py`
   - Add optional dependency to `setup.cfg`

## Testing Plugins

```python
import pytest
from ef import Project

def test_my_plugin():
    """Test plugin registration."""
    from ef.plugins import my_plugin
    
    project = Project.create('test', backend='memory')
    my_plugin.register(project)
    
    # Check components registered
    assert 'my_embedder' in project.embedders
    
    # Test functionality
    project.add_source('test', 'Sample text')
    results = project.quick_embed('Sample text', embedder='my_embedder')
    
    assert 'main' in results
    assert len(results['main']) == 768  # Check dimension
```
