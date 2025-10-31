"""
Storage layer for ef using the Mall pattern (store of stores).

This module provides:
- Storage backends with MutableMapping interfaces
- Extension-based file stores
- Mall creation for project data
- Optional dol integration with fallbacks
"""

import os
import tempfile
from collections.abc import MutableMapping, Iterator
from typing import Any

# Optional dol import with fallback
try:
    from dol import Files, wrap_kvs, add_ipython_key_completions

    HAVE_DOL = True
except ImportError:
    HAVE_DOL = False

    def add_ipython_key_completions(obj):
        """Fallback: no-op decorator."""
        return obj


# ============================================================================
# File Storage Backends
# ============================================================================


class SimpleFileStore(MutableMapping):
    """
    Simple file-based storage (fallback when dol not available).

    Provides MutableMapping interface to filesystem storage.
    """

    def __init__(self, rootdir: str, extension: str = 'pkl'):
        self.rootdir = rootdir
        self.extension = extension
        os.makedirs(rootdir, exist_ok=True)

    def _filepath(self, key: str) -> str:
        return os.path.join(self.rootdir, f"{key}.{self.extension}")

    def __getitem__(self, key: str) -> Any:
        filepath = self._filepath(key)

        import pickle
        import json

        with open(filepath, 'rb') as f:
            data = f.read()

        if self.extension == 'pkl':
            return pickle.loads(data)
        elif self.extension == 'json':
            return json.loads(data.decode())
        else:
            return data.decode()

    def __setitem__(self, key: str, value: Any) -> None:
        filepath = self._filepath(key)

        import pickle
        import json

        if self.extension == 'pkl':
            data = pickle.dumps(value)
        elif self.extension == 'json':
            data = json.dumps(value).encode()
        else:
            data = str(value).encode()

        with open(filepath, 'wb') as f:
            f.write(data)

    def __delitem__(self, key: str) -> None:
        os.remove(self._filepath(key))

    def __iter__(self) -> Iterator[str]:
        for filename in os.listdir(self.rootdir):
            if filename.endswith(f'.{self.extension}'):
                yield filename.rsplit('.', 1)[0]

    def __len__(self) -> int:
        return sum(1 for _ in self)


def mk_extension_based_store(rootdir: str, *, extension: str = 'pkl') -> MutableMapping:
    """
    Create a storage backend with extension-based serialization.

    Uses dol if available, otherwise uses SimpleFileStore fallback.

    Args:
        rootdir: Base directory for storage
        extension: File extension ('pkl', 'json', 'txt')

    Returns:
        MutableMapping interface to storage

    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmpdir:
    ...     store = mk_extension_based_store(tmpdir, extension='json')
    ...     store['test'] = {'key': 'value'}
    ...     assert store['test'] == {'key': 'value'}
    """
    import pickle
    import json

    os.makedirs(rootdir, exist_ok=True)

    if HAVE_DOL:
        # Use dol for full functionality
        base_store = Files(rootdir)

        # Set up codec based on extension
        if extension == 'pkl':
            encode = pickle.dumps
            decode = pickle.loads
        elif extension == 'json':
            encode = lambda x: json.dumps(x).encode()
            decode = lambda x: json.loads(x.decode())
        else:
            encode = lambda x: str(x).encode()
            decode = lambda x: x.decode()

        # Key transformations to add/remove extension
        def _add_ext(k: str) -> str:
            return f"{k}.{extension}"

        def _remove_ext(k: str) -> str:
            return k.rsplit('.', 1)[0] if '.' in k else k

        store = wrap_kvs(
            base_store,
            key_of_id=_add_ext,
            id_of_key=_remove_ext,
            obj_of_data=decode,
            data_of_obj=encode,
        )

        return add_ipython_key_completions(store)
    else:
        # Use simple fallback
        return SimpleFileStore(rootdir, extension)


# ============================================================================
# Mall Pattern (Store of Stores)
# ============================================================================


def mk_project_mall(
    project_id: str,
    root_dir: str | None = None,
    *,
    backend: str = 'files',
) -> dict[str, MutableMapping]:
    """
    Create a "mall" (store of stores) for a project.

    A mall provides separate storage for each pipeline stage:
    - segments: text segments
    - embeddings: vector embeddings
    - planar_embeddings: 2D coordinates
    - clusters: cluster assignments

    Args:
        project_id: Unique identifier for the project
        root_dir: Base directory for storage (uses temp if None)
        backend: Storage backend ('files', 'memory')

    Returns:
        Dictionary mapping stage names to storage objects

    >>> mall = mk_project_mall('test_project', backend='memory')
    >>> list(mall.keys())
    ['segments', 'embeddings', 'planar_embeddings', 'clusters']
    """
    if root_dir is None:
        root_dir = os.path.join(tempfile.gettempdir(), 'ef_projects', project_id)

    if backend == 'memory':
        # Use simple dicts for in-memory storage
        return {
            'segments': {},
            'embeddings': {},
            'planar_embeddings': {},
            'clusters': {},
        }
    elif backend == 'files':
        # Use filesystem storage with appropriate serialization
        return {
            'segments': mk_extension_based_store(
                os.path.join(root_dir, 'segments'), extension='txt'
            ),
            'embeddings': mk_extension_based_store(
                os.path.join(root_dir, 'embeddings'), extension='pkl'
            ),
            'planar_embeddings': mk_extension_based_store(
                os.path.join(root_dir, 'planar_embeddings'), extension='json'
            ),
            'clusters': mk_extension_based_store(
                os.path.join(root_dir, 'clusters'), extension='json'
            ),
        }
    else:
        raise ValueError(f"Unknown backend: {backend}")
