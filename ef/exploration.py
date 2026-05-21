"""Layer L5 — *explore* a corpus: project to 2-D/3-D, cluster, label clusters.

This is ``ef``'s visualization heritage, inherited from ``imbed`` and
re-presented as a **secondary** surface. The *primary* surface of ``ef`` is
search / RAG / corpus-indexing (see :func:`ef.ingest` and
:class:`~ef.source_manager.SourceManager`); ``explore`` is the "see the shape of
the corpus" tool — the backend an ``app_ef`` corpus map consumes.

Four free functions, each accepting a corpus *or* an already-computed vector
matrix (whatever :func:`_resolve_vectors` understands):

- :func:`project` — reduce embeddings to 2-D/3-D coordinates (PCA → UMAP).
- :func:`cluster` — group embeddings into clusters (k-means / HDBSCAN).
- :func:`label_clusters` — name clusters with an LLM (via ``imbed``).
- :func:`explore` — the orchestrator: project + cluster (+ optionally label) in
  one call, returning a structured, JSON-friendly :class:`ExploreResult`
  (``ids`` / ``coords`` / ``labels`` / ``cluster_titles``) — the surface an
  ``app_ef`` corpus map (or :meth:`ef.service.EfService.explore_corpus`)
  consumes directly.

The module imports with **numpy only**. UMAP, HDBSCAN and ``imbed`` are
optional: each is imported lazily, inside the function that needs it, and only
that heavy path requires the corresponding extra (``ef[explore]`` /
``ef[imbed]``). The numpy-only paths — PCA projection and k-means clustering —
work out of the box with no install.

Example — project a small set of vectors to the plane (numpy only):

>>> import numpy as np
>>> vectors = np.random.RandomState(0).rand(12, 16)
>>> coords = project(vectors, dims=2, method='pca')
>>> coords.shape
(12, 2)
>>> labels = cluster(vectors, method='kmeans', n_clusters=3, random_state=0)
>>> labels.shape
(12,)

Each returned array's rows are in the iteration order of the input — for a
:class:`~ef.source_manager.SearchableCorpus` that is ``list(searchable.collection)``,
which the caller zips back against to recover ``{id: coords}``. :func:`explore`
does that zip-back for you — its :class:`ExploreResult` keeps ``ids`` aligned
with ``coords`` and ``labels``.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping
from typing import Any, Literal, TypedDict

import numpy as np

from ef.embedder_adapters import as_embedder
from ef.source_manager import DEFAULT_EMBEDDER

__all__ = ["project", "cluster", "label_clusters", "explore", "ExploreResult"]

# --- knobs (no magic numbers buried in the functions) ----------------------

ProjectionMethod = Literal["auto", "umap", "pca"]
ClusterMethod = Literal["kmeans", "hdbscan"]

DEFAULT_DIMS = 2
DEFAULT_N_NEIGHBORS = 15
DEFAULT_MIN_DIST = 0.1
DEFAULT_PCA_COMPONENTS = 50
DEFAULT_RANDOM_STATE = 42
DEFAULT_N_CLUSTERS = 8
DEFAULT_MIN_CLUSTER_SIZE = 5
UMAP_METRIC = "cosine"

#: ``method='auto'`` uses PCA (not UMAP) below this many samples — UMAP is
#: degenerate on a handful of points, and seeds the doctest/light path.
_MIN_SAMPLES_FOR_UMAP = 5


# --- the structured result -------------------------------------------------


class ExploreResult(TypedDict):
    """A structured, JSON-friendly corpus-exploration result.

    What :func:`explore` returns and :meth:`ef.service.EfService.explore_corpus`
    serves. Every list is **row-aligned**: ``ids[i]``, ``coords[i]`` and
    ``labels[i]`` all describe the same item.

    Keys:
        ids: the per-item identifiers — a corpus's keys, or positional ``"0"``,
            ``"1"``, … for an id-less vector matrix.
        coords: the projected coordinates — one ``[x, y]`` (or ``[x, y, z]``)
            row per item.
        labels: the cluster id of each item (HDBSCAN marks noise ``-1``).
        cluster_titles: ``{cluster_id: title}`` — empty unless :func:`explore`
            was called with ``label=True``.
    """

    ids: list[str]
    coords: list[list[float]]
    labels: list[int]
    cluster_titles: dict[int, str]


# ===========================================================================
# Public surface
# ===========================================================================


def project(
    data: Any,
    *,
    dims: int = DEFAULT_DIMS,
    method: ProjectionMethod = "auto",
    embedder: Any = None,
    n_neighbors: int = DEFAULT_N_NEIGHBORS,
    min_dist: float = DEFAULT_MIN_DIST,
    metric: str = UMAP_METRIC,
    pca_components: int = DEFAULT_PCA_COMPONENTS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    """Project a corpus's embeddings to ``dims`` coordinates for visualization.

    The canonical "show me the shape of the corpus" operation (use cases §G1).
    The default ``method='auto'`` runs the design-notes §8.4 recipe — **PCA →
    UMAP**, ``cosine`` metric, seeded: PCA first compresses high-dimensional
    embeddings to ``pca_components`` (cheap denoising), then UMAP lays them out
    in ``dims`` dimensions.

    Args:
        data: what to project — anything :func:`_resolve_vectors` understands:
            an ``ndarray`` (or nested sequence) of vectors, an iterable of
            texts (embedded with ``embedder``), a :class:`~ef.corpus.Corpus`
            mapping, or a :class:`~ef.source_manager.SearchableCorpus` /
            :class:`~ef.source_manager.SourceManager` (vectors pulled from the
            index).
        dims: target dimensionality — ``2`` or ``3`` for plotting.
        method: ``'auto'`` (UMAP when available and the corpus is large enough,
            else PCA), ``'umap'`` (force UMAP — needs the ``ef[explore]``
            extra), or ``'pca'`` (numpy-only, always available).
        embedder: embedder used when ``data`` is raw text; defaults to the
            dependency-free :data:`~ef.source_manager.DEFAULT_EMBEDDER`.
        n_neighbors: UMAP neighborhood size (clamped to ``n_samples - 1``).
        min_dist: UMAP minimum point separation in the embedding.
        metric: UMAP distance metric — ``'cosine'`` for semantic embeddings.
        pca_components: PCA width for the pre-UMAP compression step; PCA is
            skipped when the source dimensionality is already smaller.
        random_state: seed — projection is reproducible.

    Returns:
        An ``ndarray`` of shape ``(n_samples, dims)``, rows in input order.

    Raises:
        ValueError: if there are fewer than 2 samples.
        ImportError: if ``method='umap'`` and ``umap-learn`` is not installed.

    >>> import numpy as np
    >>> vectors = np.random.RandomState(1).rand(20, 32)
    >>> project(vectors, dims=3, method='pca').shape
    (20, 3)
    """
    if dims < 1:
        raise ValueError(f"dims must be >= 1, got {dims}")
    vectors = _resolve_vectors(data, embedder=embedder)
    n_samples, source_dim = vectors.shape
    if n_samples < 2:
        raise ValueError(f"project() needs >= 2 samples, got {n_samples}")

    if _choose_projection(method, n_samples=n_samples) == "pca":
        return _pca(vectors, dims)

    # PCA → UMAP: compress first (denoise + speed up), then lay out.
    reduced = vectors
    if source_dim > pca_components:
        reduced = _pca(vectors, min(pca_components, n_samples - 1))
    umap = _import_umap()
    reducer = umap.UMAP(
        n_components=dims,
        n_neighbors=min(n_neighbors, n_samples - 1),
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return np.asarray(reducer.fit_transform(reduced), dtype=float)


def cluster(
    data: Any,
    *,
    method: ClusterMethod = "kmeans",
    n_clusters: int = DEFAULT_N_CLUSTERS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    embedder: Any = None,
    normalize: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    """Cluster a corpus's embeddings (use cases §G2).

    ``method='kmeans'`` is the default and is **dependency-free** — a numpy
    Lloyd's algorithm with k-means++ seeding. ``method='hdbscan'`` finds a
    density-based clustering (and a noise label ``-1``) and needs the
    ``ef[explore]`` extra.

    Args:
        data: what to cluster — see :func:`project` / :func:`_resolve_vectors`.
        method: ``'kmeans'`` (numpy-only) or ``'hdbscan'`` (optional extra).
        n_clusters: number of clusters for k-means (clamped to ``n_samples``).
        min_cluster_size: smallest cluster HDBSCAN will form.
        embedder: embedder used when ``data`` is raw text.
        normalize: L2-normalize the vectors first, so Euclidean distance
            behaves like cosine similarity — the right default for semantic
            embeddings. Set ``False`` to cluster raw vectors.
        random_state: seed for k-means initialization.

    Returns:
        An ``ndarray`` of integer cluster labels, shape ``(n_samples,)``, in
        input order. HDBSCAN labels unclustered points ``-1``.

    >>> import numpy as np
    >>> vectors = np.random.RandomState(2).rand(15, 8)
    >>> labels = cluster(vectors, method='kmeans', n_clusters=3, random_state=0)
    >>> labels.shape
    (15,)
    >>> set(labels.tolist()) <= {0, 1, 2}
    True
    """
    vectors = _resolve_vectors(data, embedder=embedder)
    if normalize:
        vectors = _l2_normalize(vectors)
    if method == "kmeans":
        return _kmeans(vectors, n_clusters, random_state=random_state)
    if method == "hdbscan":
        return _hdbscan(vectors, min_cluster_size=min_cluster_size)
    raise ValueError(
        f"unknown clustering method {method!r} — use 'kmeans' or 'hdbscan'"
    )


def label_clusters(
    segments: Iterable[Any],
    labels: Iterable[Any],
    *,
    context: str = " ",
    n_words: int = 4,
    n_samples: int | None = None,
    **labeler_kwargs: Any,
) -> dict[int, str]:
    """Name each cluster with a short LLM-generated title (use cases §G3).

    A thin wrapper over ``imbed``'s :class:`imbed.tools.ClusterLabeler`: it
    samples each cluster's segments, asks an LLM for a few-word title, and
    returns ``{cluster_id: title}``. ``ef`` does not reimplement this — it
    reuses ``imbed`` (design notes §7).

    Args:
        segments: the segment texts — plain strings or :class:`~ef.segments.Segment`
            mappings (the ``"text"`` field is used). Aligned with ``labels``.
        labels: the cluster id of each segment — typically the output of
            :func:`cluster`.
        context: the corpus's overall topic, given to the LLM so titles
            describe how a cluster *differs* from the rest, not the shared topic.
        n_words: maximum title length, in words.
        n_samples: how many segments to sample per cluster when prompting;
            ``None`` keeps ``ClusterLabeler``'s default.
        **labeler_kwargs: forwarded to :class:`imbed.tools.ClusterLabeler`
            (e.g. ``max_unique_clusters``, ``prompt``).

    Returns:
        ``{cluster_id: title}`` for every distinct label.

    Raises:
        ValueError: if ``segments`` and ``labels`` differ in length.
        ImportError: if ``imbed`` (and its ``pandas``/``oa`` deps) is missing.

    >>> # needs ef[imbed] + an LLM key — illustrative only:
    >>> # label_clusters(['neural net training ...', ...], [0, 1, 0, ...])
    >>> # -> {0: 'Neural network training', 1: 'Dataset preprocessing'}
    """
    texts = [_segment_text(s) for s in segments]
    label_list = [_as_int(x) for x in labels]
    if len(texts) != len(label_list):
        raise ValueError(
            f"segments and labels differ in length: {len(texts)} != {len(label_list)}"
        )
    pd, ClusterLabeler = _import_cluster_labeler()
    frame = pd.DataFrame({"segment": texts, "cluster_idx": label_list})
    kwargs: dict[str, Any] = dict(context=context, n_words=n_words, **labeler_kwargs)
    if n_samples is not None:
        kwargs["n_samples"] = n_samples
    titles = ClusterLabeler(**kwargs).label_clusters(frame)
    return {_as_int(cluster_id): title for cluster_id, title in titles.items()}


def explore(
    data: Any,
    *,
    dims: int = DEFAULT_DIMS,
    projection_method: ProjectionMethod = "auto",
    cluster_method: ClusterMethod = "kmeans",
    n_clusters: int = DEFAULT_N_CLUSTERS,
    min_cluster_size: int = DEFAULT_MIN_CLUSTER_SIZE,
    label: bool = False,
    context: str = " ",
    n_words: int = 4,
    embedder: Any = None,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> ExploreResult:
    """Project, cluster and (optionally) label a corpus in one call.

    The layer-L5 orchestrator — :func:`project` + :func:`cluster` (+ optionally
    :func:`label_clusters`) wired into one structured result. Where
    :func:`project` / :func:`cluster` return *bare arrays in input order*,
    ``explore`` keeps every row tied to its id: it returns an
    :class:`ExploreResult` with ``ids`` / ``coords`` / ``labels`` row-aligned
    (plus ``cluster_titles``) — the JSON-friendly shape an ``app_ef`` corpus map
    (or :meth:`ef.service.EfService.explore_corpus`) consumes directly.

    Args:
        data: the corpus to explore — anything :func:`_resolve_explorable`
            understands (a :class:`~ef.source_manager.SourceManager` /
            :class:`~ef.source_manager.SearchableCorpus`, a
            :class:`~ef.corpus.Corpus` mapping, an iterable of texts or
            :class:`~ef.segments.Segment`\\ s, or a vector matrix).
        dims: projection target dimensionality — ``2`` or ``3``.
        projection_method: forwarded to :func:`project` — ``'auto'`` /
            ``'umap'`` / ``'pca'``.
        cluster_method: forwarded to :func:`cluster` — ``'kmeans'`` /
            ``'hdbscan'``.
        n_clusters: number of k-means clusters.
        min_cluster_size: smallest cluster HDBSCAN will form.
        label: when ``True``, name each cluster with :func:`label_clusters`
            (needs the ``ef[imbed]`` extra, an LLM key, and text-bearing
            ``data``); when ``False`` (the default) ``cluster_titles`` is empty.
        context: the corpus topic passed to :func:`label_clusters`.
        n_words: maximum cluster-title length, in words.
        embedder: embedder used when ``data`` is raw text.
        random_state: seed — projection and clustering are reproducible.

    Returns:
        an :class:`ExploreResult` — ``ids``, ``coords`` and ``labels``
        row-aligned, and ``cluster_titles`` (empty unless ``label=True``).

    Raises:
        ValueError: if the corpus is empty, has fewer than 2 samples, or
            ``label=True`` is asked of a text-less vector matrix.

    >>> import numpy as np
    >>> vectors = np.random.RandomState(0).rand(12, 16)
    >>> result = explore(vectors, projection_method='pca', n_clusters=3, random_state=0)
    >>> len(result['ids']), len(result['coords']), len(result['labels'])
    (12, 12, 12)
    >>> result['ids'][:3]
    ['0', '1', '2']
    >>> len(result['coords'][0])                 # each row has `dims` coordinates
    2
    >>> set(result['labels']) <= {0, 1, 2}
    True
    >>> result['cluster_titles']                 # empty unless label=True
    {}
    """
    ids, texts, vectors = _resolve_explorable(data, embedder=embedder)
    # Fail fast: labelling needs the per-row texts — check before any costly
    # projection so a bad request does not pay for a UMAP layout it discards.
    if label and texts is None:
        raise ValueError(
            "explore(label=True) needs text — pass a corpus, searchable or "
            "iterable of texts/segments, not a bare vector matrix."
        )
    coords = project(
        vectors, dims=dims, method=projection_method, random_state=random_state
    )
    labels = cluster(
        vectors,
        method=cluster_method,
        n_clusters=n_clusters,
        min_cluster_size=min_cluster_size,
        random_state=random_state,
    )
    cluster_titles: dict[int, str] = {}
    if label and texts is not None:
        cluster_titles = label_clusters(texts, labels, context=context, n_words=n_words)
    return ExploreResult(
        ids=ids,
        coords=[[float(x) for x in row] for row in coords],
        labels=[int(x) for x in labels],
        cluster_titles=cluster_titles,
    )


# ===========================================================================
# Input resolution — corpus | vectors -> ndarray(n, dim)
# ===========================================================================


def _resolve_explorable(
    data: Any, *, embedder: Any = None
) -> tuple[list[str], list[str] | None, np.ndarray]:
    """Coerce ``data`` to ``(ids, texts, vectors)`` — the input :func:`explore` needs.

    The dispatch seam for layer L5. ``vectors`` is an ``(n_samples, dim)`` float
    array; ``ids`` are the per-row identifiers (a corpus's keys, or positional
    ``"0"``, ``"1"``, … for an id-less matrix); ``texts`` are the per-row texts
    when ``data`` carries them (a corpus / searchable / iterable of texts or
    segments) and ``None`` when it does not (a bare vector matrix —
    :func:`label_clusters` is then unavailable).

    Understands the same inputs as :func:`_resolve_vectors`: a
    :class:`~ef.source_manager.SearchableCorpus` / :class:`~ef.source_manager.SourceManager`,
    an ``ndarray`` / nested sequence of numbers, an iterable of texts or
    :class:`~ef.segments.Segment`\\ s, and a :class:`~ef.corpus.Corpus` mapping.

    >>> ids, texts, vectors = _resolve_explorable(['hello world', 'a b c'])
    >>> ids, vectors.shape[0]
    (['0', '1'], 2)
    """
    # A SearchableCorpus — has an indexed `vd` collection + a query embedder.
    if hasattr(data, "collection") and hasattr(data, "embedder"):
        return _explorable_from_searchable(data)
    # A SourceManager — get its default searchable view, then pull vectors.
    if callable(getattr(data, "searchable", None)):
        return _explorable_from_searchable(data.searchable())
    # A bare vector matrix — positional ids, no text.
    if isinstance(data, np.ndarray):
        vectors = _as_2d(data)
        return (_positional_ids(len(vectors)), None, vectors)
    # A Corpus mapping — its keys are the ids, its values the source texts.
    if isinstance(data, Mapping):
        ids = [str(key) for key in data]
        texts = [_segment_text(value) for value in data.values()]
        return (ids, texts, _embed_texts(texts, embedder=embedder))

    items = list(data)
    if not items:
        raise ValueError("cannot explore an empty corpus")
    first = items[0]
    if isinstance(first, str):
        texts = [str(item) for item in items]
        return (
            _positional_ids(len(texts)),
            texts,
            _embed_texts(texts, embedder=embedder),
        )
    if isinstance(first, Mapping):  # an iterable of Segments
        ids = [str(seg.get("id", index)) for index, seg in enumerate(items)]
        texts = [_segment_text(seg) for seg in items]
        return (ids, texts, _embed_texts(texts, embedder=embedder))
    # Otherwise: a sequence of numeric vectors — positional ids, no text.
    vectors = _as_2d(items)
    return (_positional_ids(len(vectors)), None, vectors)


def _resolve_vectors(data: Any, *, embedder: Any = None) -> np.ndarray:
    """Coerce ``data`` to a float ``ndarray`` of shape ``(n_samples, dim)``.

    The vector-only view of :func:`_resolve_explorable` — what :func:`project`
    and :func:`cluster` need (they ignore the ids and texts).

    >>> _resolve_vectors(['hello world', 'goodbye world', 'a b c']).shape[0]
    3
    """
    return _resolve_explorable(data, embedder=embedder)[2]


def _explorable_from_searchable(
    searchable: Any,
) -> tuple[list[str], list[str], np.ndarray]:
    """Pull ``(ids, texts, vectors)`` out of a ``SearchableCorpus``'s collection."""
    keys = list(searchable.collection)
    if not keys:
        raise ValueError("cannot explore an empty SearchableCorpus")
    ids: list[str] = []
    texts: list[str] = []
    rows: list[np.ndarray] = []
    for key in keys:
        document = searchable.collection[key]
        ids.append(str(key))
        texts.append(str(getattr(document, "text", "")))
        vector = getattr(document, "vector", None)
        if vector is None:  # not stored — re-embed the text
            vector = searchable.embedder([document.text])[0]
        rows.append(np.asarray(vector, dtype=float).ravel())
    return ids, texts, np.vstack(rows)


def _vectors_from_searchable(searchable: Any) -> np.ndarray:
    """Pull every indexed vector out of a ``SearchableCorpus``'s collection."""
    return _explorable_from_searchable(searchable)[2]


def _positional_ids(n: int) -> list[str]:
    """Positional string ids ``["0", "1", …, str(n - 1)]`` for an id-less input."""
    return [str(index) for index in range(n)]


def _as_2d(data: Any) -> np.ndarray:
    """Coerce ``data`` to a 2-D float ``ndarray``, raising on any other shape."""
    array = np.asarray(data, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array of vectors, got shape {array.shape}")
    return array


def _embed_texts(texts: Iterable[str], *, embedder: Any = None) -> np.ndarray:
    """Embed an iterable of texts into an ``(n, dim)`` float array."""
    resolved = as_embedder(DEFAULT_EMBEDDER if embedder is None else embedder)
    vectors = resolved(list(texts), input_type="clustering")
    return np.asarray(vectors, dtype=float)


def _segment_text(segment: Any) -> str:
    """Extract the text of a segment — a plain string or a ``Segment`` mapping."""
    if isinstance(segment, Mapping):
        return str(segment["text"])
    return str(segment)


def _as_int(value: Any) -> int:
    """Coerce a (possibly numpy) cluster id to a plain ``int``."""
    return int(value)


# ===========================================================================
# Projection — PCA (numpy) and UMAP (lazy)
# ===========================================================================


def _choose_projection(method: ProjectionMethod, *, n_samples: int) -> str:
    """Resolve ``method`` to a concrete ``'pca'`` or ``'umap'``.

    ``'auto'`` prefers UMAP, but falls back to PCA — with a warning — when
    ``umap-learn`` is absent or the corpus is too small for UMAP to be
    meaningful.
    """
    if method == "pca":
        return "pca"
    if method == "umap":
        return "umap"
    if method != "auto":
        raise ValueError(
            f"unknown projection method {method!r} — use 'auto', 'umap' or 'pca'"
        )
    if n_samples < _MIN_SAMPLES_FOR_UMAP:
        warnings.warn(
            f"project(method='auto'): only {n_samples} samples — using PCA "
            f"(UMAP needs >= {_MIN_SAMPLES_FOR_UMAP}).",
            stacklevel=3,
        )
        return "pca"
    try:
        import umap  # noqa: F401
    except ImportError:
        warnings.warn(
            "project(method='auto'): umap-learn is not installed — using PCA. "
            "Install `ef[explore]` for the PCA -> UMAP layout.",
            stacklevel=3,
        )
        return "pca"
    return "umap"


def _import_umap() -> Any:
    """Import ``umap`` lazily, with an actionable error if it is missing."""
    try:
        import umap

        return umap
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "project(method='umap') needs umap-learn — install `ef[explore]`."
        ) from error


def _pca(vectors: np.ndarray, n_components: int) -> np.ndarray:
    """Project ``vectors`` onto their top ``n_components`` principal axes.

    A dependency-free PCA via numpy's SVD: centre, decompose, keep the leading
    components. If fewer than ``n_components`` axes exist, the result is
    zero-padded to the requested width.
    """
    centred = vectors - vectors.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centred, full_matrices=False)
    keep = min(n_components, s.shape[0])
    scores = u[:, :keep] * s[:keep]
    if keep < n_components:
        scores = np.pad(scores, ((0, 0), (0, n_components - keep)))
    return np.asarray(scores, dtype=float)


# ===========================================================================
# Clustering — k-means (numpy) and HDBSCAN (lazy)
# ===========================================================================


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row to unit L2 norm (zero rows are left unchanged)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1.0, norms)


def _kmeans(
    vectors: np.ndarray,
    n_clusters: int,
    *,
    random_state: int,
    max_iter: int = 100,
) -> np.ndarray:
    """A small, seeded Lloyd's k-means with k-means++ initialization.

    Numpy-only — keeps basic clustering available with no extra installed. For
    large corpora or production use, prefer ``method='hdbscan'`` or pass
    pre-clustered labels.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(vectors)
    k = max(1, min(n_clusters, n_samples))
    centers = _kmeanspp_init(vectors, k, rng)
    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        new_labels = _sq_dists(vectors, centers).argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = vectors[labels == j]
            if len(members):
                centers[j] = members.mean(axis=0)
    return labels.astype(int)


def _kmeanspp_init(
    vectors: np.ndarray, k: int, rng: np.random.RandomState
) -> np.ndarray:
    """Pick ``k`` initial centers with the k-means++ spreading heuristic."""
    n_samples = len(vectors)
    centers = np.empty((k, vectors.shape[1]), dtype=float)
    centers[0] = vectors[rng.randint(n_samples)]
    closest_sq = _sq_dists(vectors, centers[:1]).ravel()
    for i in range(1, k):
        total = closest_sq.sum()
        probs = closest_sq / total if total > 0 else None
        chosen = rng.choice(n_samples, p=probs)
        centers[i] = vectors[chosen]
        closest_sq = np.minimum(
            closest_sq, _sq_dists(vectors, centers[i : i + 1]).ravel()
        )
    return centers


def _sq_dists(points: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances — ``(n_points, n_centers)`` matrix."""
    diff = points[:, None, :] - centers[None, :, :]
    return np.einsum("ijk,ijk->ij", diff, diff)


def _hdbscan(vectors: np.ndarray, *, min_cluster_size: int) -> np.ndarray:
    """Density-based clustering via the optional ``hdbscan`` package."""
    try:
        import hdbscan
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "cluster(method='hdbscan') needs hdbscan — install `ef[explore]`."
        ) from error
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, min_cluster_size), metric="euclidean"
    )
    return np.asarray(clusterer.fit_predict(vectors), dtype=int)


# ===========================================================================
# Cluster labelling — lazy bridge to imbed
# ===========================================================================


def _import_cluster_labeler() -> tuple[Any, Any]:
    """Import ``pandas`` and ``imbed``'s ``ClusterLabeler`` lazily."""
    try:
        import pandas as pd
        from imbed.tools import ClusterLabeler
    except ImportError as error:  # pragma: no cover - exercised without the extra
        raise ImportError(
            "label_clusters() needs imbed (and its pandas/oa deps) — "
            "install `ef[imbed]`."
        ) from error
    return pd, ClusterLabeler
