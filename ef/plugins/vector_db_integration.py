"""
Vector database integration for segmented documents.

Integrate with Pinecone, Weaviate, Chroma, and other vector databases.
"""

from typing import Any, Optional, Callable
import hashlib


class VectorDBAdapter:
    """Base adapter for vector databases."""

    def __init__(self, embedder: Optional[Callable] = None):
        """
        Initialize adapter.

        Args:
            embedder: Function that converts text to embeddings
        """
        self.embedder = embedder or self._default_embedder

    def _default_embedder(self, text: str) -> list[float]:
        """Default simple embedder (hash-based, not semantic)."""
        # Simple hash-based pseudo-embedding for demo
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        return [float(b) / 255.0 for b in hash_bytes[:16]]

    def upsert_segments(self, segments: dict[str, str], metadata: dict = None) -> None:
        """Upsert segments to database."""
        raise NotImplementedError

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for similar segments."""
        raise NotImplementedError

    def delete_segments(self, segment_ids: list[str]) -> None:
        """Delete segments by ID."""
        raise NotImplementedError


class PineconeAdapter(VectorDBAdapter):
    """Adapter for Pinecone vector database."""

    def __init__(
        self,
        api_key: str = None,
        environment: str = None,
        index_name: str = 'ef-segments',
        embedder: Optional[Callable] = None
    ):
        """Initialize Pinecone adapter."""
        super().__init__(embedder)
        self.index_name = index_name
        self.index = None

        try:
            import pinecone

            if api_key:
                pinecone.init(api_key=api_key, environment=environment)

                # Get or create index
                if index_name not in pinecone.list_indexes():
                    # Default to 16 dimensions (matches our default embedder)
                    pinecone.create_index(index_name, dimension=16)

                self.index = pinecone.Index(index_name)
                print(f"✓ Connected to Pinecone index: {index_name}")
            else:
                print("⚠ Pinecone API key not provided")

        except ImportError:
            print("⚠ Pinecone not installed. Install with: pip install pinecone-client")

    def upsert_segments(self, segments: dict[str, str], metadata: dict = None) -> None:
        """Upsert segments to Pinecone."""
        if not self.index:
            print("⚠ Pinecone not initialized")
            return

        metadata = metadata or {}
        vectors = []

        for segment_id, text in segments.items():
            embedding = self.embedder(text)
            vectors.append({
                'id': segment_id,
                'values': embedding,
                'metadata': {**metadata, 'text': text}
            })

        self.index.upsert(vectors=vectors)
        print(f"✓ Upserted {len(vectors)} segments to Pinecone")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search Pinecone for similar segments."""
        if not self.index:
            return []

        query_embedding = self.embedder(query)
        results = self.index.query(query_embedding, top_k=top_k, include_metadata=True)

        return [
            {
                'id': match['id'],
                'score': match['score'],
                'text': match['metadata'].get('text', ''),
                'metadata': match['metadata']
            }
            for match in results['matches']
        ]

    def delete_segments(self, segment_ids: list[str]) -> None:
        """Delete segments from Pinecone."""
        if not self.index:
            return

        self.index.delete(ids=segment_ids)
        print(f"✓ Deleted {len(segment_ids)} segments from Pinecone")


class WeaviateAdapter(VectorDBAdapter):
    """Adapter for Weaviate vector database."""

    def __init__(
        self,
        url: str = 'http://localhost:8080',
        api_key: str = None,
        class_name: str = 'EFSegment',
        embedder: Optional[Callable] = None
    ):
        """Initialize Weaviate adapter."""
        super().__init__(embedder)
        self.class_name = class_name
        self.client = None

        try:
            import weaviate

            if api_key:
                auth_config = weaviate.AuthApiKey(api_key=api_key)
                self.client = weaviate.Client(url=url, auth_client_secret=auth_config)
            else:
                self.client = weaviate.Client(url=url)

            # Create schema if needed
            if not self.client.schema.exists(class_name):
                schema = {
                    'class': class_name,
                    'properties': [
                        {'name': 'segment_id', 'dataType': ['string']},
                        {'name': 'text', 'dataType': ['text']},
                        {'name': 'metadata', 'dataType': ['string']}
                    ]
                }
                self.client.schema.create_class(schema)

            print(f"✓ Connected to Weaviate class: {class_name}")

        except ImportError:
            print("⚠ Weaviate not installed. Install with: pip install weaviate-client")

    def upsert_segments(self, segments: dict[str, str], metadata: dict = None) -> None:
        """Upsert segments to Weaviate."""
        if not self.client:
            print("⚠ Weaviate not initialized")
            return

        import json
        metadata = metadata or {}

        with self.client.batch as batch:
            for segment_id, text in segments.items():
                embedding = self.embedder(text)

                properties = {
                    'segment_id': segment_id,
                    'text': text,
                    'metadata': json.dumps(metadata)
                }

                batch.add_data_object(
                    properties,
                    class_name=self.class_name,
                    vector=embedding
                )

        print(f"✓ Upserted {len(segments)} segments to Weaviate")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search Weaviate for similar segments."""
        if not self.client:
            return []

        query_embedding = self.embedder(query)

        results = (
            self.client.query
            .get(self.class_name, ['segment_id', 'text', 'metadata'])
            .with_near_vector({'vector': query_embedding})
            .with_limit(top_k)
            .do()
        )

        if 'data' not in results or 'Get' not in results['data']:
            return []

        items = results['data']['Get'].get(self.class_name, [])

        return [
            {
                'id': item['segment_id'],
                'text': item['text'],
                'metadata': item.get('metadata', {})
            }
            for item in items
        ]

    def delete_segments(self, segment_ids: list[str]) -> None:
        """Delete segments from Weaviate."""
        if not self.client:
            return

        for segment_id in segment_ids:
            results = (
                self.client.query
                .get(self.class_name)
                .with_where({
                    'path': ['segment_id'],
                    'operator': 'Equal',
                    'valueString': segment_id
                })
                .do()
            )

            # Delete found objects
            if 'data' in results and 'Get' in results['data']:
                items = results['data']['Get'].get(self.class_name, [])
                for item in items:
                    if '_additional' in item and 'id' in item['_additional']:
                        self.client.data_object.delete(item['_additional']['id'])

        print(f"✓ Deleted {len(segment_ids)} segments from Weaviate")


class ChromaAdapter(VectorDBAdapter):
    """Adapter for Chroma vector database."""

    def __init__(
        self,
        path: str = './chroma_db',
        collection_name: str = 'ef_segments',
        embedder: Optional[Callable] = None
    ):
        """Initialize Chroma adapter."""
        super().__init__(embedder)
        self.collection_name = collection_name
        self.collection = None

        try:
            import chromadb

            self.client = chromadb.PersistentClient(path=path)
            self.collection = self.client.get_or_create_collection(name=collection_name)

            print(f"✓ Connected to Chroma collection: {collection_name}")

        except ImportError:
            print("⚠ Chroma not installed. Install with: pip install chromadb")

    def upsert_segments(self, segments: dict[str, str], metadata: dict = None) -> None:
        """Upsert segments to Chroma."""
        if not self.collection:
            print("⚠ Chroma not initialized")
            return

        metadata = metadata or {}

        ids = list(segments.keys())
        documents = list(segments.values())
        embeddings = [self.embedder(text) for text in documents]
        metadatas = [{**metadata, 'segment_id': seg_id} for seg_id in ids]

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"✓ Upserted {len(segments)} segments to Chroma")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search Chroma for similar segments."""
        if not self.collection:
            return []

        query_embedding = self.embedder(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        if not results['ids'] or not results['ids'][0]:
            return []

        return [
            {
                'id': results['ids'][0][i],
                'text': results['documents'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None,
                'metadata': results['metadatas'][0][i] if 'metadatas' in results else {}
            }
            for i in range(len(results['ids'][0]))
        ]

    def delete_segments(self, segment_ids: list[str]) -> None:
        """Delete segments from Chroma."""
        if not self.collection:
            return

        self.collection.delete(ids=segment_ids)
        print(f"✓ Deleted {len(segment_ids)} segments from Chroma")


class QdrantAdapter(VectorDBAdapter):
    """Adapter for Qdrant vector database."""

    def __init__(
        self,
        url: str = 'localhost',
        port: int = 6333,
        api_key: str = None,
        collection_name: str = 'ef_segments',
        embedder: Optional[Callable] = None
    ):
        """Initialize Qdrant adapter."""
        super().__init__(embedder)
        self.collection_name = collection_name
        self.client = None

        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            if api_key:
                self.client = QdrantClient(url=url, api_key=api_key)
            else:
                self.client = QdrantClient(host=url, port=port)

            # Create collection if needed
            collections = self.client.get_collections().collections
            if collection_name not in [c.name for c in collections]:
                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(size=16, distance=Distance.COSINE)
                )

            print(f"✓ Connected to Qdrant collection: {collection_name}")

        except ImportError:
            print("⚠ Qdrant not installed. Install with: pip install qdrant-client")

    def upsert_segments(self, segments: dict[str, str], metadata: dict = None) -> None:
        """Upsert segments to Qdrant."""
        if not self.client:
            print("⚠ Qdrant not initialized")
            return

        from qdrant_client.models import PointStruct

        metadata = metadata or {}
        points = []

        for i, (segment_id, text) in enumerate(segments.items()):
            embedding = self.embedder(text)

            points.append(PointStruct(
                id=hash(segment_id) % (2**63),  # Convert to int
                vector=embedding,
                payload={**metadata, 'segment_id': segment_id, 'text': text}
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

        print(f"✓ Upserted {len(points)} segments to Qdrant")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search Qdrant for similar segments."""
        if not self.client:
            return []

        query_embedding = self.embedder(query)

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )

        return [
            {
                'id': result.payload.get('segment_id', str(result.id)),
                'score': result.score,
                'text': result.payload.get('text', ''),
                'metadata': result.payload
            }
            for result in results
        ]

    def delete_segments(self, segment_ids: list[str]) -> None:
        """Delete segments from Qdrant."""
        if not self.client:
            return

        from qdrant_client.models import Filter, FieldCondition, MatchValue

        for segment_id in segment_ids:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key='segment_id',
                            match=MatchValue(value=segment_id)
                        )
                    ]
                )
            )

        print(f"✓ Deleted {len(segment_ids)} segments from Qdrant")


def segment_and_index(
    project,
    text: str,
    segmenter: str,
    vector_db: VectorDBAdapter,
    metadata: dict = None
) -> dict[str, str]:
    """
    Segment text and index in vector database.

    Args:
        project: EF project
        text: Text to segment
        segmenter: Name of segmenter to use
        vector_db: Vector database adapter
        metadata: Optional metadata

    Returns:
        Segments dictionary
    """
    if segmenter not in project.segmenters:
        raise ValueError(f"Segmenter '{segmenter}' not found")

    seg_func = project.segmenters[segmenter]
    segments = seg_func(text)

    vector_db.upsert_segments(segments, metadata)

    return segments


def semantic_search_segments(
    query: str,
    vector_db: VectorDBAdapter,
    top_k: int = 5
) -> list[dict]:
    """
    Semantic search for segments.

    Args:
        query: Search query
        vector_db: Vector database adapter
        top_k: Number of results

    Returns:
        List of matching segments
    """
    return vector_db.search(query, top_k)


def register_vector_db_helpers(project) -> int:
    """Register vector database helper segmenters."""
    count = 0

    # Note: These are helper functions, not actual segmenters
    # They're for documentation/reference

    print(f"✓ Vector DB integration ready (use adapters directly)")
    return count


# Create default adapters
def create_adapter(db_type: str, **kwargs) -> VectorDBAdapter:
    """
    Create vector database adapter.

    Args:
        db_type: Type of database ('pinecone', 'weaviate', 'chroma', 'qdrant')
        **kwargs: Database-specific configuration

    Returns:
        Vector database adapter
    """
    adapters = {
        'pinecone': PineconeAdapter,
        'weaviate': WeaviateAdapter,
        'chroma': ChromaAdapter,
        'qdrant': QdrantAdapter
    }

    if db_type not in adapters:
        raise ValueError(f"Unknown database type: {db_type}")

    return adapters[db_type](**kwargs)
