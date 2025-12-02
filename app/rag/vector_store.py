"""Qdrant vector store management."""

import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    SparseVectorParams,
    SparseIndexParams,
)
from sentence_transformers import SentenceTransformer
import json
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)


class QdrantVectorStore:
    """Manages Qdrant vector database operations."""

    def __init__(self):
        """Initialize Qdrant client and embedding model."""
        # Use in-memory Qdrant if remote server is not available
        use_memory = False
        try:
            self.client = QdrantClient(url=settings.qdrant_url)
            # Test connection
            self.client.get_collections()
            logger.info(f"Connected to Qdrant at: {settings.qdrant_url}")
        except Exception as e:
            logger.warning(f"Failed to connect to Qdrant server: {e}")
            logger.info("Using in-memory Qdrant instance")
            self.client = QdrantClient(":memory:")
            use_memory = True

        self.collection_name = settings.qdrant_collection_name
        self.embedding_model = SentenceTransformer(settings.embedding_model)
        self.vector_size = self.embedding_model.get_sentence_embedding_dimension()
        logger.info(
            f"Embedding model: {settings.embedding_model}, dimension: {self.vector_size}"
        )
        if use_memory:
            logger.info("Qdrant running in-memory mode")

    def create_collection(self, recreate: bool = False) -> None:
        """
        Create Qdrant collection with dense and sparse vectors for hybrid search.

        Args:
            recreate: If True, delete existing collection and create new one
        """
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            exists = any(col.name == self.collection_name for col in collections)

            if exists:
                if recreate:
                    logger.info(f"Deleting existing collection: {self.collection_name}")
                    self.client.delete_collection(self.collection_name)
                else:
                    logger.info(f"Collection already exists: {self.collection_name}")
                    return

            # Create collection with dense vectors (semantic search)
            logger.info(f"Creating collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=self.vector_size, distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False))
                },
            )
            logger.info(f"Collection created successfully: {self.collection_name}")

        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    def _create_sparse_vector(self, text: str) -> Dict[str, Any]:
        """
        Create sparse vector for keyword search using simple term frequency.

        Args:
            text: Input text

        Returns:
            Sparse vector with indices and values
        """
        # Simple keyword extraction (can be enhanced with BM25)
        words = text.lower().split()
        word_freq = {}

        for word in words:
            if len(word) > 2:  # Filter short words
                word_hash = hash(word) % 100000  # Simple hashing
                word_freq[word_hash] = word_freq.get(word_hash, 0) + 1

        indices = list(word_freq.keys())
        values = list(word_freq.values())

        return {"indices": indices, "values": values}

    def insert_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Insert documents into Qdrant with dense and sparse vectors.

        Args:
            documents: List of document dictionaries
        """
        try:
            points = []

            for doc in documents:
                # Create searchable text from document
                searchable_text = self._create_searchable_text(doc)

                # Generate dense embedding
                dense_vector = self.embedding_model.encode(searchable_text).tolist()

                # Generate sparse vector for keyword search
                sparse_vector = self._create_sparse_vector(searchable_text)

                # Create point
                point = PointStruct(
                    id=hash(doc.get("id", doc.get("title", str(len(points))))),
                    vector={"dense": dense_vector, "sparse": sparse_vector},
                    payload=doc,
                )
                points.append(point)

            # Upsert points
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"Inserted {len(points)} documents into {self.collection_name}")

        except Exception as e:
            logger.error(f"Error inserting documents: {e}")
            raise

    def _create_searchable_text(self, doc: Dict[str, Any]) -> str:
        """
        Create searchable text from document fields.

        Args:
            doc: Document dictionary

        Returns:
            Combined searchable text
        """
        parts = []

        # Add basic fields
        if "title" in doc:
            parts.append(doc["title"])
        if "country" in doc:
            parts.append(doc["country"])
        if "content" in doc:
            parts.append(doc["content"])

        # Add visa requirements text
        if "visa_requirements" in doc:
            visa_info = doc["visa_requirements"]
            if isinstance(visa_info, dict) and "indian_citizens" in visa_info:
                indian_visa = visa_info["indian_citizens"]
                if isinstance(indian_visa, dict):
                    parts.append(json.dumps(indian_visa))

        # Add attractions
        if "attractions" in doc and isinstance(doc["attractions"], list):
            parts.extend(doc["attractions"])

        # Add other fields
        for key in ["best_time_to_visit", "climate", "currency", "language"]:
            if key in doc:
                parts.append(str(doc[key]))

        return " ".join(parts)

    def load_data_from_json(self, json_path: str) -> None:
        """
        Load data from JSON file and insert into Qdrant.

        Args:
            json_path: Path to JSON file
        """
        try:
            logger.info(f"Loading data from: {json_path}")
            with open(json_path, "r", encoding="utf-8") as f:
                documents = json.load(f)

            logger.info(f"Loaded {len(documents)} documents")
            self.insert_documents(documents)

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": collection_info.points_count,
                "status": collection_info.status,
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {}
