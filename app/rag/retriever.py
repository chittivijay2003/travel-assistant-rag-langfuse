"""Hybrid search retriever combining semantic and keyword search."""

import logging
from typing import List, Dict, Any
from qdrant_client.models import NamedVector, Prefetch

try:
    from langfuse import observe
except ImportError:
    # Fallback dummy decorator
    def observe(*args, **kwargs):
        def decorator(func):
            return func

        if args and callable(args[0]):
            return args[0]
        return decorator


from app.rag.vector_store import QdrantVectorStore
from app.config import settings

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Implements hybrid search combining semantic (dense) and keyword (sparse) search."""

    def __init__(self, vector_store: QdrantVectorStore):
        """
        Initialize hybrid retriever.

        Args:
            vector_store: QdrantVectorStore instance
        """
        self.vector_store = vector_store
        self.client = vector_store.client
        self.collection_name = vector_store.collection_name
        self.embedding_model = vector_store.embedding_model
        logger.info("Initialized HybridRetriever")

    def _create_query_sparse_vector(self, query: str) -> Dict[str, Any]:
        """
        Create sparse vector from query for keyword matching.
        Boosts country names for better country-specific retrieval.

        Args:
            query: Search query

        Returns:
            Sparse vector dict
        """
        # List of country names to boost
        country_keywords = [
            "japan",
            "thailand",
            "singapore",
            "dubai",
            "uae",
            "maldives",
            "usa",
            "america",
            "uk",
            "britain",
            "australia",
            "france",
            "paris",
            "switzerland",
            "bali",
            "indonesia",
            "malaysia",
            "sri lanka",
            "nepal",
            "bhutan",
        ]

        words = query.lower().split()
        word_freq = {}

        for word in words:
            if len(word) > 2:
                word_hash = hash(word) % 100000
                # Boost country names 3x for better relevance
                boost = 3.0 if word in country_keywords else 1.0
                word_freq[word_hash] = word_freq.get(word_hash, 0) + boost

        return {"indices": list(word_freq.keys()), "values": list(word_freq.values())}

    @observe(name="hybrid_search", as_type="retriever")
    def hybrid_search(
        self,
        query: str,
        top_k: int = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        min_score: float = 0.45,
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword search.

        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic search (0-1)
            keyword_weight: Weight for keyword search (0-1)
            min_score: Minimum relevance score threshold (filters out low-relevance results)

        Returns:
            List of retrieved documents with scores
        """
        try:
            if top_k is None:
                top_k = settings.top_k_results

            logger.info(f"Performing hybrid search for: '{query}' (top_k={top_k})")

            # Generate dense embedding for semantic search
            dense_vector = self.embedding_model.encode(query).tolist()

            # Generate sparse vector for keyword search
            sparse_vector = self._create_query_sparse_vector(query)

            # Perform hybrid search using query_points
            try:
                from qdrant_client.models import SparseVector

                results = self.client.query_points(
                    collection_name=self.collection_name,
                    prefetch=[
                        Prefetch(query=dense_vector, using="dense", limit=top_k * 2),
                        Prefetch(
                            query=SparseVector(**sparse_vector),
                            using="sparse",
                            limit=top_k * 2,
                        ),
                    ],
                    query="rrf",  # Reciprocal Rank Fusion
                    limit=top_k,
                )

                # Format results and filter by minimum score
                documents = []
                for point in results.points:
                    # Only include results above minimum score threshold
                    if point.score >= min_score:
                        doc = {
                            "content": point.payload,
                            "score": point.score,
                            "id": point.id,
                        }
                        documents.append(doc)

                logger.info(
                    f"Retrieved {len(documents)} documents (filtered by min_score={min_score})"
                )
                return documents
            except Exception as fallback_error:
                # Fallback to semantic only
                logger.warning(
                    f"Hybrid search failed: {fallback_error}, falling back to semantic search"
                )
                return self._semantic_search_fallback(query, top_k, min_score)

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            # Fallback to semantic-only search
            return self._semantic_search_fallback(query, top_k, min_score)

    def _semantic_search_fallback(
        self, query: str, top_k: int, min_score: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Fallback to semantic-only search if hybrid fails.

        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum score threshold

        Returns:
            List of documents
        """
        try:
            logger.warning("Falling back to semantic-only search")
            dense_vector = self.embedding_model.encode(query).tolist()

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using="dense",
                limit=top_k,
            )

            documents = []
            for point in results.points:
                # Apply minimum score filter
                if point.score >= min_score:
                    doc = {
                        "content": point.payload,
                        "score": point.score,
                        "id": point.id,
                    }
                    documents.append(doc)

            logger.info(
                f"Semantic fallback retrieved {len(documents)} documents (filtered by min_score={min_score})"
            )
            return documents

        except Exception as e:
            logger.error(f"Error in semantic search fallback: {e}")
            return []

    def search_with_filter(
        self, query: str, filter_conditions: Dict[str, Any], top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search with additional filters.

        Args:
            query: Search query
            filter_conditions: Qdrant filter conditions
            top_k: Number of results

        Returns:
            List of filtered documents
        """
        try:
            if top_k is None:
                top_k = settings.top_k_results

            dense_vector = self.embedding_model.encode(query).tolist()

            results = self.client.query_points(
                collection_name=self.collection_name,
                query=dense_vector,
                using="dense",
                query_filter=filter_conditions,
                limit=top_k,
            )

            documents = []
            for point in results.points:
                doc = {"content": point.payload, "score": point.score, "id": point.id}
                documents.append(doc)

            logger.info(f"Retrieved {len(documents)} filtered documents")
            return documents

        except Exception as e:
            logger.error(f"Error in filtered search: {e}")
            return []
