"""Script to initialize Qdrant and load travel destination data."""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.rag.vector_store import QdrantVectorStore
from app.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Initialize Qdrant collection and load data."""
    try:
        logger.info("Starting data ingestion process...")

        # Initialize vector store
        vector_store = QdrantVectorStore()

        # Create collection (recreate if exists)
        logger.info("Creating Qdrant collection...")
        vector_store.create_collection(recreate=True)

        # Load data from JSON
        data_path = Path(__file__).parent.parent / "data" / "destinations.json"
        logger.info(f"Loading data from: {data_path}")
        vector_store.load_data_from_json(str(data_path))

        # Get collection info
        info = vector_store.get_collection_info()
        logger.info(f"Collection info: {info}")

        logger.info("Data ingestion completed successfully!")

    except Exception as e:
        logger.error(f"Error during data ingestion: {e}")
        raise


if __name__ == "__main__":
    main()
