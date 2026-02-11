import os
import json
import logging

import faiss
import numpy as np

from typing import Tuple, List


class FaceDatabase:
    """FAISS-backed face embedding database using IndexFlatIP (inner product)
    on L2-normalized vectors for cosine similarity search."""

    def __init__(
        self,
        embedding_size: int = 512,
        db_path: str = "./database/face_database",
    ) -> None:
        """Initialize the face database.

        Args:
            embedding_size: Dimension of face embeddings.
            db_path: Directory to persist the FAISS index and metadata.
        """
        self.embedding_size = embedding_size
        self.db_path = db_path
        self.index_file = os.path.join(db_path, "faiss_index.bin")
        self.meta_file = os.path.join(db_path, "metadata.json")

        os.makedirs(db_path, exist_ok=True)

        # Inner-product index -- cosine similarity when vectors are L2-normalised.
        self.index = faiss.IndexFlatIP(embedding_size)

        # Parallel list of names; metadata[i] corresponds to index row i.
        self.metadata: List[str] = []


    def add_face(self, embedding: np.ndarray, name: str) -> None:
        """Add a single face embedding to the database.

        Args:
            embedding: Face embedding vector.
            name: Identity label for this embedding.
        """
        vec = self._normalise(embedding).reshape(1, -1)
        self.index.add(vec)
        self.metadata.append(name)

    def search(self, embedding: np.ndarray, threshold: float = 0.4) -> Tuple[str, float]:
        """Find the closest identity for a single query embedding.

        Args:
            embedding: Query face embedding.
            threshold: Minimum cosine similarity to accept a match.

        Returns:
            Tuple of (name, similarity) for the best match.
        """
        if self.index.ntotal == 0:
            return "Unknown", 0.0

        vec = self._normalise(embedding).reshape(1, -1)
        similarities, indices = self.index.search(vec, 1)

        similarity = float(similarities[0][0])
        idx = int(indices[0][0])

        if similarity > threshold and idx < len(self.metadata):
            return self.metadata[idx], similarity
        return "Unknown", similarity


    def add_faces_batch(self, embeddings: List[np.ndarray], names: List[str]) -> None:
        """Add multiple face embeddings at once (vectorised).

        Args:
            embeddings: List of face embedding vectors.
            names: Corresponding identity labels.
        """
        if not embeddings:
            return
        mat = np.stack(embeddings).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # guard against zero-norm
        mat /= norms
        self.index.add(mat)
        self.metadata.extend(names)

    def batch_search(
        self,
        embeddings: List[np.ndarray],
        threshold: float = 0.4,
    ) -> List[Tuple[str, float]]:
        """Search closest identities for multiple embeddings in a single FAISS call.

        Args:
            embeddings: List of query face embeddings.
            threshold: Minimum cosine similarity to accept a match.

        Returns:
            List of (name, similarity) tuples in input order.
        """
        if not embeddings:
            return []

        if self.index.ntotal == 0:
            return [("Unknown", 0.0)] * len(embeddings)

        # Stack and normalise all queries at once.
        mat = np.stack(embeddings).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        mat /= norms

        # Single FAISS call for the entire batch.
        similarities, indices = self.index.search(mat, 1)

        results: List[Tuple[str, float]] = []
        for sim_row, idx_row in zip(similarities, indices):
            similarity = float(sim_row[0])
            idx = int(idx_row[0])
            if similarity > threshold and idx < len(self.metadata):
                results.append((self.metadata[idx], similarity))
            else:
                results.append(("Unknown", similarity))

        return results


    def save(self) -> None:
        """Persist the FAISS index and metadata to disk."""
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.meta_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logging.info(f"Face database saved with {self.index.ntotal} faces")
        except Exception as e:
            logging.error(f"Failed to save face database: {e}")
            raise

    def load(self) -> bool:
        """Load a previously-saved FAISS index and metadata from disk.

        Returns:
            True if loaded successfully, False otherwise.
        """
        if not (os.path.exists(self.index_file) and os.path.exists(self.meta_file)):
            return False
        try:
            self.index = faiss.read_index(self.index_file)
            with open(self.meta_file, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logging.info(f"Loaded face database with {self.index.ntotal} faces")
            return True
        except Exception as e:
            logging.error(f"Failed to load face database: {e}")
            return False


    @staticmethod
    def _normalise(vec: np.ndarray) -> np.ndarray:
        """L2-normalise a single embedding vector."""
        v = vec.astype(np.float32).ravel()
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        return v
