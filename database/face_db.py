import os
import faiss
import numpy as np
import json
import logging
from typing import Tuple


class FaceDatabase:
    def __init__(self, embedding_size: int = 512, db_path: str = "./database/face_database") -> None:
        """
        Initialize the face database.

        Args:
            embedding_size: Dimension of face embeddings
            db_path: Directory to store database files
        """
        self.embedding_size = embedding_size
        self.db_path = db_path
        self.index_file = os.path.join(db_path, "faiss_index.bin")
        self.meta_file = os.path.join(db_path, "metadata.json")

        # Create directory if it doesn't exist
        os.makedirs(db_path, exist_ok=True)

        # Initialize FAISS index for L2 distance (can be converted to similarity)
        self.index = faiss.IndexFlatIP(embedding_size)  # Inner product for cosine similarity

        # Metadata to store names corresponding to indices
        self.metadata = []

    def add_face(self, embedding: np.ndarray, name: str) -> None:
        """
        Add a face embedding to the database.

        Args:
            embedding: Face embedding vector
            name: Name of the person
        """
        # Normalize for cosine similarity
        normalized_embedding = embedding / np.linalg.norm(embedding)
        self.index.add(np.array([normalized_embedding], dtype=np.float32))
        self.metadata.append(name)

    def search(self, embedding: np.ndarray, threshold: float = 0.4) -> Tuple[str, float]:
        """
        Search for the closest face in the database.

        Args:
            embedding: Query face embedding
            threshold: Similarity threshold

        Returns:
            Tuple containing the name and similarity score
        """
        if self.index.ntotal == 0:
            return "Unknown", 0.0

        # Normalize
        normalized_embedding = embedding / np.linalg.norm(embedding)

        # Search
        similarities, indices = self.index.search(np.array([normalized_embedding], dtype=np.float32), 1)

        # Get the best match
        best_similarity = similarities[0][0]
        best_idx = indices[0][0]

        if best_similarity > threshold and best_idx < len(self.metadata):
            return self.metadata[best_idx], best_similarity
        else:
            return "Unknown", best_similarity

    def save(self) -> None:
        """Save the database to disk"""
        faiss.write_index(self.index, self.index_file)
        with open(self.meta_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        logging.info(f"Face database saved with {self.index.ntotal} faces")

    def load(self) -> bool:
        """
        Load the database from disk.

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            self.index = faiss.read_index(self.index_file)
            # --- Load metadata from JSON ---
            with open(self.meta_file, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logging.info(f"Loaded face database with {self.index.ntotal} faces")
            return True
        return False
