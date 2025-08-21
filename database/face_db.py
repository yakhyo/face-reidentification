import os
import faiss
import numpy as np
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List, Optional
from queue import Queue


class FaceDatabase:
    def __init__(self, embedding_size: int = 512, db_path: str = "./database/face_database", max_workers: int = 4) -> None:
        """
        Initialize the face database with thread support.

        Args:
            embedding_size: Dimension of face embeddings
            db_path: Directory to store database files
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.embedding_size = embedding_size
        self.db_path = db_path
        self.index_file = os.path.join(db_path, "faiss_index.bin")
        self.meta_file = os.path.join(db_path, "metadata.json")
        self.max_workers = max_workers

        os.makedirs(db_path, exist_ok=True)

        # Use inner product for cosine similarity search
        self.index = faiss.IndexFlatIP(embedding_size)

        # Thread-safe queue for batch processing
        self.search_queue = Queue()

        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Lock for thread-safe operations
        self.lock = threading.Lock()

        # Stores associated names for each embedding
        self.metadata = []

    def add_face(self, embedding: np.ndarray, name: str) -> None:
        """
        Add a face embedding to the database thread-safely.

        Args:
            embedding: Face embedding vector
            name: Name of the person
        """
        normalized_embedding = embedding / np.linalg.norm(embedding)
        with self.lock:
            self.index.add(np.array([normalized_embedding], dtype=np.float32))
            self.metadata.append(name)

    def search(self, embedding: np.ndarray, threshold: float = 0.4) -> Tuple[str, float]:
        """
        Search for the closest face in the database.

        Args:
            embedding: Query face embedding
            threshold: Similarity threshold

        Returns:
            Tuple containing the name and similarity score of the best match
        """
        if self.index.ntotal == 0:
            return "Unknown", 0.0

        normalized_embedding = embedding / np.linalg.norm(embedding)
        with self.lock:
            similarities, indices = self.index.search(np.array([normalized_embedding], dtype=np.float32), 1)

        similarity = float(similarities[0][0])
        idx = indices[0][0]

        if similarity > threshold and idx < len(self.metadata):
            return self.metadata[idx], similarity
        return "Unknown", similarity

    def batch_search(self, embeddings: List[np.ndarray], threshold: float = 0.4) -> List[Tuple[str, float]]:
        """
        Perform batch search for multiple face embeddings in parallel.

        Args:
            embeddings: List of face embeddings to search for
            threshold: Similarity threshold

        Returns:
            List of tuples containing names and similarity scores
        """
        def search_worker(emb):
            return self.search(emb, threshold)

        # Submit all searches to thread pool
        futures = [self.executor.submit(search_worker, emb) for emb in embeddings]

        # Gather results in order
        results = []
        for future in as_completed(futures):
            results.append(future.result())

        return results

    def add_faces_batch(self, embeddings: List[np.ndarray], names: List[str]) -> None:
        """
        Add multiple faces to the database in parallel.

        Args:
            embeddings: List of face embeddings
            names: List of corresponding names
        """
        normalized_embeddings = [emb / np.linalg.norm(emb) for emb in embeddings]

        with self.lock:
            self.index.add(np.array(normalized_embeddings, dtype=np.float32))
            self.metadata.extend(names)

    def save(self) -> None:
        """
        Save the FAISS index and metadata to disk thread-safely.
        """
        with self.lock:
            faiss.write_index(self.index, self.index_file)
            with open(self.meta_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
            logging.info(f"Face database saved with {self.index.ntotal} faces")

    def load(self) -> bool:
        """
        Load the FAISS index and metadata from disk thread-safely.

        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if os.path.exists(self.index_file) and os.path.exists(self.meta_file):
            with self.lock:
                self.index = faiss.read_index(self.index_file)
                with open(self.meta_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
                logging.info(f"Loaded face database with {self.index.ntotal} faces")
            return True
        return False

    def __del__(self):
        """
        Clean up thread pool on deletion.
        """
        self.executor.shutdown(wait=True)
