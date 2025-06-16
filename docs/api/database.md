# Face Database API Reference

The Face Database module provides efficient storage and retrieval of face embeddings using FAISS (Facebook AI Similarity Search).

## FaceDatabase Class

### Initialization
```python
FaceDatabase(embedding_size: int = 512, db_path: str = "./database/face_database")
```

Creates a new face database instance.

**Parameters:**
- `embedding_size` (int): Dimension of face embeddings (default: 512)
- `db_path` (str): Directory to store database files (default: "./database/face_database")

### Methods

#### add_face
```python
add_face(embedding: np.ndarray, name: str) -> None
```

Add a face embedding to the database.

**Parameters:**
- `embedding` (np.ndarray): Face embedding vector
- `name` (str): Name of the person

#### search
```python
search(embedding: np.ndarray, threshold: float = 0.4) -> Tuple[str, float]
```

Search for the closest face in the database.

**Parameters:**
- `embedding` (np.ndarray): Query face embedding
- `threshold` (float): Similarity threshold (default: 0.4)

**Returns:**
- Tuple containing the matched person's name and similarity score

## Usage Example

```python
from database.face_db import FaceDatabase

# Initialize database
db = FaceDatabase(embedding_size=512)

# Add a face
embedding = model.get_embedding(face_image)
db.add_face(embedding, "John Doe")

# Search for a face
match_name, similarity = db.search(query_embedding)
if similarity > 0.4:
    print(f"Match found: {match_name}")
```
