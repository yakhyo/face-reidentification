# Face Database API Reference

The Face Database module provides efficient storage and retrieval of face embeddings using FAISS (Facebook AI Similarity Search).

## FaceDatabase Class

### Initialization
```python
FaceDatabase(embedding_size: int = 512, db_path: str = "./database/face_database") -> None
```

Creates a new face database instance.

**Parameters:**
- `embedding_size` (int): Dimension of face embeddings (default: 512)
- `db_path` (str): Directory to store database files (default: "./database/face_database")

**Files Created:**
- `{db_path}/faiss_index.bin`: FAISS index file
- `{db_path}/metadata.json`: Metadata for face identities

### Methods

#### add_face
```python
add_face(embedding: np.ndarray, name: str) -> None
```

Add a face embedding to the database.

**Parameters:**
- `embedding` (np.ndarray): Face embedding vector
- `name` (str): Name of the person

**Note:** The embedding is automatically normalized before adding to the database.

#### search
```python
search(embedding: np.ndarray, threshold: float = 0.4) -> Tuple[str, float]
```

Search for the closest face in the database.

**Parameters:**
- `embedding` (np.ndarray): Query face embedding
- `threshold` (float): Similarity threshold (default: 0.4)

**Returns:**
- Tuple[str, float]: (matched_name, similarity_score)

#### save
```python
save() -> bool
```

Save the database to disk.

**Returns:**
- bool: True if save was successful

#### load
```python
load() -> bool
```

Load the database from disk.

**Returns:**
- bool: True if load was successful

## Usage Example

```python
from database.face_db import FaceDatabase
import numpy as np

# Initialize database
db = FaceDatabase(embedding_size=512, db_path="./database/face_database")

# Add a face
embedding = np.random.rand(512)  # Replace with actual face embedding
db.add_face(embedding, "John Doe")

# Search for a face
query_embedding = np.random.rand(512)  # Replace with actual query embedding
name, similarity = db.search(query_embedding, threshold=0.4)
if similarity > 0.4:
    print(f"Match found: {name} with similarity {similarity:.2f}")

# Save database
db.save()

# Load database
new_db = FaceDatabase()
if new_db.load():
    print("Database loaded successfully")
```

## Implementation Details

1. **Similarity Measure**
   - Uses cosine similarity via inner product on normalized vectors
   - FAISS IndexFlatIP for efficient similarity search
   - Embeddings are automatically normalized before storage

2. **Data Persistence**
   - FAISS index saved to binary file
   - Metadata stored in JSON format
   - Automatic directory creation if not exists

3. **Error Handling**
   - Safe file operations with proper error handling
   - Logging of operations for debugging
   - Type hints for better code safety
