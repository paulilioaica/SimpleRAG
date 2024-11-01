from database import Database
import sqlite3
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

class VectorDatabase(Database):
    def __init__(self, db_name='vectors.db'):
        """Initialize the connection to the database and create the table if it doesn't exist."""
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        """Create the table to store vectors and their associated text, with an auto-incrementing primary key."""
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS vectors (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            text TEXT NOT NULL,
            embedding TEXT NOT NULL
        )''')
        self.conn.commit()

    def insert_vector(self, text, vector):
        """Insert a vector and its associated text into the database."""
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        
        vector_json = json.dumps(vector.tolist())
        self.cursor.execute('INSERT INTO vectors (text, embedding) VALUES (?, ?)', (text, vector_json))
        self.conn.commit()

    def get_vectors(self):
        """Retrieve all vectors and their associated text from the database."""
        self.cursor.execute('SELECT id, text, embedding FROM vectors')
        rows = self.cursor.fetchall()
        return [(row[0], row[1], np.array(json.loads(row[2]))) for row in rows]

    def delete_vector(self, vector_id):
        """Delete a vector by its ID."""
        self.cursor.execute('DELETE FROM vectors WHERE id = ?', (vector_id,))
        self.conn.commit()

    def find_most_similar(self, query_vector, top_n=1):
        """Find the most similar vector to the query vector using cosine similarity."""
        if not isinstance(query_vector, np.ndarray):
            query_vector = np.array(query_vector)
        
        vectors = self.get_vectors()
        if not vectors:
            return None
        
        similarities = [(id, text, cosine_similarity([query_vector], [vec])[0][0]) for id, text, vec in vectors]
        return sorted(similarities, key=lambda x: x[2], reverse=True)[:top_n]
    
    def close(self):
        """Close the connection to the database."""
        self.conn.close()

    def __enter__(self):
        """Enable usage of the class as a context manager."""
        return self

    def __exit__(self):
        """Ensure the database connection is closed when used in a context manager."""
        self.close()
