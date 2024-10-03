from sklearn.metrics.pairwise import cosine_similarity

class Database:
    def __init__(self, *args, **kwargs):
        return NotImplementedError

    def _create_table(self):
        return NotImplementedError

    def insert_vector(self, *args, **kwargs):
        return NotImplementedError

    def get_vectors(self):
        return NotImplementedError

    def delete_vector(self, *args, **kwargs):
        return NotImplementedError

    def find_most_similar(self, *args, **kwargs):
        return NotImplementedError
    
    def close(self):
        return NotImplementedError

    def __enter__(self):
        return NotImplementedError

    def __exit__(self, *args):
        return NotImplementedError