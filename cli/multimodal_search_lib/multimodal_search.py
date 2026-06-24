from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import json
from pathlib import Path
from cli.constants import CACHE_DIR


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)


class MultiModelSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.docmap = {}
        self.text_embeddings = []
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = CACHE_DIR / "clip_embeddings.npy"
        self.docmap_path = CACHE_DIR / "clip_docmap.pkl"

    def embed_image(self, imageUrl):
        image = Image.open(imageUrl)
        return self.model.encode([image])

    def build(self, movie_file="./data/movies.json"):
        with open(movie_file, "r") as f:
            data = json.load(f)
        movies = data["movies"]
        texts = []
        for doc_id, movie in enumerate(movies):
            self.docmap[doc_id] = movie
            texts.append(f"{movie['title']} {movie['description']}")
        self.text_embeddings = self.model.encode(texts, show_progress_bar=True)
        np.save(self.embeddings_path, self.text_embeddings)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)

    def load_or_build(self, movie_file="./data/movies.json"):
        if self.embeddings_path.exists() and self.docmap_path.exists():
            self.text_embeddings = np.load(self.embeddings_path)
            with open(self.docmap_path, "rb") as f:
                self.docmap = pickle.load(f)
        else:
            self.build(movie_file)

    def search_with_image(self, imageUrl, limit):
        image_embedding = self.embed_image(imageUrl)[0]
        scores = []
        for id, embedding in enumerate(self.text_embeddings):
            similarity = cosine_similarity(image_embedding, embedding)
            scores.append(
                {
                    "score": similarity,
                    "title": self.docmap[id]["title"],
                    "description": self.docmap[id]["description"],
                }
            )
        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:limit]
