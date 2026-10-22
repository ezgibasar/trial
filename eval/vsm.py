import os
import logging
import numpy as np


class VSM:
    def __init__(self, vecs_path, normalize=True):
        self.labels = []
        self.vectors = np.array([], dtype=np.float32)
        self.indices = {}
        self.ndims = 0
        self._load(vecs_path)
        if normalize and len(self.vectors) > 0:
            self.normalize()

    def _load(self, path):
        if not os.path.exists(path):
            logging.warning(f"Vector file not found: {path}")
            return
        vectors = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                elems = line.split()
                if len(elems) < 2:
                    continue
                self.labels.append(elems[0])
                vectors.append(
                    np.array(list(map(float, elems[1:])), dtype=np.float32)
                )
        if vectors:
            self.vectors = np.vstack(vectors)
            self.indices = {l: i for i, l in enumerate(self.labels)}
            self.ndims = self.vectors.shape[1]

    def normalize(self):
        if len(self.vectors) > 0:
            self.vectors = (
                self.vectors.T / np.linalg.norm(self.vectors, axis=1)
            ).T

    def most_similar_vec(self, vec, topn=None):
        if len(self.vectors) == 0:
            return [("NULL", -1)]
        sims = np.dot(self.vectors, vec)
        sims_list = sims.tolist()
        out = []
        for idx in sims.argsort().tolist()[::-1][:topn]:
            out.append((self.labels[idx], sims_list[idx]))
        return out