# backend/retriever.py
import os
import sqlite3
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Dict

class Retriever:
    def __init__(self, data_dir: str = "../data", model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.data_dir = os.path.abspath(data_dir)
        os.makedirs(self.data_dir, exist_ok=True)
        self.db_path = os.path.join(self.data_dir, "metadata.db")
        self.index_path = os.path.join(self.data_dir, "faiss.index")
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        # initialize or load faiss index
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            # ensure correct dim
            if self.index.d != self.dim:
                raise RuntimeError("FAISS index dimension mismatch.")
        else:
            self.index = faiss.IndexFlatIP(self.dim)  # cosine via normalized vectors
        # ensure sqlite tables
        self._ensure_tables()

    def _ensure_tables(self):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT,
                offset INTEGER,
                text TEXT
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS vector_map (
                vector_pos INTEGER PRIMARY KEY,
                chunk_id INTEGER,
                FOREIGN KEY(chunk_id) REFERENCES chunks(id)
            )
        """)
        conn.commit()
        conn.close()

    def embed(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(emb)
        return emb

    def persist_index(self):
        faiss.write_index(self.index, self.index_path)

    def add_chunks(self, source: str, chunks: List[str]) -> List[int]:
        """
        Add chunks to DB and FAISS. Returns list of chunk ids.
        """
        if not chunks:
            return []
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        # insert chunks, collect their ids
        chunk_ids = []
        for i, chunk in enumerate(chunks):
            c.execute("INSERT INTO chunks (source, offset, text) VALUES (?, ?, ?)", (source, i, chunk))
            chunk_ids.append(c.lastrowid)
        conn.commit()
        # compute embeddings and add to index
        embeddings = self.embed(chunks)
        prev_n = self.index.ntotal
        self.index.add(embeddings)
        # map new vector positions to chunk ids
        conn2 = sqlite3.connect(self.db_path)
        c2 = conn2.cursor()
        for idx, cid in enumerate(chunk_ids):
            vector_pos = prev_n + idx
            c2.execute("INSERT INTO vector_map (vector_pos, chunk_id) VALUES (?, ?)", (vector_pos, cid))
        conn2.commit()
        conn2.close()
        conn.close()
        self.persist_index()
        return chunk_ids

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        if self.index.ntotal == 0:
            return []
        q_emb = self.embed([query])
        D, I = self.index.search(q_emb, top_k)
        ids = I[0].tolist()
        scores = D[0].tolist()
        results = []
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        for pos, score in zip(ids, scores):
            if pos < 0:
                continue
            c.execute("SELECT chunk_id FROM vector_map WHERE vector_pos=?", (int(pos),))
            row = c.fetchone()
            if not row:
                continue
            chunk_id = row[0]
            c.execute("SELECT source, offset, text FROM chunks WHERE id=?", (chunk_id,))
            r = c.fetchone()
            if not r:
                continue
            results.append({"id": chunk_id, "source": r[0], "offset": r[1], "text": r[2], "score": float(score)})
        conn.close()
        return results
