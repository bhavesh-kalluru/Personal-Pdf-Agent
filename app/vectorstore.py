from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings

from utils import ensure_dir, load_env_path


class ChromaStore:
    """Local persistent Chroma store."""

    def __init__(self, collection_name: str = "personal_faq") -> None:
        base = load_env_path()
        ensure_dir(base)
        self.persist_dir = os.path.join(base, "chroma")
        ensure_dir(self.persist_dir)

        self.client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def reset(self) -> None:
        name = self.collection.name
        self.client.delete_collection(name=name)
        self.collection = self.client.get_or_create_collection(name=name)

    def add_chunks(
        self,
        ids: List[str],
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> None:
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[str], List[Dict[str, Any]], List[float], List[str]]:
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["metadatas", "documents", "distances"],
        )

        ids = (res.get("ids") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]

        scores = [-(d if d is not None else 0.0) for d in distances]
        return ids, metas, scores, docs

    def count(self) -> int:
        return self.collection.count()
