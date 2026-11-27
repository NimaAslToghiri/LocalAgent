"""Conversation storage kept separate from inference and routing."""

import json
import logging
import time
from pathlib import Path
from uuid import uuid4

from .state import Turn


logger = logging.getLogger(__name__)


class ConversationMemory:
    """Store complete turns in Chroma and retrieve them within one conversation."""

    def __init__(self, path: Path, recent_limit: int = 4, recall_limit: int = 2):
        """Open the conversation collection without loading the embedding model yet."""
        import chromadb

        self.collection = chromadb.PersistentClient(path=str(path)).get_or_create_collection(
            name="conversation_turns", configuration={"hnsw": {"space": "cosine"}},
            embedding_function=None,
        )
        self.recent_limit = recent_limit
        self.recall_limit = recall_limit
        self._encoder = None

    def _embed(self, text: str) -> list[float]:
        """Generate an embedding on the CPU to leave GPU memory for Qwen."""
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        return self._encoder.encode(text, normalize_embeddings=True).tolist()

    def _recent(self, conversation_id: str) -> tuple[list[Turn], int | None]:
        """Load only the newest turn documents after sorting scoped timestamps."""
        records = self.collection.get(where={"conversation_id": conversation_id}, include=["metadatas"])
        pairs = sorted(zip(records["ids"], records["metadatas"]), key=lambda pair: pair[1]["time_ns"])
        if not pairs:
            return [], None
        latest = pairs[-self.recent_limit:]
        batch = self.collection.get(ids=[record_id for record_id, _ in latest], include=["documents"])
        documents = dict(zip(batch["ids"], batch["documents"]))
        turns = [json.loads(documents[record_id]) for record_id, _ in latest]
        return turns, latest[0][1]["time_ns"] if len(pairs) > len(latest) else None

    def recent(self, conversation_id: str) -> list[Turn]:
        """Return recent turns in chronological order, including after a restart."""
        return self._recent(conversation_id)[0]

    def history(self, conversation_id: str) -> list[Turn]:
        """Read the stored transcript in order, independent of embedding or prompt limits."""
        records = self.collection.get(
            where={"conversation_id": conversation_id}, include=["documents", "metadatas"],
        )
        ordered = sorted(
            zip(records["ids"], records["documents"], records["metadatas"]),
            key=lambda record: (record[2]["time_ns"], record[0]),
        )
        return [json.loads(document) for _, document, _ in ordered]

    def recall(self, conversation_id: str, question: str) -> tuple[list[Turn], list[Turn]]:
        """Return recent turns and relevant older pairs as separate groups."""
        recent, cutoff = self._recent(conversation_id)
        if cutoff is None:
            return recent, []
        try:
            records = self.collection.query(
                query_embeddings=[self._embed(question)],
                n_results=self.recall_limit,
                where={"$and": [{"conversation_id": conversation_id}, {"time_ns": {"$lt": cutoff}}]},
                include=["documents", "distances"],
            )
            remembered = [json.loads(document) for document, distance in zip(
                records["documents"][0], records["distances"][0]
            ) if distance <= 0.55]
        except Exception:
            logger.warning("Semantic recall unavailable; continuing with recent turns")
            remembered = []
        return recent, remembered

    def remember(self, conversation_id: str, turn: Turn) -> None:
        """Write both sides of an exchange atomically under a restart-safe ID."""
        self.collection.add(
            ids=[uuid4().hex],
            embeddings=[self._embed(turn["user"])],
            documents=[json.dumps(turn, ensure_ascii=False)],
            metadatas=[{"conversation_id": conversation_id, "time_ns": time.time_ns()}],
        )

    def clear(self, conversation_id: str) -> None:
        """Remove the selected conversation from active memory."""
        self.collection.delete(where={"conversation_id": conversation_id})
