from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput
import re
import logging
from typing import List, Dict, Optional
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime

logger = logging.getLogger("AgentLogger")


class QwenAssistant:
    def __init__(
        self,
        model_path: str,
        max_tokens: int = 4096,
        temperature: float = 0.5,
        embedding_model: str = "all-MiniLM-L6-v2",  # fast & good enough
        collection_name: str = "conversation_memory",
        top_k: int = 4,
        recent_k: int = 3
    ):
        self.llm = LLM(model=model_path, max_model_len=8192)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.recent_k = recent_k

        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # Conversation history (full, for accuracy)
        self.conversation: List[Dict[str, str]] = []

        # Vector store for retrieval
        self.embedding_model = SentenceTransformer(embedding_model)
        self.chroma_client = chromadb.PersistentClient(path="./conversation_db")
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def _get_embedding(self, text: str) -> List[float]:
        return self.embedding_model.encode(text, normalize_embeddings=True).tolist()

    def _build_context(self, user_input: str, system_prompt: str) -> List[Dict[str, str]]:
        """Build dynamic context with hard token limit awareness."""

        context = []

        if not self.conversation:
            context = [{"role": "system", "content": system_prompt},
                       {"role": "user", "content": user_input}]
            return context

        # === STRONG LIMIT: Keep only recent messages + limited retrieval ===
        MAX_HISTORY_TURNS = 4   # ~ last 4 full exchanges
        recent_messages = self.conversation[-MAX_HISTORY_TURNS * 2:]  

        context.extend(recent_messages)

        # Retrieve only if we have enough room
        if len(self.conversation) > MAX_HISTORY_TURNS * 2:
            try:
                query_embedding = self._get_embedding(user_input)
                results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=4,          # reduced
                    include=["metadatas", "documents"]
                )

                metadatas = results.get("metadatas", [[]])[0]
                retrieved_messages = [
                    {"role": meta["role"], "content": meta["content"]}
                    for meta in metadatas
                ]

                # Deduplicate
                seen = {msg["content"] for msg in context}
                for msg in retrieved_messages:
                    if msg["content"] not in seen:
                        with open("retrived_messages.txt", "w") as f:
                            f.write(str(msg))
                        context.append(msg)
                        seen.add(msg["content"])
            except Exception as e:
                logger.warning(f"Vector retrieval failed: {e}")

        context.append({"role": "system", "content": system_prompt})
        context.append({"role": "user", "content": user_input})
        return context

    def _store_to_vector(self, user_input: str, assistant_response: str):
        """Store both user and assistant messages"""
        timestamp = datetime.now().isoformat()
        
        # Store user message
        self.collection.add(
            documents=[user_input],
            embeddings=[self._get_embedding(user_input)],
            metadatas=[{"role": "user", "content": user_input, "timestamp": timestamp}],
            ids=[f"user_{len(self.conversation)}"]
        )
        
        # Store assistant message
        self.collection.add(
            documents=[assistant_response],
            embeddings=[self._get_embedding(assistant_response)],
            metadatas=[{"role": "assistant", "content": assistant_response, "timestamp": timestamp}],
            ids=[f"assistant_{len(self.conversation)}"]
        )

    def ask_qwen(self, user_input: str, system_prompt: str = "You are a helpful assistant.", save_in_history=True) -> str:

        # Build context with limits
        messages = self._build_context(user_input, system_prompt)

        # Generate
        outputs: List[RequestOutput] = self.llm.chat(
            messages,
            self.sampling_params,
            use_tqdm=False
        )
        
        text = outputs[0].outputs[0].text.strip()
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        if save_in_history:
            self.conversation.append({"role": "user", "content": user_input})
            self.conversation.append({"role": "assistant", "content": text})
            self._store_to_vector(user_input, text)

        return text

    def clear_history(self):
        """Reset conversation"""
        self.conversation.clear()
        # Optional: delete collection and recreate
        try:
            self.chroma_client.delete_collection(self.collection.name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection.name, metadata={"hnsw:space": "cosine"}
            )
        except:
            pass