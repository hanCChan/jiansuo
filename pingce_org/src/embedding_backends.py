"""Embedding backends for intent retrieval evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def _normalize_device(device: str) -> str:
    if device in {"cuda", "cpu"}:
        return device
    if device.startswith("cuda:"):
        return device
    if device.isdigit():
        return f"cuda:{device}"
    return device


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return vectors / norms


class EmbeddingBackend(ABC):
    @abstractmethod
    def set_mode(self, mode: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def score(self, query: str, candidates: list[str]) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class DenseBackend(EmbeddingBackend):
    """SentenceTransformer dense cosine retrieval."""

    DEFAULT_E5_TASK = (
        "Given a user question in Modern Standard Arabic, "
        "retrieve the matching FAQ question"
    )
    DEFAULT_GEMMA_TASK = "search result"

    def __init__(
        self,
        model_path: str,
        device: str,
        batch_size: int,
        model_cfg: dict[str, Any],
        mode: str = "dense",
    ) -> None:
        self.mode = mode
        self.batch_size = batch_size
        self.model_cfg = model_cfg
        self.query_style = model_cfg.get("query_style", "plain")
        self.task_description = model_cfg.get("task_description", self.DEFAULT_E5_TASK)
        self.gemma_task = model_cfg.get("gemma_task", self.DEFAULT_GEMMA_TASK)

        resolved_device = _normalize_device(device)
        model_kwargs: dict[str, Any] = {}
        if model_cfg.get("use_fp16", True) and resolved_device.startswith("cuda"):
            model_kwargs["torch_dtype"] = torch.float16

        self.model = SentenceTransformer(
            model_path,
            device=resolved_device,
            model_kwargs=model_kwargs or None,
        )
        self.model.eval()

    def set_mode(self, mode: str) -> None:
        if mode != "dense":
            raise ValueError(f"DenseBackend only supports mode=dense, got {mode}")
        self.mode = mode

    def _format_query(self, query: str) -> str:
        if self.query_style == "e5_instruct":
            return f"Instruct: {self.task_description}\nQuery: {query}"
        if self.query_style == "snowflake_prefix":
            return f"query: {query}"
        if self.query_style == "qwen3_prompt":
            return query
        if self.query_style == "gemma_retrieval":
            return f"task: {self.gemma_task} | query: {query}"
        return query

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        if self.query_style == "qwen3_prompt":
            return np.asarray(
                self.model.encode(
                    queries,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    prompt_name="query",
                ),
                dtype=np.float32,
            )

        formatted = [self._format_query(q) for q in queries]
        if self.query_style == "gemma_retrieval" and hasattr(self.model, "encode_query"):
            if len(formatted) == 1:
                return np.asarray(
                    self.model.encode_query(
                        formatted[0],
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    ),
                    dtype=np.float32,
                ).reshape(1, -1)
            return np.asarray(
                self.model.encode(
                    formatted,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )

        return np.asarray(
            self.model.encode(
                formatted,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    def _encode_documents(self, documents: list[str]) -> np.ndarray:
        if self.query_style == "gemma_retrieval" and hasattr(self.model, "encode_document"):
            return np.asarray(
                self.model.encode_document(
                    documents,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ),
                dtype=np.float32,
            )
        return np.asarray(
            self.model.encode(
                documents,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ),
            dtype=np.float32,
        )

    def score(self, query: str, candidates: list[str]) -> np.ndarray:
        if not candidates:
            return np.asarray([], dtype=np.float32)

        query_emb = _l2_normalize(self._encode_queries([query]))
        doc_emb = _l2_normalize(self._encode_documents(candidates))
        return (query_emb @ doc_emb.T).reshape(-1).astype(np.float32)

    def close(self) -> None:
        del self.model


class BgeM3Backend(EmbeddingBackend):
    """BGE-M3 multi-mode retrieval via FlagEmbedding."""

    MODE_KEYS = {
        "dense": "dense",
        "sparse": "sparse",
        "colbert": "colbert",
        "hybrid": "colbert+sparse+dense",
        "dense+sparse": "sparse+dense",
    }

    def __init__(
        self,
        model_path: str,
        device: str,
        batch_size: int,
        model_cfg: dict[str, Any],
        mode: str = "dense",
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel

        self.mode = mode
        self.batch_size = batch_size
        self.model_cfg = model_cfg
        self.hybrid_weights = list(model_cfg.get("hybrid_weights", [0.4, 0.2, 0.4]))
        self.max_length = int(model_cfg.get("max_length", 8192))
        self.pair_batch_size = int(model_cfg.get("pair_batch_size", 64))

        resolved_device = _normalize_device(device)
        use_fp16 = bool(model_cfg.get("use_fp16", True))
        self.model = BGEM3FlagModel(
            model_path,
            use_fp16=use_fp16,
            device=resolved_device,
        )

    def set_mode(self, mode: str) -> None:
        if mode not in self.MODE_KEYS:
            raise ValueError(f"BgeM3Backend unsupported mode: {mode}")
        self.mode = mode

    def _encode_batch(
        self,
        texts: list[str],
        *,
        return_dense: bool,
        return_sparse: bool,
        return_colbert: bool,
    ) -> dict:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert,
        )

    def _score_dense(self, query: str, candidates: list[str]) -> np.ndarray:
        q_out = self._encode_batch([query], return_dense=True, return_sparse=False, return_colbert=False)
        c_out = self._encode_batch(candidates, return_dense=True, return_sparse=False, return_colbert=False)
        q = _l2_normalize(np.asarray(q_out["dense_vecs"], dtype=np.float32))
        c = _l2_normalize(np.asarray(c_out["dense_vecs"], dtype=np.float32))
        return (q @ c.T).reshape(-1)

    def _score_sparse(self, query: str, candidates: list[str]) -> np.ndarray:
        q_out = self._encode_batch([query], return_dense=False, return_sparse=True, return_colbert=False)
        c_out = self._encode_batch(candidates, return_dense=False, return_sparse=True, return_colbert=False)
        q_lex = q_out["lexical_weights"][0]
        scores = np.empty(len(candidates), dtype=np.float32)
        for i, c_lex in enumerate(c_out["lexical_weights"]):
            scores[i] = float(self.model.compute_lexical_matching_score(q_lex, c_lex))
        return scores

    def _score_colbert(self, query: str, candidates: list[str]) -> np.ndarray:
        q_out = self._encode_batch([query], return_dense=False, return_sparse=False, return_colbert=True)
        c_out = self._encode_batch(candidates, return_dense=False, return_sparse=False, return_colbert=True)
        q_vec = q_out["colbert_vecs"][0]
        scores = np.empty(len(candidates), dtype=np.float32)
        for i, c_vec in enumerate(c_out["colbert_vecs"]):
            scores[i] = float(self.model.colbert_score(q_vec, c_vec))
        return scores

    def _score_hybrid_like(self, query: str, candidates: list[str], mode: str) -> np.ndarray:
        weights = self.hybrid_weights
        if mode == "dense+sparse":
            d, s, _ = weights if len(weights) == 3 else (0.6, 0.4, 0.0)
            weights = [float(d), float(s), 0.0]

        scores = np.empty(len(candidates), dtype=np.float32)
        for start in range(0, len(candidates), self.pair_batch_size):
            chunk = candidates[start : start + self.pair_batch_size]
            pairs = [[query, cand] for cand in chunk]
            result = self.model.compute_score(
                pairs,
                max_passage_length=self.max_length,
                weights_for_different_modes=weights,
            )
            key = self.MODE_KEYS[mode]
            chunk_scores = result[key]
            scores[start : start + len(chunk)] = np.asarray(chunk_scores, dtype=np.float32)
        return scores

    def score(self, query: str, candidates: list[str]) -> np.ndarray:
        if not candidates:
            return np.asarray([], dtype=np.float32)
        if self.mode == "dense":
            return self._score_dense(query, candidates)
        if self.mode == "sparse":
            return self._score_sparse(query, candidates)
        if self.mode == "colbert":
            return self._score_colbert(query, candidates)
        if self.mode in {"hybrid", "dense+sparse"}:
            return self._score_hybrid_like(query, candidates, self.mode)
        raise ValueError(f"unsupported BGE-M3 mode: {self.mode}")

    def close(self) -> None:
        del self.model


def create_backend(
    model_cfg: dict[str, Any],
    mode: str,
    device: str,
    batch_size: int,
) -> EmbeddingBackend:
    backend_name = model_cfg.get("backend", "dense")
    model_path = model_cfg["path"]

    if backend_name == "dense":
        return DenseBackend(model_path, device, batch_size, model_cfg, mode=mode)

    if backend_name == "bge_m3":
        return BgeM3Backend(model_path, device, batch_size, model_cfg, mode=mode)

    raise ValueError(
        f"Unsupported backend '{backend_name}'. "
        "Supported backends: dense, bge_m3."
    )
