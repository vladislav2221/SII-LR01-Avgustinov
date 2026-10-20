"""FastAPI application for the document RAG assistant."""
from functools import lru_cache
from typing import List

import chromadb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from chunker import chunk_text
from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MODEL_NAME,
    SIMILARITY_THRESHOLD,
    TOP_K,
)

app = FastAPI(
    title="RAG-ассистент по документам",
    description="Ответы только по найденным фрагментам локального корпуса.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, description="Вопрос пользователя")


class Source(BaseModel):
    file: str
    chunk_id: int
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]
    model_version: str


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    model = SentenceTransformer(MODEL_NAME)
    # Explicitly allow the configured 320-token chunk size where the model
    # architecture supports it. The chunker still uses the model tokenizer.
    model.max_seq_length = max(model.max_seq_length, CHUNK_SIZE)
    return model


@lru_cache(maxsize=1)
def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        raise RuntimeError(
            "Индекс не найден. Сначала выполните: python scripts/build_index.py"
        ) from exc


@app.get("/")
def root():
    return {
        "service": "RAG-ассистент по документам",
        "docs": "/docs",
        "ask_endpoint": "POST /ask",
    }


@app.get("/health")
def health():
    try:
        collection = get_collection()
        return {"status": "ok", "indexed_chunks": collection.count(), "model_version": MODEL_NAME}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="Вопрос не должен быть пустым")

    try:
        model = get_model()
        collection = get_collection()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    # The same chunker is applied to a query. Usually a question fits into one
    # chunk; if it is longer, its chunk embeddings are averaged into one vector.
    query_chunks = chunk_text(
        question, model.tokenizer, CHUNK_SIZE, CHUNK_OVERLAP
    )
    if not query_chunks:
        return AskResponse(
            answer="Не нашёл информации по вашему вопросу",
            sources=[],
            model_version=MODEL_NAME,
        )

    query_vectors = model.encode(query_chunks, normalize_embeddings=True)
    if len(query_vectors.shape) == 1:
        query_vector = query_vectors
    else:
        import numpy as np
        query_vector = np.mean(query_vectors, axis=0)
        norm = np.linalg.norm(query_vector)
        if norm:
            query_vector = query_vector / norm

    if collection.count() == 0:
        return AskResponse(
            answer="Не нашёл информации по вашему вопросу",
            sources=[],
            model_version=MODEL_NAME,
        )

    result = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    matches = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        similarity = max(-1.0, min(1.0, 1.0 - float(distance)))
        if similarity >= SIMILARITY_THRESHOLD:
            matches.append((document, metadata, similarity))

    if not matches:
        return AskResponse(
            answer="Не нашёл информации по вашему вопросу",
            sources=[],
            model_version=MODEL_NAME,
        )

    # No generative model is used: the answer is composed exclusively from
    # retrieved chunks, so the service cannot add facts beyond the corpus.
    answer_parts = []
    sources = []
    for document, metadata, score in matches:
        answer_parts.append(f"Согласно документу «{metadata['file']}»: {document}")
        sources.append(
            Source(
                file=str(metadata["file"]),
                chunk_id=int(metadata["chunk_id"]),
                score=round(score, 4),
            )
        )

    return AskResponse(
        answer="\n\n".join(answer_parts),
        sources=sources,
        model_version=MODEL_NAME,
    )
