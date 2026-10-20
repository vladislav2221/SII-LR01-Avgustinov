"""Build/rebuild the persistent ChromaDB index from data/."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import chromadb
from sentence_transformers import SentenceTransformer

from chunker import chunk_text
from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    MODEL_NAME,
)


SUPPORTED_EXTENSIONS = {".txt", ".md"}


def read_document(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(
        p for p in DATA_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not files:
        raise SystemExit(f"В {DATA_DIR} нет .txt/.md документов для индексации")

    print(f"Модель: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = max(model.max_seq_length, CHUNK_SIZE)

    records = []
    for path in files:
        text = read_document(path)
        chunks = chunk_text(text, model.tokenizer, CHUNK_SIZE, CHUNK_OVERLAP)
        relative_name = path.relative_to(DATA_DIR).as_posix()
        for chunk_id, chunk in enumerate(chunks):
            records.append({
                "id": f"{relative_name}::{chunk_id}",
                "document": chunk,
                "metadata": {"file": relative_name, "chunk_id": chunk_id},
            })
        print(f"{relative_name}: {len(chunks)} чанков")

    if not records:
        raise SystemExit("Документы найдены, но в них нет индексируемого текста")

    embeddings = model.encode(
        [record["document"] for record in records],
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=32,
    )

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "model_name": MODEL_NAME},
    )

    batch_size = 256
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        collection.add(
            ids=[r["id"] for r in batch],
            documents=[r["document"] for r in batch],
            metadatas=[r["metadata"] for r in batch],
            embeddings=embeddings[start : start + batch_size].tolist(),
        )

    print(f"Готово: проиндексировано {len(records)} чанков из {len(files)} файлов.")
    print(f"ChromaDB: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
