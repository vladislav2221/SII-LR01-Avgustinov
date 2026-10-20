"""One shared token-based chunker used during indexing and question processing."""
from typing import Any, List


def chunk_text(
    text: str,
    tokenizer: Any,
    chunk_size: int = 320,
    overlap: int = 48,
) -> List[str]:
    """Split text into token windows.

    The overlap must be smaller than the chunk size. The tokenizer is taken
    from the same SentenceTransformer model used for both documents and queries.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    text = (text or "").strip()
    if not text:
        return []

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if not token_ids:
        return []

    step = chunk_size - overlap
    chunks = []
    for start in range(0, len(token_ids), step):
        window = token_ids[start : start + chunk_size]
        if not window:
            break
        chunk = tokenizer.decode(
            window,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(token_ids):
            break
    return chunks
