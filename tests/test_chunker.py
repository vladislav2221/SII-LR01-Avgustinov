from chunker import chunk_text


class TinyTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return " ".join(ids)


def test_chunker_returns_chunks_with_overlap():
    tokenizer = TinyTokenizer()
    words = [f"w{i}" for i in range(700)]
    chunks = chunk_text(" ".join(words), tokenizer, chunk_size=300, overlap=45)
    assert len(chunks) >= 3
    assert len(chunks[0].split()) == 300
    assert chunks[0].split()[-45:] == chunks[1].split()[:45]


def test_empty_text_returns_no_chunks():
    assert chunk_text("  ", TinyTokenizer()) == []
