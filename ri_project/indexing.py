"""Construção do índice invertido e estatísticas do corpus."""
from collections import Counter, defaultdict
from time import perf_counter

from .preprocessing import preprocess_tokens, tokenize


def build_inverted_index(documents):
    """Cria postings por termo, comprimentos, vocabulário e estatísticas."""
    start = perf_counter()
    index = defaultdict(dict)
    document_lengths = {}
    raw_vocabulary = set()

    for doc_id, document in documents.items():
        raw_tokens = tokenize(
            f"{document.get('title', '')} {document.get('text', '')}"
        )
        raw_vocabulary.update(raw_tokens)
        term_frequencies = Counter(preprocess_tokens(raw_tokens))
        document_lengths[doc_id] = sum(term_frequencies.values())
        for term, frequency in term_frequencies.items():
            index[term][doc_id] = frequency

    inverted_index = dict(index)
    document_count = len(documents)
    average_document_length = (
        sum(document_lengths.values()) / document_count if document_count else 0
    )
    return {
        'inverted_index': inverted_index,
        'document_lengths': document_lengths,
        'raw_vocabulary': raw_vocabulary,
        'num_documents': document_count,
        'avg_document_length': average_document_length,
        'num_postings': sum(map(len, inverted_index.values())),
        'build_time_seconds': perf_counter() - start,
    }
