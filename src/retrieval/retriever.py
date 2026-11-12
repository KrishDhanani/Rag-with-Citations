import re
import sys

from src import config
from src.retrieval.dense_search import DenseSearch
from src.retrieval.sparse_search import SparseSearch
from src.retrieval.hybrid_fusion import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker


def make_snippet(text, max_chars=200):
    """Short one-line preview of a chunk for the citation list."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def tag_citations(chunks):
    """Give every retrieved chunk a citation number ([1], [2], ...) and collect
    the fields needed to show the source to the user."""
    tagged = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        # section titles from the PDF can contain line breaks
        section = re.sub(r"\s+", " ", str(meta.get("section", ""))).strip() or None
        tagged.append({
            **chunk,
            "citation_id": f"[{i}]",
            "document": meta.get("source", "unknown"),
            "page": meta.get("page"),
            "section": section,
            "snippet": make_snippet(chunk["text"]),
        })
    return tagged


def build_context(tagged_chunks):
    """Assemble the context block for the LLM prompt. Each passage starts with
    its citation tag so the model can refer to it in the answer."""
    blocks = []
    for c in tagged_chunks:
        header = f"{c['citation_id']} Source: {c['document']}, page {c['page']}"
        blocks.append(f"{header}\n{c['text'].strip()}")
    return "\n\n".join(blocks)


class RetrievalEngine:
    """
    Full retrieval pipeline:
        query -> dense search + BM25 -> RRF fusion -> cross-encoder rerank
              -> citation tagging -> context string

    Usage:
        engine = RetrievalEngine()
        result = engine.retrieve("Who is Hagrid?")
        result["context"]    -> text to put in the prompt
        result["chunks"]     -> list of chunks with citation_id, document, page, snippet
        result["top_score"]  -> confidence (0..1) of the best chunk, for the refusal check
    """

    def __init__(self, dense=None, sparse=None, reranker=None):
        self.dense = dense or DenseSearch()
        self.sparse = sparse or SparseSearch()
        self.reranker = reranker or Reranker()

    def retrieve(self, query, top_k=config.FINAL_TOP_K, use_reranker=True):
        query = query.strip()
        if not query:
            return {"query": query, "chunks": [], "context": "", "top_score": 0.0}

        dense_results = self.dense.search(query, config.DENSE_TOP_K)
        sparse_results = self.sparse.search(query, config.SPARSE_TOP_K)
        fused = reciprocal_rank_fusion(dense_results, sparse_results)

        if use_reranker:
            top_chunks = self.reranker.rerank(query, fused[:config.RERANK_TOP_N], top_k)
        else:
            top_chunks = fused[:top_k]

        chunks = tag_citations(top_chunks)
        # without the reranker there is no calibrated score, so top_score is None
        # pass this to Dev 3 for the refusal threshold
        top_score = chunks[0].get("confidence") if chunks else 0.0

        return {
            "query": query,
            "chunks": chunks,
            "context": build_context(chunks),
            "top_score": top_score,
        }


if __name__ == "__main__":
    # quick manual check:  python -m src.retrieval.retriever "Who gave Harry the invisibility cloak?"
    question = " ".join(sys.argv[1:]) or "Who gave Harry the invisibility cloak?"
    result = RetrievalEngine().retrieve(question)

    print(f"\nQuestion: {question}")
    print(f"Top confidence: {result['top_score']:.3f}\n")
    for c in result["chunks"]:
        print(f"{c['citation_id']} p.{c['page']}  rerank={c['rerank_score']:.2f}  "
              f"dense_rank={c['dense_rank']}  bm25_rank={c['sparse_rank']}")
        print(f"    {c['snippet']}\n")
