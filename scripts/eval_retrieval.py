"""
Small retrieval ablation for the Dev 2 part.
Compares dense only, BM25 only, hybrid (RRF) and hybrid + reranker on a few
hand-written questions where we know on which page(s) the answer is.

Run from the repo root:  python -m scripts.eval_retrieval
"""
from src import config
from src.retrieval.dense_search import DenseSearch
from src.retrieval.sparse_search import SparseSearch
from src.retrieval.hybrid_fusion import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker

K = 5

# (question, pages of the PDF that contain the answer)
TEST_SET = [
    ("What does the Mirror of Erised show to the person looking into it?", {172}),
    ("Who is Nicolas Flamel?", {176, 177}),
    ("What name does Hagrid give to his baby dragon?", {189}),
    ("What is Harry's wand made of?", {67}),
    ("From which platform does the Hogwarts Express leave?", {70, 71}),
    ("What is the number of the Gringotts vault that Hagrid emptied?", {58, 60, 131}),
    ("Where was Hermione when the troll got into the castle?", {138, 139, 141}),
    ("How many points does a team get for catching the Golden Snitch?", {136, 196}),
    ("What shape is the scar on Harry's forehead?", {12, 15}),
    ("Which position does Harry play on the Gryffindor Quidditch team?", {121, 122}),
]


def score(results, gold_pages):
    """Returns (hit, reciprocal rank) for the top K results."""
    for rank, r in enumerate(results[:K], start=1):
        if r["metadata"]["page"] in gold_pages:
            return 1, 1.0 / rank
    return 0, 0.0


def main():
    dense, sparse, reranker = DenseSearch(), SparseSearch(), Reranker()

    systems = {
        "dense only": lambda q: dense.search(q),
        "bm25 only": lambda q: sparse.search(q),
        "hybrid (RRF)": lambda q: reciprocal_rank_fusion(dense.search(q), sparse.search(q)),
        "hybrid + rerank": lambda q: reranker.rerank(
            q, reciprocal_rank_fusion(dense.search(q), sparse.search(q))[:config.RERANK_TOP_N], K),
    }

    print(f"{'system':<18}{'Hit@' + str(K):>8}{'MRR':>8}")
    for name, run in systems.items():
        hits, rr = 0, 0.0
        for question, gold in TEST_SET:
            h, r = score(run(question), gold)
            hits += h
            rr += r
        n = len(TEST_SET)
        print(f"{name:<18}{hits / n:>8.2f}{rr / n:>8.2f}")


if __name__ == "__main__":
    main()
