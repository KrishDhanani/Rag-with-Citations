from src import config


def reciprocal_rank_fusion(dense_results, sparse_results, k=config.RRF_K):
    """
    Merge the dense and the BM25 ranking with Reciprocal Rank Fusion:

        score(d) = sum over rankings of 1 / (k + rank(d))

    Only the rank positions are used. This is the reason for choosing RRF over
    a weighted sum: cosine similarities and BM25 scores live on different
    scales, so they cannot be added directly without normalisation.
    """
    fused = {}

    for name, results in (("dense", dense_results), ("sparse", sparse_results)):
        for rank, item in enumerate(results, start=1):
            chunk_id = item["chunk_id"]
            if chunk_id not in fused:
                fused[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": item["text"],
                    "metadata": item["metadata"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "sparse_rank": None,
                }
            # RRF formula: only uses rank position, not raw scores
            fused[chunk_id]["rrf_score"] += 1.0 / (k + rank)
            fused[chunk_id][f"{name}_rank"] = rank    # kept for debugging / analysis

    return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)
