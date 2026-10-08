# Large-file policy

Git contains compact products needed to inspect the reported calculations:
manifests, score tables, feature matrices, out-of-fold predictions, result
records, manuscript figures, and the visual matching supplement.

The following remain outside Git:

- primary and external encoder checkpoints;
- DR10 and COSMOS source pixels;
- generated recovery and source-photon image banks;
- local caches, databases, and intermediate render shards;
- submission ZIP files and presentation render trees.

No tracked file may exceed GitHub's 100 MB per-file limit. Files above 10 MB
must have a documented scientific or review purpose. The largest committed
arrays are retained because they support direct recomputation of the
representation and checkpoint-panel summaries.

External artifacts are listed in `data/downloads/artifacts.yaml` with their
expected destinations, sizes where known, hashes, and availability state.
When public archival records become active, retrieval URLs should be added
there without changing the reported scientific identities.
