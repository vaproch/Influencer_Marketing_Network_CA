#!/usr/bin/env python3
"""
network_diagnostics.py - structural properties of the original ego-Twitter
network and of the randomised versions used in the robustness checks.

It rebuilds the networks EXACTLY as simulation/04_robustness.py and
simulation/05_partial_rewiring.py did (same cleaning,
same random seed, same shuffling), so the numbers describe the networks that
were actually simulated:

    original  : 0% of edges randomised
    rewire10 / rewire25 / rewire50 : 10 / 25 / 50% (05_partial_rewiring.py)
    rewired   : 100% (04_robustness.py)

For each network it reports
    * number of nodes and directed edges
    * self-loops and duplicate edges created by the randomisation
      (they were left in place in the simulations)
    * the largest change in any account's number of followers or followed
      accounts (must be 0: degrees are preserved)
    * reciprocity: share of directed edges whose reverse edge also exists
    * transitivity (global clustering) of the undirected network, estimated by
      sampling wedges (open and closed triples), with a standard error
    * degree assortativity of the undirected network
    * share of accounts in the largest weakly connected component
      (only if scipy is installed)

Self-loops and duplicate edges are removed before reciprocity, transitivity,
assortativity and components are computed.

Usage (from the repository root; downloads the edge list if needed):
    python analysis/network_diagnostics.py
    python analysis/network_diagnostics.py --wedges 1000000

Writes analysis/output/network_diagnostics.csv and prints a table.
"""

import argparse
import csv
import os
import urllib.request
from pathlib import Path

import numpy as np

SNAP_URL = "https://snap.stanford.edu/data/twitter_combined.txt.gz"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_GZ = RAW_DIR / "twitter_combined.txt.gz"
OUT_CSV = ROOT / "analysis" / "output" / "network_diagnostics.csv"
FIRST_COLUMN_FOLLOWS_SECOND = True        # same as in simulation/*.py
BASE_SEED = 12345
REWIRE_SEED = BASE_SEED + 777             # same as in simulation/04 and 05


def load_edges():
    """Same cleaning as the simulation: src = followed account, tgt = follower."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(RAW_GZ):
        print("Downloading ego-Twitter from SNAP...")
        urllib.request.urlretrieve(SNAP_URL, RAW_GZ)
    print(f"Parsing '{RAW_GZ}'...")
    edges = np.loadtxt(RAW_GZ, dtype=np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(edges, axis=0)
    unique_ids, inverse = np.unique(edges, return_inverse=True)
    inverse = inverse.reshape(edges.shape)
    follower, followee = ((inverse[:, 0], inverse[:, 1]) if FIRST_COLUMN_FOLLOWS_SECOND
                          else (inverse[:, 1], inverse[:, 0]))
    return followee.astype(np.int64), follower.astype(np.int64), len(unique_ids)


def randomise(src, tgt, share):
    """Shuffle the targets of a share of the edges (as in simulation/04 and 05)."""
    if share <= 0:
        return src.copy(), tgt.copy()
    tgt = tgt.copy()
    rng = np.random.default_rng(REWIRE_SEED)
    if share >= 1.0:
        tgt = tgt[rng.permutation(len(tgt))]
    else:
        m = len(tgt)
        idx = rng.choice(m, size=int(round(share * m)), replace=False)
        tgt[idx] = tgt[idx][rng.permutation(len(idx))]
    return src.copy(), tgt


def diagnostics(name, src, tgt, n, ref_src, ref_tgt, n_wedges, rng):
    row = {"network": name, "nodes": n, "edges": len(src)}
    row["self_loops"] = int((src == tgt).sum())
    keys = src.astype(np.int64) * n + tgt
    row["duplicate_edges"] = int(len(keys) - len(np.unique(keys)))

    d_followers = np.abs(np.bincount(src, minlength=n) - np.bincount(ref_src, minlength=n)).max()
    d_following = np.abs(np.bincount(tgt, minlength=n) - np.bincount(ref_tgt, minlength=n)).max()
    row["max_degree_change"] = int(max(d_followers, d_following))

    # simple directed graph (no self-loops, no duplicates)
    keep = src != tgt
    dkeys = np.unique(src[keep].astype(np.int64) * n + tgt[keep])
    s, t = dkeys // n, dkeys % n
    reverse = t * n + s
    row["reciprocity"] = float(np.isin(reverse, dkeys).mean())

    # undirected simple graph
    lo, hi = np.minimum(s, t), np.maximum(s, t)
    ukeys = np.unique(lo * n + hi)
    u, v = ukeys // n, ukeys % n
    deg = np.bincount(np.concatenate([u, v]), minlength=n)

    # degree assortativity (Pearson correlation of end-point degrees)
    a = np.concatenate([deg[u], deg[v]]).astype(float)
    b = np.concatenate([deg[v], deg[u]]).astype(float)
    row["degree_assortativity"] = float(np.corrcoef(a, b)[0, 1])

    # transitivity by wedge sampling
    a_nodes = np.concatenate([u, v])
    b_nodes = np.concatenate([v, u])
    order = np.argsort(a_nodes, kind="stable")
    nbrs = b_nodes[order]
    ptr = np.concatenate([[0], np.cumsum(deg)])
    weights = deg * (deg - 1) / 2.0
    centres = rng.choice(n, size=n_wedges, p=weights / weights.sum())
    dc = deg[centres]
    r1 = np.floor(rng.random(n_wedges) * dc).astype(np.int64)
    r2 = np.floor(rng.random(n_wedges) * (dc - 1)).astype(np.int64)
    r2 = r2 + (r2 >= r1)
    x = nbrs[ptr[centres] + r1]
    y = nbrs[ptr[centres] + r2]
    wk = np.minimum(x, y) * n + np.maximum(x, y)
    pos = np.searchsorted(ukeys, wk)
    pos[pos == len(ukeys)] = len(ukeys) - 1
    closed = ukeys[pos] == wk
    p = closed.mean()
    row["transitivity"] = float(p)
    row["transitivity_se"] = float(np.sqrt(p * (1 - p) / n_wedges))

    # largest weakly connected component
    try:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        g = coo_matrix((np.ones(len(u)), (u, v)), shape=(n, n))
        _, labels = connected_components(g, directed=False)
        row["largest_component_share"] = float(np.bincount(labels).max() / n)
    except ImportError:
        row["largest_component_share"] = float("nan")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--wedges", type=int, default=400_000,
                    help="wedges sampled for the transitivity estimate (default 400000)")
    args = ap.parse_args()

    src, tgt, n = load_edges()
    print(f"Network: {n:,} accounts, {len(src):,} directed edges.\n")

    rows = []
    for name, share in [("original (0%)", 0.0), ("randomised 10%", 0.10),
                        ("randomised 25%", 0.25), ("randomised 50%", 0.50),
                        ("randomised 100%", 1.0)]:
        print(f"Computing: {name} ...")
        s2, t2 = randomise(src, tgt, share)
        rows.append(diagnostics(name, s2, t2, n, src, tgt, args.wedges,
                                np.random.default_rng(BASE_SEED)))

    cols = ["network", "edges", "self_loops", "duplicate_edges", "max_degree_change",
            "reciprocity", "transitivity", "transitivity_se", "degree_assortativity",
            "largest_component_share"]
    print("\n" + " | ".join(cols))
    for r in rows:
        print(" | ".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})
    print(f"\nWritten to {OUT_CSV}")


if __name__ == "__main__":
    main()
