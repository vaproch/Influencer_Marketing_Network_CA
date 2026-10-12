#!/usr/bin/env python3
"""
make_fig2.py - illustration of the four seeding strategies (Figure 2).

What it does
------------
1. Loads the ego-Twitter network (the same file / cache the simulation uses).
2. Draws a small sample of it: a few mega-influencer "circles" (an account plus
   a random subset of the accounts connected to it) and keeps the connections
   among all sampled accounts. This keeps the community structure that matters
   for the paper while being small enough to draw (about 250 accounts).
3. Inside the sample it applies the SAME seeding rules as the simulation, with
   the budget equal to the price of one mega-influencer (price = followers):
       Mega   - the account with the most followers in the sample
       Micro  - random accounts from the 80th-95th follower percentile
       Random - random accounts from everyone
       Niche  - micro accounts found by a breadth-first search from one account,
                so they sit in the same neighbourhood
4. Draws one panel per strategy on the same layout and saves PNG files:
       fig2_strategies_combined.png   all four panels + legend (2 x 2)
       fig2_a_mega.png, fig2_b_micro.png, fig2_c_random.png, fig2_d_niche.png
       fig2_legend.png                legend only
   In the panels, large blue circles are seed accounts, colours show how many
   seed accounts reach a person (their audience), grey accounts are not reached.

IMPORTANT: this is an ILLUSTRATION, not a result. The seeds are one random draw
in a small sample, and --search picks the first draw in which the contrast
between the strategies is clearly visible. Say so in the figure caption.

Usage
-----
    pip install numpy matplotlib networkx
    python make_fig2.py
    python make_fig2.py --seed 11 --anchors 5 --per-anchor 50 --outdir fig2

Run it in the folder that contains twitter_combined.txt.gz or twitter_graph.pt.
If neither exists, the edge list is downloaded from SNAP.
"""

import argparse
import csv
import os
import urllib.request
from collections import deque

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import networkx as nx

SNAP_URL = "https://snap.stanford.edu/data/twitter_combined.txt.gz"
RAW_GZ = "twitter_combined.txt.gz"
CACHE_PT = "twitter_graph.pt"
FIRST_COLUMN_FOLLOWS_SECOND = True     # keep identical to main_v*.py

STRATEGIES = ["Mega", "Micro", "Random", "Niche"]
LETTERS = {"Mega": "a", "Micro": "b", "Random": "c", "Niche": "d"}
SEED_COLOR = "#1f4e9e"
NEUTRAL = "#d0d0d0"
EXPOSURE_COLORS = {1: "#fdd49e", 2: "#fc8d59", 3: "#d7301f"}   # 1, 2, 3+ seeds


# ----------------------------------------------------------------------------
# Loading the network
# ----------------------------------------------------------------------------
def load_graph():
    """Returns (src, tgt, n): influence flows src -> tgt (src = followed account)."""
    if os.path.exists(CACHE_PT):
        try:
            import torch
            d = torch.load(CACHE_PT)
            print(f"Loaded cached graph '{CACHE_PT}'.")
            return (d["src"].numpy().astype(np.int64),
                    d["tgt"].numpy().astype(np.int64), int(d["num_nodes"]))
        except ImportError:
            print("PyTorch not installed; reading the raw edge list instead.")

    if not os.path.exists(RAW_GZ):
        print("Downloading ego-Twitter from SNAP...")
        urllib.request.urlretrieve(SNAP_URL, RAW_GZ)

    print(f"Parsing '{RAW_GZ}' (takes a few seconds)...")
    edges = np.loadtxt(RAW_GZ, dtype=np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(edges, axis=0)
    unique_ids, inverse = np.unique(edges, return_inverse=True)
    inverse = inverse.reshape(edges.shape)
    follower, followee = ((inverse[:, 0], inverse[:, 1]) if FIRST_COLUMN_FOLLOWS_SECOND
                          else (inverse[:, 1], inverse[:, 0]))
    return followee.astype(np.int64), follower.astype(np.int64), len(unique_ids)


def undirected_csr(a_nodes, b_nodes, n):
    a = np.concatenate([a_nodes, b_nodes])
    b = np.concatenate([b_nodes, a_nodes])
    order = np.argsort(a, kind="stable")
    ptr = np.concatenate([[0], np.cumsum(np.bincount(a, minlength=n))])
    return ptr, b[order]


# ----------------------------------------------------------------------------
# Sampling a small sub-network
# ----------------------------------------------------------------------------
def sample_subnetwork(full, rng, n_anchors, per_anchor):
    """A few mega 'circles': anchor account + random connected accounts."""
    followers = full["followers"]
    mega_pool = np.where(followers >= np.quantile(followers, 0.99))[0]

    anchors, neighbour_sets = [], []
    for cand in rng.permutation(mega_pool):
        nb = np.unique(full["nbrs"][full["ptr"][cand]:full["ptr"][cand + 1]])
        nb = nb[nb != cand]
        if len(nb) >= per_anchor:
            anchors.append(int(cand))
            neighbour_sets.append(nb)
        if len(anchors) == n_anchors:
            break
    if len(anchors) < n_anchors:
        raise RuntimeError("Not enough anchors with that many neighbours; "
                           "lower --per-anchor or --anchors.")

    circle_of = {a: i for i, a in enumerate(anchors)}
    for i, nb in enumerate(neighbour_sets):
        for x in rng.choice(nb, size=per_anchor, replace=False):
            circle_of.setdefault(int(x), i)          # first circle wins
    nodes = np.array(sorted(circle_of))
    circle = np.array([circle_of[int(x)] for x in nodes])

    n = full["n"]
    local = np.full(n, -1, dtype=np.int64)
    local[nodes] = np.arange(len(nodes))
    keep = (local[full["src"]] >= 0) & (local[full["tgt"]] >= 0)
    s, t = local[full["src"][keep]], local[full["tgt"][keep]]

    used = np.unique(np.concatenate([s, t]))          # drop accounts with no edge
    remap = np.full(len(nodes), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    s, t, m = remap[s], remap[t], len(used)

    ptr, nbrs = undirected_csr(s, t, m)
    return {"m": m, "s": s, "t": t, "followers": np.bincount(s, minlength=m),
            "ptr": ptr, "nbrs": nbrs, "circle": circle[used], "n_circles": len(anchors)}


# ----------------------------------------------------------------------------
# Seeding rules (same logic as the simulation, applied inside the sample)
# ----------------------------------------------------------------------------
def draw_budget_matched(pool, cost, budget, rng, tol):
    selected, spent = [], 0.0
    upper, lower = budget * (1 + tol), budget * (1 - tol)
    for node in rng.permutation(pool):
        c = cost[node]
        if spent + c > upper:
            continue
        selected.append(int(node))
        spent += c
        if spent >= lower:
            break
    return np.array(selected, dtype=np.int64), spent


def draw_niche(sm, pool, cost, budget, rng, tol):
    in_pool = np.zeros(sm["m"], dtype=bool)
    in_pool[pool] = True
    visited = np.zeros(sm["m"], dtype=bool)
    upper, lower = budget * (1 + tol), budget * (1 - tol)
    selected, spent = [], 0.0
    for start in rng.permutation(pool):
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([int(start)])
        while queue:
            v = queue.popleft()
            if in_pool[v] and spent + cost[v] <= upper:
                selected.append(v)
                spent += cost[v]
                if spent >= lower:
                    return np.array(selected, dtype=np.int64), spent
            nb = sm["nbrs"][sm["ptr"][v]:sm["ptr"][v + 1]]
            fresh = np.unique(nb[~visited[nb]])
            if len(fresh):
                visited[fresh] = True
                queue.extend(rng.permutation(fresh).tolist())
    return np.array(selected, dtype=np.int64), spent


def pick_seed_sets(sm, rng, gamma, tol):
    f = sm["followers"]
    cost = np.maximum(f, 1.0) ** gamma
    mega = int(np.argmax(f))
    budget = float(cost[mega])

    p80, p95 = np.quantile(f, 0.80), np.quantile(f, 0.95)
    micro_pool = np.where((f >= p80) & (f <= p95))[0]
    micro_pool = micro_pool[micro_pool != mega]
    all_pool = np.array([i for i in range(sm["m"]) if i != mega])   # do not re-draw the mega
    if len(micro_pool) < 5:
        return None, budget

    sets = {"Mega": (np.array([mega]), budget)}
    sets["Micro"] = draw_budget_matched(micro_pool, cost, budget, rng, tol)
    sets["Random"] = draw_budget_matched(all_pool, cost, budget, rng, tol)
    sets["Niche"] = draw_niche(sm, micro_pool, cost, budget, rng, tol)
    return sets, budget


def exposure_stats(sm, seeds):
    is_seed = np.zeros(sm["m"], dtype=bool)
    is_seed[seeds] = True
    mask = is_seed[sm["s"]]                              # edges leaving a seed
    expo = np.bincount(sm["t"][mask], minlength=sm["m"])  # seeds reaching each person
    reach = int((expo > 0).sum())
    overlap = float(expo.sum() / reach) if reach else 1.0
    return {"expo": expo, "mask": mask, "reach": reach, "overlap": overlap}


def evaluate(sets, stats, budget):
    """Is the contrast clearly visible? (used only to choose a nice illustration)"""
    n = {k: len(sets[k][0]) for k in sets}
    ov = {k: stats[k]["overlap"] for k in sets}
    ratios = [sets[k][1] / budget for k in sets]
    other = max(ov["Micro"], ov["Random"])
    ok = (budget >= 30 and 4 <= n["Micro"] <= 14 and 4 <= n["Niche"] <= 14
          and n["Random"] <= 40 and min(ratios) >= 0.85 and max(ratios) <= 1.15
          and ov["Niche"] >= max(2.0, 1.6 * other))
    return ok, ov["Niche"] - other


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def compute_layout(sm, seed, spread):
    """Layout identical for all panels: every circle is laid out on its own
    (force-directed) and the circles are placed evenly on a ring."""
    m, k = sm["m"], sm["n_circles"]
    xy = np.zeros((m, 2))
    ring = 1.5 if k > 1 else 0.0
    for c in range(k):
        idx = np.where(sm["circle"] == c)[0]
        pos_of = {int(v): j for j, v in enumerate(idx)}
        G = nx.Graph()
        G.add_nodes_from(range(len(idx)))
        inside = np.isin(sm["s"], idx) & np.isin(sm["t"], idx)
        G.add_edges_from((pos_of[int(u)], pos_of[int(v)])
                         for u, v in zip(sm["s"][inside], sm["t"][inside]))
        lay = nx.spring_layout(G, k=spread / np.sqrt(len(idx)), iterations=300,
                               seed=seed + c)
        P = np.array([lay[j] for j in range(len(idx))])
        P -= P.mean(axis=0)
        P /= np.abs(P).max()
        angle = 2 * np.pi * c / k + np.pi / 4
        centre = ring * np.array([np.cos(angle), np.sin(angle)])
        xy[idx] = centre + 0.9 * P
    return xy


def draw_panel(ax, sm, xy, bg_segments, name, sets, stats, budget, titled=True):
    seeds, spent = sets[name]
    st = stats[name]
    expo, mask = st["expo"], st["mask"]

    ax.add_collection(LineCollection(bg_segments, colors="#b0b0b0", linewidths=0.25,
                                     alpha=0.22, zorder=1))
    seg = np.stack([xy[sm["s"][mask]], xy[sm["t"][mask]]], axis=1)
    ax.add_collection(LineCollection(seg, colors="#555555", linewidths=0.5,
                                     alpha=0.35, zorder=2))

    colors = np.array([NEUTRAL] * sm["m"], dtype=object)
    sizes = np.full(sm["m"], 22.0)
    for k, col in EXPOSURE_COLORS.items():
        sel = (expo == k) if k < 3 else (expo >= 3)
        colors[sel] = col
        sizes[sel] = 46.0
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=list(colors), linewidths=0, zorder=3)

    seed_sizes = 60 + 26 * np.sqrt(sm["followers"][seeds])
    ax.scatter(xy[seeds, 0], xy[seeds, 1], s=seed_sizes, c=SEED_COLOR,
               edgecolors="black", linewidths=0.8, zorder=5)

    pad = 0.04
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.axis("off")
    if titled:
        ax.set_title(f"({LETTERS[name]}) {name}", fontsize=13, fontweight="bold")
    label = (f"{len(seeds)} seed{'s' if len(seeds) != 1 else ''} · spend {spent:.0f}"
             f" of {budget:.0f}\nreach {st['reach']} people · overlap {st['overlap']:.2f}")
    ax.text(0.5, -0.02, label, transform=ax.transAxes, ha="center", va="top", fontsize=9.5)


def legend_handles():
    def dot(color, label, size=8, edge="none"):
        return Line2D([0], [0], marker="o", linestyle="", markersize=size,
                      markerfacecolor=color, markeredgecolor=edge, label=label)
    return [dot(SEED_COLOR, "Seed account", 11, "black"),
            dot(EXPOSURE_COLORS[1], "Reached by 1 seed"),
            dot(EXPOSURE_COLORS[2], "Reached by 2 seeds"),
            dot(EXPOSURE_COLORS[3], "Reached by 3+ seeds"),
            dot(NEUTRAL, "Not reached")]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--seed", type=int, default=7, help="random seed (default 7)")
    ap.add_argument("--anchors", type=int, default=4, help="number of circles (default 4)")
    ap.add_argument("--per-anchor", type=int, default=60,
                    help="accounts sampled per circle (default 60)")
    ap.add_argument("--gamma", type=float, default=1.0, help="price exponent (default 1)")
    ap.add_argument("--tol", type=float, default=0.10,
                    help="budget tolerance in this small sample (default 0.10)")
    ap.add_argument("--search", type=int, default=60,
                    help="draws to try for a clear illustration (default 60)")
    ap.add_argument("--spread", type=float, default=2.0,
                    help="spacing inside each circle; larger = looser (default 2.0)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--outdir", default="fig2")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    src, tgt, n = load_graph()
    followers = np.bincount(src, minlength=n)
    ptr, nbrs = undirected_csr(src, tgt, n)
    full = {"src": src, "tgt": tgt, "n": n, "followers": followers, "ptr": ptr, "nbrs": nbrs}
    print(f"Network: {n:,} accounts, {len(src):,} edges.")

    best = None
    for trial in range(args.search):
        rng = np.random.default_rng(args.seed + trial)
        sm = sample_subnetwork(full, rng, args.anchors, args.per_anchor)
        sets, budget = pick_seed_sets(sm, rng, args.gamma, args.tol)
        if sets is None:
            continue
        stats = {k: exposure_stats(sm, sets[k][0]) for k in sets}
        ok, score = evaluate(sets, stats, budget)
        if best is None or score > best["score"]:
            best = {"score": score, "trial": trial, "sm": sm, "sets": sets,
                    "stats": stats, "budget": budget, "ok": ok}
        if ok:
            best = {"score": score, "trial": trial, "sm": sm, "sets": sets,
                    "stats": stats, "budget": budget, "ok": ok}
            break
    if best is None:
        raise RuntimeError("No usable sample found; try other --seed / --per-anchor.")
    if not best["ok"]:
        print("Warning: no draw met all illustration criteria; using the best one found. "
              "Try --search 200 or another --seed.")
    sm, sets, stats, budget = best["sm"], best["sets"], best["stats"], best["budget"]
    print(f"Using draw #{best['trial']} (seed {args.seed + best['trial']}): "
          f"{sm['m']} accounts, {len(sm['s'])} edges, budget {budget:.0f}.")

    # one common layout for all panels
    xy = compute_layout(sm, args.seed, args.spread)

    all_seg = np.stack([xy[sm["s"]], xy[sm["t"]]], axis=1)
    if len(all_seg) > 2000:
        keep = np.random.default_rng(args.seed).choice(len(all_seg), 2000, replace=False)
        all_seg = all_seg[keep]

    # summary table
    print("\nStrategy  seeds  spend  reach  overlap")
    rows = []
    for name in STRATEGIES:
        seeds, spent = sets[name]
        st = stats[name]
        print(f"{name:<8} {len(seeds):>5} {spent:>6.0f} {st['reach']:>6} {st['overlap']:>8.2f}")
        rows.append([name, len(seeds), round(spent, 1), st["reach"], round(st["overlap"], 3)])
    with open(os.path.join(args.outdir, "fig2_sample_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["strategy", "seeds", "spend", "reach", "overlap"])
        w.writerows(rows)

    # combined figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 11.8))
    for ax, name in zip(axes.ravel(), STRATEGIES):
        draw_panel(ax, sm, xy, all_seg, name, sets, stats, budget)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False,
               fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=3.0)
    fig.savefig(os.path.join(args.outdir, "fig2_strategies_combined.png"), dpi=args.dpi,
                facecolor="white")
    plt.close(fig)

    # individual panels
    for name in STRATEGIES:
        fig, ax = plt.subplots(figsize=(5.2, 5.8))
        draw_panel(ax, sm, xy, all_seg, name, sets, stats, budget)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"fig2_{LETTERS[name]}_{name.lower()}.png"),
                    dpi=args.dpi, facecolor="white")
        plt.close(fig)

    # legend only
    fig = plt.figure(figsize=(8.5, 0.6))
    fig.legend(handles=legend_handles(), loc="center", ncol=5, frameon=False, fontsize=10.5)
    fig.savefig(os.path.join(args.outdir, "fig2_legend.png"), dpi=args.dpi, facecolor="white")
    plt.close(fig)

    print(f"\nFiles written to '{args.outdir}/'.")


if __name__ == "__main__":
    main()