#!/usr/bin/env python3
"""
make_fig2_synthetic.py - illustration of the four seeding strategies (Figure 2)
on a small GENERATED network designed to be easy to read.

Why a generated network?
------------------------
A sample from the real Twitter network is tangled: circles overlap and their
members are densely interconnected, so the drawing looks messy. Here the
network is built to show the same structure cleanly:

  * four communities (circles) laid out in the four corners, with only a few
    links between them;
  * in each community one large hub account (candidate "mega-influencer"),
    several mid-sized accounts ("micro-influencers") and ordinary users;
  * the mid-sized accounts of a community share a taste group: a contiguous
    group of users who follow them, so seeds taken from one community have
    overlapping audiences, while mid-sized accounts from different communities
    do not;
  * ordinary users also follow a few neighbours inside their community, so
    even ordinary accounts have a handful of followers (as in the real data).

The seeding rules and the budget logic are the same as in the simulation:
the budget equals the price of one mega-influencer (price = followers), and
    Mega   - the largest hub
    Micro  - random mid-sized accounts (from any community)
    Random - random accounts from everyone
    Niche  - mid-sized accounts found by a breadth-first search from one account,
             so they come from the same neighbourhood
Pools are defined by account role here (in the paper: follower percentiles).

Output (in --outdir, default fig2):
    fig2_strategies_combined.png    all four panels + legend (2 x 2)
    fig2_a_mega.png, fig2_b_micro.png, fig2_c_random.png, fig2_d_niche.png
    fig2_legend.png
    fig2_sample_stats.csv           seeds, spend, reach, overlap per panel

IMPORTANT: this is a SCHEMATIC, not a result. The network is invented and the
seeds are one random draw (--search picks the first draw with a clear
contrast). Say so in the caption.

Usage:
    pip install numpy matplotlib
    python make_fig2_synthetic.py
    python make_fig2_synthetic.py --seed 3 --users 40 --mids 6
"""

import argparse
import csv
import os
from collections import deque

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

STRATEGIES = ["Mega", "Micro", "Random", "Niche"]
LETTERS = {"Mega": "a", "Micro": "b", "Random": "c", "Niche": "d"}
SEED_COLOR = "#1f4e9e"
NEUTRAL = "#d9d9d9"
EXPOSURE_COLORS = {1: "#fdd49e", 2: "#fc8d59", 3: "#d7301f"}      # 1, 2, 3+ seeds
ROLE_SIZE = {0: 150.0, 1: 85.0, 2: 38.0}                          # hub, mid, user
ROLE_EDGE = {0: "#7a7a7a", 1: "#9a9a9a", 2: "none"}

CENTRES = np.array([[-1.45, 1.45], [1.45, 1.45], [-1.45, -1.45], [1.45, -1.45]])
NEIGHBOUR_PAIRS = [(0, 1), (2, 3), (0, 2), (1, 3)]


# ----------------------------------------------------------------------------
# Generating the network
# ----------------------------------------------------------------------------
def build_network(rng, users, mids, fan_share, p_fan, p_other, local_follows):
    """Returns a dict. Edge (s -> t): account t follows account s."""
    xy, role, comm = [], [], []
    src, tgt = [], []
    hub_prob = rng.permutation([0.92, 0.70, 0.70, 0.70])     # one clearly largest hub
    groups = []

    for c, centre in enumerate(CENTRES):
        hub = len(xy)
        xy.append(centre.copy()); role.append(0); comm.append(c)

        mid_ids, offset = [], rng.uniform(0, 2 * np.pi)
        for j in range(mids):
            a = offset + 2 * np.pi * j / mids
            mid_ids.append(len(xy))
            xy.append(centre + 0.36 * np.array([np.cos(a), np.sin(a)]))
            role.append(1); comm.append(c)

        n_out = int(round(users * 0.56))
        user_ids, user_ang = [], []
        for n_r, radius, shift in [(n_out, 1.00, 0.0), (users - n_out, 0.70, 0.5)]:
            for k in range(n_r):
                a = 2 * np.pi * (k + shift) / n_r + rng.normal(0, 0.03)
                r = radius + rng.normal(0, 0.02)
                user_ids.append(len(xy)); user_ang.append(a % (2 * np.pi))
                xy.append(centre + r * np.array([np.cos(a), np.sin(a)]))
                role.append(2); comm.append(c)

        # taste group: users on one contiguous arc follow the mid-sized accounts
        ang = np.array(user_ang)
        arc_centre = rng.uniform(0, 2 * np.pi)
        fan = np.abs((ang - arc_centre + np.pi) % (2 * np.pi) - np.pi) <= fan_share * np.pi
        order = np.argsort(ang)
        sorted_users = [user_ids[i] for i in order]
        rank = {u: i for i, u in enumerate(sorted_users)}

        for idx, u in enumerate(user_ids):
            if rng.random() < hub_prob[c]:
                src.append(hub); tgt.append(u)
            p = p_fan if fan[idx] else p_other
            for m in mid_ids:
                if rng.random() < p:
                    src.append(m); tgt.append(u)
            i = rank[u]
            near = [sorted_users[(i + d) % len(sorted_users)] for d in (-3, -2, -1, 1, 2, 3)]
            for v in rng.choice(near, size=local_follows, replace=False):
                src.append(int(v)); tgt.append(u)
        groups.append({"hub": hub, "mids": mid_ids, "users": user_ids})

    xy = np.array(xy)
    # a few links between neighbouring communities (kept to a minimum)
    for a_c, b_c in NEIGHBOUR_PAIRS:
        d = CENTRES[b_c] - CENTRES[a_c]
        d = d / np.linalg.norm(d)
        ua = sorted(groups[a_c]["users"], key=lambda u: -np.dot(xy[u] - CENTRES[a_c], d))[:2]
        ub = sorted(groups[b_c]["users"], key=lambda u: np.dot(xy[u] - CENTRES[b_c], d))[:2]
        src.append(ub[0]); tgt.append(ua[0])
        src.append(ua[1]); tgt.append(ub[1])

    s, t = np.array(src), np.array(tgt)
    m = len(xy)
    # undirected adjacency (CSR) for the breadth-first search
    a = np.concatenate([s, t]); b = np.concatenate([t, s])
    o = np.argsort(a, kind="stable")
    ptr = np.concatenate([[0], np.cumsum(np.bincount(a, minlength=m))])
    return {"m": m, "xy": xy, "role": np.array(role), "comm": np.array(comm),
            "s": s, "t": t, "followers": np.bincount(s, minlength=m),
            "ptr": ptr, "nbrs": b[o]}


# ----------------------------------------------------------------------------
# Seeding rules
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


def draw_niche(net, pool, cost, budget, rng, tol):
    in_pool = np.zeros(net["m"], dtype=bool)
    in_pool[pool] = True
    visited = np.zeros(net["m"], dtype=bool)
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
            nb = net["nbrs"][net["ptr"][v]:net["ptr"][v + 1]]
            fresh = np.unique(nb[~visited[nb]])
            if len(fresh):
                visited[fresh] = True
                queue.extend(rng.permutation(fresh).tolist())
    return np.array(selected, dtype=np.int64), spent


def pick_seed_sets(net, rng, tol):
    f = net["followers"]
    cost = np.maximum(f, 1.0)
    hubs = np.where(net["role"] == 0)[0]
    mega = int(hubs[np.argmax(f[hubs])])
    budget = float(cost[mega])
    micro_pool = np.where(net["role"] == 1)[0]
    all_pool = np.array([i for i in range(net["m"]) if i != mega])   # do not re-draw the mega

    sets = {"Mega": (np.array([mega]), budget)}
    sets["Micro"] = draw_budget_matched(micro_pool, cost, budget, rng, tol)
    sets["Random"] = draw_budget_matched(all_pool, cost, budget, rng, tol)
    sets["Niche"] = draw_niche(net, micro_pool, cost, budget, rng, tol)
    return sets, budget


def exposure_stats(net, seeds):
    is_seed = np.zeros(net["m"], dtype=bool)
    is_seed[seeds] = True
    mask = is_seed[net["s"]]
    expo = np.bincount(net["t"][mask], minlength=net["m"])
    reach = int((expo > 0).sum())
    overlap = float(expo.sum() / reach) if reach else 1.0
    return {"expo": expo, "mask": mask, "reach": reach, "overlap": overlap}


def evaluate(sets, stats, budget):
    n = {k: len(sets[k][0]) for k in sets}
    ov = {k: stats[k]["overlap"] for k in sets}
    ratios = [sets[k][1] / budget for k in sets]
    other = max(ov["Micro"], ov["Random"])
    ok = (budget >= 20 and 4 <= n["Micro"] <= 9 and 4 <= n["Niche"] <= 9
          and 6 <= n["Random"] <= 16 and min(ratios) >= 0.85 and max(ratios) <= 1.15
          and ov["Niche"] >= max(1.8, 1.5 * other))
    return ok, ov["Niche"] - other


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def draw_panel(ax, net, bg_segments, name, sets, stats, budget, titled=True):
    seeds, spent = sets[name]
    st = stats[name]
    xy, expo, mask = net["xy"], st["expo"], st["mask"]

    ax.add_collection(LineCollection(bg_segments, colors="#9a9a9a", linewidths=0.3,
                                     alpha=0.20, zorder=1))
    seg = np.stack([xy[net["s"][mask]], xy[net["t"][mask]]], axis=1)
    ax.add_collection(LineCollection(seg, colors="#444444", linewidths=0.6,
                                     alpha=0.45, zorder=2))

    colors = np.array([NEUTRAL] * net["m"], dtype=object)
    for k, col in EXPOSURE_COLORS.items():
        sel = (expo == k) if k < 3 else (expo >= 3)
        colors[sel] = col
    sizes = np.array([ROLE_SIZE[r] for r in net["role"]])
    edges = [ROLE_EDGE[r] for r in net["role"]]
    ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=list(colors), edgecolors=edges,
               linewidths=0.7, zorder=3)

    seed_sizes = 90 + 30 * np.sqrt(net["followers"][seeds])
    ax.scatter(xy[seeds, 0], xy[seeds, 1], s=seed_sizes, c=SEED_COLOR,
               edgecolors="black", linewidths=0.9, zorder=5)

    ax.set_xlim(-2.75, 2.75)
    ax.set_ylim(-2.75, 2.75)
    ax.set_aspect("equal")
    ax.axis("off")
    if titled:
        ax.set_title(f"({LETTERS[name]}) {name}", fontsize=13, fontweight="bold")
    label = (f"{len(seeds)} seed{'s' if len(seeds) != 1 else ''} · spend {spent:.0f}"
             f" of {budget:.0f}\nreach {st['reach']} people · overlap {st['overlap']:.2f}")
    ax.text(0.5, -0.01, label, transform=ax.transAxes, ha="center", va="top", fontsize=9.5)


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
    ap.add_argument("--seed", type=int, default=1, help="random seed (default 1)")
    ap.add_argument("--users", type=int, default=36, help="ordinary users per community")
    ap.add_argument("--mids", type=int, default=5, help="mid-sized accounts per community")
    ap.add_argument("--fan-share", type=float, default=0.40,
                    help="share of a community in the taste group (default 0.40)")
    ap.add_argument("--p-fan", type=float, default=0.40,
                    help="chance a taste-group user follows a mid-sized account (default 0.40)")
    ap.add_argument("--p-other", type=float, default=0.02,
                    help="same for other users (default 0.02)")
    ap.add_argument("--local-follows", type=int, default=3,
                    help="neighbours each user follows (default 3)")
    ap.add_argument("--tol", type=float, default=0.10, help="budget tolerance (default 0.10)")
    ap.add_argument("--search", type=int, default=200,
                    help="draws to try for a clear illustration (default 200)")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--outdir", default="fig2")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    best = None
    for trial in range(args.search):
        rng = np.random.default_rng(args.seed + trial)
        net = build_network(rng, args.users, args.mids, args.fan_share, args.p_fan,
                            args.p_other, args.local_follows)
        sets, budget = pick_seed_sets(net, rng, args.tol)
        stats = {k: exposure_stats(net, sets[k][0]) for k in sets}
        ok, score = evaluate(sets, stats, budget)
        if best is None or score > best["score"] or ok:
            best = {"score": score, "trial": trial, "net": net, "sets": sets,
                    "stats": stats, "budget": budget, "ok": ok}
        if ok:
            break
    if not best["ok"]:
        print("Warning: no draw met all illustration criteria; using the best one found. "
              "Try another --seed or a larger --search.")
    net, sets, stats, budget = best["net"], best["sets"], best["stats"], best["budget"]
    print(f"Using draw #{best['trial']} (seed {args.seed + best['trial']}): "
          f"{net['m']} accounts, {len(net['s'])} follow links, budget {budget:.0f}.")

    all_seg = np.stack([net["xy"][net["s"]], net["xy"][net["t"]]], axis=1)

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

    fig, axes = plt.subplots(2, 2, figsize=(11, 11.8))
    for ax, name in zip(axes.ravel(), STRATEGIES):
        draw_panel(ax, net, all_seg, name, sets, stats, budget)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=5, frameon=False,
               fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1), h_pad=3.0)
    fig.savefig(os.path.join(args.outdir, "fig2_strategies_combined.png"), dpi=args.dpi,
                facecolor="white")
    plt.close(fig)

    for name in STRATEGIES:
        fig, ax = plt.subplots(figsize=(5.2, 5.8))
        draw_panel(ax, net, all_seg, name, sets, stats, budget)
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, f"fig2_{LETTERS[name]}_{name.lower()}.png"),
                    dpi=args.dpi, facecolor="white")
        plt.close(fig)

    fig = plt.figure(figsize=(8.5, 0.6))
    fig.legend(handles=legend_handles(), loc="center", ncol=5, frameon=False, fontsize=10.5)
    fig.savefig(os.path.join(args.outdir, "fig2_legend.png"), dpi=args.dpi, facecolor="white")
    plt.close(fig)

    print(f"\nFiles written to '{args.outdir}/'.")


if __name__ == "__main__":
    main()