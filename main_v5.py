"""
Network Cellular Automaton for influencer seeding.  VERSION 5  (partial rewiring)

Same model as versions 3 and 4. This script asks HOW MUCH network clustering is
needed for cascades (and for the advantage of the Niche strategy) by randomising
only a fraction of the edges of the Twitter network:

  rewire10 : 10% of edges randomised
  rewire25 : 25% of edges randomised
  rewire50 : 50% of edges randomised

Randomising means: the targets of the selected edges are shuffled among those
edges, which keeps EVERY account's number of followers and number of followed
accounts exactly unchanged but breaks clustering. Because follower counts are
unchanged, Mega / Micro / Random seeds and thresholds are identical to
version 3 (paired comparison). The 0% case is version 3 itself and the 100%
case is the "rewired" variant of version 4.
"""

import os
import csv
import time
import urllib.request
from collections import deque

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SNAP_URL = "https://snap.stanford.edu/data/twitter_combined.txt.gz"
RAW_GZ = "twitter_combined.txt.gz"
CACHE_PT = "twitter_graph.pt"
FIRST_COLUMN_FOLLOWS_SECOND = True

BASE_SEED = 12345
REWIRE_SEED = BASE_SEED + 777

THETA_STD = 0.10
THETA_MIN, THETA_MAX = 0.05, 0.95

MEGA_PCT = 0.99
MICRO_PCT_LOW, MICRO_PCT_HIGH = 0.80, 0.95
MEGA_PICK = "random"

BETA = 1.0
BUDGET_TOLERANCE = 0.02
MAX_TICKS = 500
CASCADE_FRACTION = 0.10

VARIANTS = [
    {"name": "rewire10", "rewire": 0.10, "gamma": 1.0},
    {"name": "rewire25", "rewire": 0.25, "gamma": 1.0},
    {"name": "rewire50", "rewire": 0.50, "gamma": 1.0},
]


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------
def load_or_process_graph(cache_pt: str = CACHE_PT):
    """Returns (src, tgt, num_nodes) as CPU tensors; influence flows src -> tgt."""
    if os.path.exists(cache_pt):
        print(f"Loading cached graph from '{cache_pt}'...")
        d = torch.load(cache_pt)
        return d["src"], d["tgt"], int(d["num_nodes"])

    if not os.path.exists(RAW_GZ):
        print("Downloading ego-Twitter from SNAP...")
        urllib.request.urlretrieve(SNAP_URL, RAW_GZ)

    print("Parsing edge list (one-off, takes a few seconds)...")
    edges = np.loadtxt(RAW_GZ, dtype=np.int64)
    edges = edges[edges[:, 0] != edges[:, 1]]
    edges = np.unique(edges, axis=0)

    unique_ids, inverse = np.unique(edges, return_inverse=True)
    inverse = inverse.reshape(edges.shape)
    num_nodes = len(unique_ids)

    follower, followee = (
        (inverse[:, 0], inverse[:, 1])
        if FIRST_COLUMN_FOLLOWS_SECOND
        else (inverse[:, 1], inverse[:, 0])
    )
    src = torch.from_numpy(followee.copy())
    tgt = torch.from_numpy(follower.copy())

    torch.save({"src": src, "tgt": tgt, "num_nodes": num_nodes}, cache_pt)
    print(f"Saved cache to '{cache_pt}'.")
    return src, tgt, num_nodes


class Graph:
    """Graph on the compute device plus CPU adjacency structures for seeding."""

    def __init__(self, rewire: float = 0.0):
        src, tgt, n = load_or_process_graph()
        self.num_nodes = n
        src_np, tgt_np = src.numpy().copy(), tgt.numpy().copy()

        if rewire > 0:
            rng = np.random.default_rng(REWIRE_SEED)
            m = len(tgt_np)
            idx = rng.choice(m, size=int(round(rewire * m)), replace=False)
            tgt_np[idx] = tgt_np[idx][rng.permutation(len(idx))]   # keeps all degrees
            n_self = int((src_np == tgt_np).sum())
            keys = src_np.astype(np.int64) * n + tgt_np
            n_dup = len(keys) - len(np.unique(keys))
            print(f"Rewired network: {n_self} self-loops and {n_dup} duplicate edges "
                  f"left in place (negligible relative to {len(keys):,} edges).")

        src = torch.from_numpy(src_np)
        tgt = torch.from_numpy(tgt_np)

        self.src = src.to(DEVICE)
        self.tgt = tgt.to(DEVICE)
        self.followers = np.bincount(src_np, minlength=n)
        self.followees_t = torch.bincount(tgt, minlength=n).to(DEVICE)
        self.denom = self.followees_t.clamp(min=1).float()

        order = np.argsort(src_np, kind="stable")
        self.fol_nbrs = tgt_np[order]
        self.fol_ptr = np.concatenate([[0], np.cumsum(np.bincount(src_np, minlength=n))])

        a = np.concatenate([src_np, tgt_np])
        b = np.concatenate([tgt_np, src_np])
        order = np.argsort(a, kind="stable")
        self.und_nbrs = b[order]
        self.und_ptr = np.concatenate([[0], np.cumsum(np.bincount(a, minlength=n))])

        print(f"Graph: {n:,} nodes, {len(src_np):,} directed edges "
              f"({rewire:.0%} of edges rewired).")


# ----------------------------------------------------------------------------
# Pools, costs, seed selection
# ----------------------------------------------------------------------------
def influencer_cost(followers, gamma: float, beta: float = BETA):
    """Price = beta * max(followers, 1) ** gamma (scalars and arrays)."""
    return beta * np.maximum(followers, 1) ** gamma


def build_pools(followers: np.ndarray):
    p_mega = np.quantile(followers, MEGA_PCT)
    p_lo = np.quantile(followers, MICRO_PCT_LOW)
    p_hi = np.quantile(followers, MICRO_PCT_HIGH)
    mega_pool = np.where(followers >= p_mega)[0]
    micro_pool = np.where((followers >= p_lo) & (followers <= p_hi))[0]
    all_pool = np.arange(len(followers))
    return mega_pool, micro_pool, all_pool


def pick_mega(followers, mega_pool, rng, k, gamma):
    if MEGA_PICK == "top":
        nodes = mega_pool[np.argsort(followers[mega_pool])[::-1][:k]]
    else:
        nodes = rng.choice(mega_pool, size=k, replace=False)
    return (np.asarray(nodes, dtype=np.int64),
            float(influencer_cost(followers[nodes], gamma).sum()))


def draw_budget_matched(pool, costs, budget, rng, tol=BUDGET_TOLERANCE):
    selected, spent = [], 0.0
    upper, lower = budget * (1 + tol), budget * (1 - tol)
    for node in rng.permutation(pool):
        c = costs[node]
        if spent + c > upper:
            continue
        selected.append(node)
        spent += c
        if spent >= lower:
            break
    return np.array(selected, dtype=np.int64), spent


def draw_niche(g: Graph, micro_pool, costs, budget, rng, tol=BUDGET_TOLERANCE,
               max_restarts=500):
    """Micro accounts from one network neighbourhood (breadth-first search)."""
    n = g.num_nodes
    in_pool = np.zeros(n, dtype=bool)
    in_pool[micro_pool] = True
    visited = np.zeros(n, dtype=bool)
    upper, lower = budget * (1 + tol), budget * (1 - tol)
    selected, spent = [], 0.0

    for start in rng.permutation(micro_pool)[:max_restarts]:
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([int(start)])
        while queue:
            v = queue.popleft()
            if in_pool[v]:
                c = costs[v]
                if spent + c <= upper:
                    selected.append(v)
                    spent += c
                    if spent >= lower:
                        return np.array(selected, dtype=np.int64), spent
            nbrs = g.und_nbrs[g.und_ptr[v]:g.und_ptr[v + 1]]
            fresh = nbrs[~visited[nbrs]]
            if len(fresh):
                visited[fresh] = True
                fresh = rng.permutation(fresh)
                queue.extend(fresh.tolist())
    return np.array(selected, dtype=np.int64), spent


def audience_stats(g: Graph, seeds):
    parts = [g.fol_nbrs[g.fol_ptr[s]:g.fol_ptr[s + 1]] for s in seeds]
    total = sum(len(p) for p in parts)
    if total == 0:
        return 0, 1.0
    unique = len(np.unique(np.concatenate(parts)))
    return unique, total / unique


def select_all_strategies(g, pools, rng, niche_rng, k, gamma):
    mega_pool, micro_pool, all_pool = pools
    costs = influencer_cost(g.followers, gamma)

    mega_seeds, budget = pick_mega(g.followers, mega_pool, rng, k, gamma)
    micro_seeds, micro_spent = draw_budget_matched(micro_pool, costs, budget, rng)
    rand_seeds, rand_spent = draw_budget_matched(all_pool, costs, budget, rng)
    niche_seeds, niche_spent = draw_niche(g, micro_pool, costs, budget, niche_rng)

    sets = {
        "Mega": (mega_seeds, budget),
        "Micro": (micro_seeds, micro_spent),
        "Niche": (niche_seeds, niche_spent),
        "Random": (rand_seeds, rand_spent),
    }
    stats = {name: audience_stats(g, seeds) for name, (seeds, _) in sets.items()}
    return sets, budget, stats


# ----------------------------------------------------------------------------
# Thresholds
# ----------------------------------------------------------------------------
def thresholds_from_uniforms(u: torch.Tensor, mean: float) -> torch.Tensor:
    a = torch.tensor((THETA_MIN - mean) / THETA_STD, dtype=torch.float64, device=u.device)
    b = torch.tensor((THETA_MAX - mean) / THETA_STD, dtype=torch.float64, device=u.device)
    cdf_a, cdf_b = torch.special.ndtr(a), torch.special.ndtr(b)
    p = (cdf_a + u * (cdf_b - cdf_a)).clamp(1e-12, 1 - 1e-12)
    theta = mean + THETA_STD * torch.special.ndtri(p)
    return theta.clamp(THETA_MIN, THETA_MAX).float()


# ----------------------------------------------------------------------------
# The cellular automaton
# ----------------------------------------------------------------------------
def run_simulation(g: Graph, seeds: np.ndarray, alpha: float, thresholds: torch.Tensor):
    n = g.num_nodes
    states = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    states[torch.as_tensor(seeds, device=DEVICE)] = True
    exposures = torch.zeros(n, device=DEVICE)
    history = [int(states.sum())]

    for _ in range(MAX_TICKS):
        active_cnt = torch.zeros(n, device=DEVICE)
        active_cnt.index_add_(0, g.tgt, states[g.src].float())

        ratio = active_cnt / g.denom
        fatigue = torch.exp(-alpha * exposures)
        candidates = ~states
        exposed_now = candidates & (active_cnt > 0)

        adopt = exposed_now & (ratio * fatigue >= thresholds)
        exposures += exposed_now.float()

        if not adopt.any():
            break
        states |= adopt
        history.append(int(states.sum()))

    return history, int((exposures > 0).sum())


def ticks_to_fraction(history, n, fraction=CASCADE_FRACTION):
    limit = fraction * n
    for tick, h in enumerate(history):
        if h >= limit:
            return tick
    return "NA"


# ----------------------------------------------------------------------------
# Robustness sweep
# ----------------------------------------------------------------------------
def execute_robustness(
    output_csv: str = "simulation_results_v5_rewiring.csv",
    replications: int = 30,
    strategies=("Mega", "Micro", "Niche", "Random"),
    alphas=(0.0,),
    mean_thetas=(0.10, 0.15, 0.20, 0.25),
    n_megas_list=(1, 3, 10),
    variants=VARIANTS,
):
    print(f"Using compute device: {DEVICE}")
    per_variant = (len(strategies) * len(alphas) * len(mean_thetas)
                   * len(n_megas_list) * replications)
    print(f"{len(variants)} variants x {per_variant:,} runs = "
          f"{len(variants) * per_variant:,} runs in total.")

    headers = [
        "variant", "run_id", "replication", "strategy", "n_megas", "fatigue_alpha",
        "mean_threshold_nominal", "mean_threshold_realized",
        "seed_count", "target_budget", "achieved_budget",
        "audience_reach", "audience_overlap",
        "convergence_tick", "ticks_to_10pct", "exposed_nodes", "final_adopters",
        "market_saturation", "ross",
    ]

    start, run_id = time.time(), 0
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for v_idx, variant in enumerate(variants, start=1):
            name, gamma = variant["name"], variant["gamma"]
            print(f"\n=== Variant {v_idx}/{len(variants)}: {name} "
                  f"(rewire={variant['rewire']}, gamma={gamma}) ===")
            g = Graph(rewire=variant["rewire"])
            pools = build_pools(g.followers)

            for rep in range(1, replications + 1):
                rng = np.random.default_rng(BASE_SEED + rep)
                niche_rng = np.random.default_rng(BASE_SEED + 1_000_000 + rep)
                gen = torch.Generator(device=DEVICE).manual_seed(BASE_SEED + rep)

                seed_data = {
                    k: select_all_strategies(g, pools, rng, niche_rng, k, gamma)
                    for k in n_megas_list
                }
                u = torch.rand(g.num_nodes, generator=gen, device=DEVICE,
                               dtype=torch.float64)

                for mean_theta in mean_thetas:
                    thresholds = thresholds_from_uniforms(u, mean_theta)
                    realized_theta = thresholds.mean().item()

                    for k in n_megas_list:
                        seed_sets, target_budget, stats = seed_data[k]

                        for alpha in alphas:
                            for strategy in strategies:
                                run_id += 1
                                seeds, budget = seed_sets[strategy]
                                reach, overlap = stats[strategy]
                                history, exposed = run_simulation(
                                    g, seeds, alpha, thresholds)

                                final_adopters = history[-1]
                                seed_count = len(seeds)
                                ross = ((final_adopters - seed_count) / budget
                                        if budget > 0 else 0.0)

                                writer.writerow([
                                    name, run_id, rep, strategy, k, alpha,
                                    mean_theta, round(realized_theta, 4),
                                    seed_count, round(target_budget, 2),
                                    round(budget, 2), reach, round(overlap, 4),
                                    len(history) - 1,
                                    ticks_to_fraction(history, g.num_nodes),
                                    exposed, final_adopters,
                                    round(final_adopters / g.num_nodes, 5),
                                    round(ross, 5),
                                ])

                f.flush()
                print(f"[{name}] replication {rep}/{replications} | "
                      f"elapsed {time.time() - start:.1f}s")

            del g
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\nDone. Results written to '{output_csv}' ({run_id} rows).")


if __name__ == "__main__":
    execute_robustness()