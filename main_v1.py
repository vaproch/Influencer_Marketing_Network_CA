"""
Network Cellular Automaton for influencer seeding (mega vs micro vs random).

Model summary
-------------
* Network: SNAP ego-Twitter (directed). An edge "u v" means u FOLLOWS v, so
  influence flows v -> u. A node's "followers" (out-degree in influence terms)
  determines its price as an influencer; the accounts a node follows determine
  who can influence it.
* State: each user is either an adopter (True) or not. Adoption is permanent.
* Update rule (synchronous): a non-adopter i adopts at tick t if

        R_i(t) * phi_i(t) >= theta_i

  where
    R_i(t)   = (# followed accounts that are adopters) / (# followed accounts)
    phi_i(t) = exp(-alpha * E_i(t))   <- AD FATIGUE
    E_i(t)   = number of earlier ticks in which i was exposed to the campaign
               (had >= 1 adopter among followed accounts) but did not adopt.
    theta_i  = individual adoption threshold (product "friction").
* Fatigue only ever decreases phi, so if a tick produces no new adopters the
  process can never restart -> stopping there is exact convergence.
* Seeding: budget B is the price of ONE mega influencer. Micro and Random
  strategies spend (approximately) the same B on random accounts from their
  pools. Price = beta * max(followers, 1) ** gamma.
* Common random numbers: within one replication, seeds and the underlying
  uniform draws for thresholds are shared across alpha and theta conditions,
  and thresholds are shared across strategies -> paired comparisons.
"""

import os
import csv
import time
import urllib.request

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Global configuration
# ----------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SNAP_URL = "https://snap.stanford.edu/data/twitter_combined.txt.gz"
RAW_GZ = "twitter_combined.txt.gz"
CACHE_PT = "twitter_graph.pt"

# True  -> line "u v" means u follows v (influence flows v -> u)
# False -> line "u v" means v follows u (influence flows u -> v)
FIRST_COLUMN_FOLLOWS_SECOND = True

BASE_SEED = 12345

# Threshold distribution: truncated normal on [THETA_MIN, THETA_MAX]
THETA_STD = 0.10
THETA_MIN, THETA_MAX = 0.05, 0.95

# Pools (percentiles of follower count over all nodes)
MEGA_PCT = 0.99                  # mega: followers >= p99
MICRO_PCT_LOW, MICRO_PCT_HIGH = 0.80, 0.95   # micro: p80 <= followers <= p95
MEGA_PICK = "random"             # "random" mega from the pool, or "top" (max followers)

# Cost model
BETA, GAMMA = 1.0, 1.0
BUDGET_TOLERANCE = 0.02          # micro/random stop within +-2% of the budget

MAX_TICKS = 500


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
    edges = np.loadtxt(RAW_GZ, dtype=np.int64)           # handles .gz transparently
    edges = edges[edges[:, 0] != edges[:, 1]]             # drop self-loops
    edges = np.unique(edges, axis=0)                      # drop duplicate edges

    # Remap arbitrary node IDs to 0..N-1
    unique_ids, inverse = np.unique(edges, return_inverse=True)
    inverse = inverse.reshape(edges.shape)
    num_nodes = len(unique_ids)

    follower, followee = (
        (inverse[:, 0], inverse[:, 1])
        if FIRST_COLUMN_FOLLOWS_SECOND
        else (inverse[:, 1], inverse[:, 0])
    )
    src = torch.from_numpy(followee.copy())   # influencer
    tgt = torch.from_numpy(follower.copy())   # follower (the one being influenced)

    torch.save({"src": src, "tgt": tgt, "num_nodes": num_nodes}, cache_pt)
    print(f"Saved cache to '{cache_pt}'.")
    return src, tgt, num_nodes


class Graph:
    """Holds the graph on the compute device plus precomputed degree vectors."""

    def __init__(self):
        src, tgt, n = load_or_process_graph()
        self.num_nodes = n
        self.src = src.to(DEVICE)
        self.tgt = tgt.to(DEVICE)
        # followers[j] = how many accounts follow j  (reach / price driver)
        self.followers = torch.bincount(src, minlength=n).numpy()
        # followees_t[i] = how many accounts i follows (denominator of R_i)
        self.followees_t = torch.bincount(tgt, minlength=n).to(DEVICE)
        self.denom = self.followees_t.clamp(min=1).float()

        print(f"Graph: {n:,} nodes, {len(src):,} directed edges.")
        print(f"  max followers: {self.followers.max():,} | "
              f"max accounts followed: {int(self.followees_t.max()):,}")
        print(f"  nodes that follow nobody (cannot be influenced): "
              f"{int((self.followees_t == 0).sum()):,}")


# ----------------------------------------------------------------------------
# Pools, costs, seed selection
# ----------------------------------------------------------------------------
def influencer_cost(followers, beta: float = BETA, gamma: float = GAMMA):
    """Price = beta * max(followers, 1) ** gamma (works on scalars and arrays)."""
    return beta * np.maximum(followers, 1) ** gamma


def build_pools(followers: np.ndarray):
    p_mega = np.quantile(followers, MEGA_PCT)
    p_lo = np.quantile(followers, MICRO_PCT_LOW)
    p_hi = np.quantile(followers, MICRO_PCT_HIGH)
    mega_pool = np.where(followers >= p_mega)[0]
    micro_pool = np.where((followers >= p_lo) & (followers <= p_hi))[0]
    all_pool = np.arange(len(followers))   # truly random: everyone

    print("Pools (by follower count):")
    print(f"  Mega  : {len(mega_pool):>6,} nodes, followers >= {p_mega:.0f}")
    print(f"  Micro : {len(micro_pool):>6,} nodes, {p_lo:.0f} <= followers <= {p_hi:.0f}")
    print(f"  Random: {len(all_pool):>6,} nodes (all users)")
    return mega_pool, micro_pool, all_pool


def pick_mega(followers, mega_pool, rng):
    if MEGA_PICK == "top":
        node = mega_pool[np.argmax(followers[mega_pool])]
    else:
        node = rng.choice(mega_pool)
    return np.array([node]), float(influencer_cost(followers[node]))


def draw_budget_matched(pool, costs, budget, rng, tol=BUDGET_TOLERANCE):
    """Random accounts from `pool` until spend is within `tol` of `budget`.

    Accounts that would push spend above budget*(1+tol) are skipped, so the
    achieved spend never overshoots meaningfully.
    """
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


def select_all_strategies(followers, pools, rng):
    """One budget-matched seed set per strategy for one replication."""
    mega_pool, micro_pool, all_pool = pools
    costs = influencer_cost(followers)

    mega_seeds, budget = pick_mega(followers, mega_pool, rng)
    micro_seeds, micro_spent = draw_budget_matched(micro_pool, costs, budget, rng)
    rand_seeds, rand_spent = draw_budget_matched(all_pool, costs, budget, rng)

    return {
        "Mega": (mega_seeds, budget),
        "Micro": (micro_seeds, micro_spent),
        "Random": (rand_seeds, rand_spent),
    }, budget


# ----------------------------------------------------------------------------
# Thresholds (truncated normal via inverse CDF, so uniforms can be reused)
# ----------------------------------------------------------------------------
def thresholds_from_uniforms(u: torch.Tensor, mean: float) -> torch.Tensor:
    """Truncated N(mean, THETA_STD) on [THETA_MIN, THETA_MAX], no clamping spike."""
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
    """Synchronous threshold CA with exposure-based ad fatigue.

    Returns (adoption_history, number_of_nodes_ever_exposed).
    """
    n = g.num_nodes
    states = torch.zeros(n, dtype=torch.bool, device=DEVICE)
    states[torch.as_tensor(seeds, device=DEVICE)] = True
    exposures = torch.zeros(n, device=DEVICE)       # E_i: ticks exposed so far
    history = [int(states.sum())]

    for _ in range(MAX_TICKS):
        # Number of adopted accounts each node follows
        active_cnt = torch.zeros(n, device=DEVICE)
        active_cnt.index_add_(0, g.tgt, states[g.src].float())

        ratio = active_cnt / g.denom                       # R_i
        fatigue = torch.exp(-alpha * exposures)            # phi_i
        candidates = ~states
        exposed_now = candidates & (active_cnt > 0)

        adopt = exposed_now & (ratio * fatigue >= thresholds)
        exposures += exposed_now.float()                   # update AFTER deciding

        if not adopt.any():
            break    # fatigue only lowers phi, so no future adoption is possible
        states |= adopt
        history.append(int(states.sum()))

    return history, int((exposures > 0).sum())


# ----------------------------------------------------------------------------
# Factorial Monte Carlo sweep
# ----------------------------------------------------------------------------
def execute_factorial_sweep(
    output_csv: str = "simulation_results.csv",
    replications: int = 30,
    strategies=("Mega", "Micro", "Random"),
    alphas=(0.0, 0.05, 0.10, 0.20, 0.40),
    mean_thetas=(0.10, 0.20, 0.30, 0.40),
):
    print(f"Using compute device: {DEVICE}")
    g = Graph()
    pools = build_pools(g.followers)

    total = len(strategies) * len(alphas) * len(mean_thetas) * replications
    print(f"Starting sweep: {total:,} runs ({replications} replications).")

    headers = [
        "run_id", "replication", "strategy", "fatigue_alpha",
        "mean_threshold_nominal", "mean_threshold_realized",
        "seed_count", "target_budget", "achieved_budget",
        "convergence_tick", "exposed_nodes", "final_adopters",
        "market_saturation", "ross",
    ]

    start, run_id = time.time(), 0
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        for rep in range(1, replications + 1):
            rng = np.random.default_rng(BASE_SEED + rep)
            gen = torch.Generator(device=DEVICE).manual_seed(BASE_SEED + rep)

            # Seeds: drawn once per replication, shared by all alpha/theta cells
            seed_sets, target_budget = select_all_strategies(g.followers, pools, rng)
            # Uniforms: drawn once per replication, shared by everything
            u = torch.rand(g.num_nodes, generator=gen, device=DEVICE, dtype=torch.float64)

            for mean_theta in mean_thetas:
                thresholds = thresholds_from_uniforms(u, mean_theta)
                realized_theta = thresholds.mean().item()

                for alpha in alphas:
                    for strategy in strategies:
                        run_id += 1
                        seeds, budget = seed_sets[strategy]
                        history, exposed = run_simulation(g, seeds, alpha, thresholds)

                        final_adopters = history[-1]
                        seed_count = len(seeds)
                        ross = (final_adopters - seed_count) / budget if budget > 0 else 0.0

                        writer.writerow([
                            run_id, rep, strategy, alpha,
                            mean_theta, round(realized_theta, 4),
                            seed_count, round(target_budget, 2), round(budget, 2),
                            len(history) - 1, exposed, final_adopters,
                            round(final_adopters / g.num_nodes, 5), round(ross, 5),
                        ])

            f.flush()
            print(f"Replication {rep}/{replications} done | "
                  f"elapsed {time.time() - start:.1f}s")

    print(f"\nDone. Results written to '{output_csv}' ({run_id} rows).")


if __name__ == "__main__":
    execute_factorial_sweep(output_csv="simulation_results.csv", replications=30)