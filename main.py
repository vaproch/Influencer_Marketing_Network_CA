import os
import gzip
import shutil
import urllib.request
import torch
import networkx as nx
from torch_geometric.utils import from_networkx


def get_snap_edge_index(
        url: str = "https://snap.stanford.edu/data/facebook_combined.txt.gz",
        local_txt: str = "facebook_combined.txt"
) -> tuple[torch.Tensor, int]:

    # Download and extract only if local file does not exist
    if not os.path.exists(local_txt):
        print(f"File '{local_txt}' not found locally. Downloading from SNAP...")
        gz_filename = local_txt + ".gz"
        urllib.request.urlretrieve(url, gz_filename)

        with gzip.open(gz_filename, 'rb') as f_in:
            with open(local_txt, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

        # Clean up temporary archive
        if os.path.exists(gz_filename):
            os.remove(gz_filename)
        print("Download complete.")
    else:
        print(f"Loading dataset '{local_txt}' from local storage...")

    # Parse edgelist into NetworkX
    G = nx.read_edgelist(local_txt, create_using=nx.Graph(), nodetype=int)
    num_nodes = G.number_of_nodes()
    print(f"Graph ready: {num_nodes:,} nodes, {G.number_of_edges():,} edges.")

    # Convert to PyG Tensor and move to GPU if available
    pyg_data = from_networkx(G)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    edge_index = pyg_data.edge_index.to(device)

    return edge_index, num_nodes


# Usage
edge_index, num_nodes = get_snap_edge_index()

def load_or_process_graph(txt_filename: str = "facebook_combined.txt", pt_filename: str = "facebook_tensor.pt"):
    # 1. Load directly from PyTorch binary if it exists
    if os.path.exists(pt_filename):
        print(f"Fast loading tensor from '{pt_filename}'...")
        edge_index, num_nodes = torch.load(pt_filename)
    else:
        # 2. Parse text file and save binary version for future runs
        print(f"Processing '{txt_filename}' into PyTorch tensor...")
        G = nx.read_edgelist(txt_filename, create_using=nx.Graph(), nodetype=int)
        num_nodes = G.number_of_nodes()
        pyg_data = from_networkx(G)
        edge_index = pyg_data.edge_index

        torch.save((edge_index, num_nodes), pt_filename)
        print(f"Saved binary cache to '{pt_filename}'.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return edge_index.to(device), num_nodes

import os
import csv
import time
import torch

# Define device globally
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_graph_data(pt_filename: str = "facebook_tensor.pt"):
    """Loads preprocessed tensor graph and computes node degrees."""
    if not os.path.exists(pt_filename):
        raise FileNotFoundError(
            f"'{pt_filename}' not found. Please run the preprocessing script first."
        )

    edge_index, num_nodes = torch.load(pt_filename)
    edge_index = edge_index.to(DEVICE)
    src_nodes, tgt_nodes = edge_index[0], edge_index[1]

    # Calculate node degrees on GPU
    node_degrees = torch.zeros(num_nodes, device=DEVICE)
    node_degrees.scatter_add_(
        0, src_nodes, torch.ones_like(src_nodes, dtype=torch.float)
    )

    return edge_index, num_nodes, node_degrees, src_nodes, tgt_nodes


def stratify_node_pools(node_degrees: torch.Tensor):
    """Categorizes nodes into Mega, Micro, and Consumer pools based on degree percentiles."""
    p80 = torch.quantile(node_degrees, 0.80)
    p95 = torch.quantile(node_degrees, 0.95)
    p99 = torch.quantile(node_degrees, 0.99)

    mega_pool = torch.where(node_degrees >= p99)[0]
    micro_pool = torch.where((node_degrees >= p80) & (node_degrees <= p95))[0]
    consumer_pool = torch.where(node_degrees < p80)[0]

    return mega_pool, micro_pool, consumer_pool


def calculate_influencer_cost(degree: float, beta: float = 1.0, gamma: float = 1.0) -> float:
    """Calculates influencer cost using a power-law function: Cost = beta * (degree^gamma)."""
    return beta * (degree ** gamma)


def select_seeds(
        strategy: str,
        target_budget: float,
        node_degrees: torch.Tensor,
        mega_pool: torch.Tensor,
        micro_pool: torch.Tensor,
        consumer_pool: torch.Tensor,
        beta: float = 1.0,
        gamma: float = 1.0
):
    if strategy == "Mega":
        top_mega_idx = torch.argmax(node_degrees[mega_pool])
        selected_node = mega_pool[top_mega_idx].unsqueeze(0)
        deg = node_degrees[selected_node].item()
        achieved_cost = calculate_influencer_cost(deg, beta, gamma)
        return selected_node, achieved_cost

    elif strategy == "Micro":
        pool = micro_pool
    elif strategy == "Random":
        pool = consumer_pool

    perm = torch.randperm(len(pool), device=DEVICE)
    shuffled_pool = pool[perm]

    selected = []
    accumulated_cost = 0.0
    for node in shuffled_pool:
        deg = node_degrees[node].item()
        node_cost = calculate_influencer_cost(deg, beta, gamma)

        selected.append(node.item())
        accumulated_cost += node_cost
        if accumulated_cost >= target_budget:
            break

    seeds_tensor = torch.tensor(selected, device=DEVICE, dtype=torch.long)
    return seeds_tensor, accumulated_cost

    """Selects initial seed nodes given a strategy and matches degree budget B."""
    if strategy == "Mega":
        # Select the single highest-degree hub in the mega pool
        top_mega_idx = torch.argmax(node_degrees[mega_pool])
        selected_node = mega_pool[top_mega_idx].unsqueeze(0)
        achieved_budget = node_degrees[selected_node].item()
        return selected_node, achieved_budget

    elif strategy == "Micro":
        pool = micro_pool
    elif strategy == "Random":
        pool = consumer_pool
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Randomly shuffle candidate pool and accumulate nodes until budget is reached
    perm = torch.randperm(len(pool), device=DEVICE)
    shuffled_pool = pool[perm]

    selected = []
    accumulated_reach = 0.0
    for node in shuffled_pool:
        deg = node_degrees[node].item()
        selected.append(node.item())
        accumulated_reach += deg
        if accumulated_reach >= target_budget:
            break

    seeds_tensor = torch.tensor(selected, device=DEVICE, dtype=torch.long)
    return seeds_tensor, accumulated_reach


def run_simulation(
    seeds: torch.Tensor,
    alpha: float,
    mean_theta: float,
    num_nodes: int,
    node_degrees: torch.Tensor,
    src_nodes: torch.Tensor,
    tgt_nodes: torch.Tensor,
    std_theta: float = 0.10,
    max_ticks: int = 100,
):
    """Executes parallel GPU Network Cellular Automaton transition loop."""
    # State vector
    states = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
    states[seeds] = True

    # Draw individual consumer thresholds theta ~ N(mean_theta, std_theta) bounded in [0.05, 0.95]
    thresholds = torch.normal(mean_theta, std_theta, size=(num_nodes,), device=DEVICE)
    thresholds = torch.clamp(thresholds, min=0.05, max=0.95)

    # Persuasion weight decay: w_j = 1 / (1 + alpha * log10(k_j + 1))
    weights = 1.0 / (1.0 + alpha * torch.log10(node_degrees + 1.0))

    # Precompute total incoming weight capacity W_i for each node
    edge_weights = weights[src_nodes]
    total_capacity = torch.zeros(num_nodes, device=DEVICE)
    total_capacity.scatter_add_(0, tgt_nodes, edge_weights)

    adoption_history = [states.sum().item()]

    for tick in range(1, max_ticks + 1):
        # Calculate active incoming weight I_i
        active_edge_weights = edge_weights * states[src_nodes].float()
        active_influence = torch.zeros(num_nodes, device=DEVICE)
        active_influence.scatter_add_(0, tgt_nodes, active_edge_weights)

        # Calculate influence ratio R_i
        influence_ratio = torch.zeros(num_nodes, device=DEVICE)
        mask = total_capacity > 0
        influence_ratio[mask] = active_influence[mask] / total_capacity[mask]

        # Update state: adopt if influence ratio >= threshold
        new_states = states | (influence_ratio >= thresholds)

        # Check for convergence
        if new_states.sum() == states.sum():
            break

        states = new_states
        adoption_history.append(states.sum().item())

    return adoption_history


def execute_factorial_sweep(
    output_csv: str = "simulation_results.csv",
    replications: int = 30,
):
    """Executes full Monte Carlo sweep and logs structured results to CSV."""
    print(f"Using Compute Device: {DEVICE}")
    edge_index, num_nodes, node_degrees, src_nodes, tgt_nodes = load_graph_data()
    mega_pool, micro_pool, consumer_pool = stratify_node_pools(node_degrees)

    # Experimental Factors
    strategies = ["Mega", "Micro", "Random"]
    alphas = [0.0, 0.25, 0.50, 0.75, 1.00]
    mean_thetas = [0.10, 0.20, 0.30, 0.40]

    total_runs = len(strategies) * len(alphas) * len(mean_thetas) * replications
    print(
        f"Starting Factorial Sweep: {total_runs:,} total simulation runs across {replications} replications..."
    )

    # CSV Header
    headers = [
        "run_id",
        "replication",
        "strategy",
        "fatigue_alpha",
        "mean_threshold",
        "seed_count",
        "achieved_budget",
        "convergence_tick",
        "final_adopters",
        "market_saturation",
        "ross",
    ]

    start_time = time.time()
    run_id = 0

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for rep in range(1, replications + 1):
            for alpha in alphas:
                for mean_theta in mean_thetas:
                    # Select Mega hub first to establish baseline campaign budget B
                    mega_seeds, target_budget = select_seeds(
                        "Mega", 0, node_degrees, mega_pool, micro_pool, consumer_pool
                    )

                    for strategy in strategies:
                        run_id += 1

                        # Select seeds strictly budget-matched to baseline B
                        if strategy == "Mega":
                            seeds, budget = mega_seeds, target_budget
                        else:
                            seeds, budget = select_seeds(
                                strategy,
                                target_budget,
                                node_degrees,
                                mega_pool,
                                micro_pool,
                                consumer_pool,
                            )

                        history = run_simulation(
                            seeds=seeds,
                            alpha=alpha,
                            mean_theta=mean_theta,
                            num_nodes=num_nodes,
                            node_degrees=node_degrees,
                            src_nodes=src_nodes,
                            tgt_nodes=tgt_nodes,
                        )

                        # Output metrics
                        final_adopters = history[-1]
                        convergence_tick = len(history) - 1
                        market_saturation = final_adopters / num_nodes
                        seed_count = len(seeds)
                        ross = (
                            (final_adopters - seed_count) / budget
                            if budget > 0
                            else 0.0
                        )

                        writer.writerow(
                            [
                                run_id,
                                rep,
                                strategy,
                                alpha,
                                mean_theta,
                                seed_count,
                                budget,
                                convergence_tick,
                                final_adopters,
                                round(market_saturation, 5),
                                round(ross, 4),
                            ]
                        )

            # Progress update per replication
            elapsed = time.time() - start_time
            print(
                f"Replication {rep}/{replications} completed | Elapsed: {elapsed:.1f}s"
            )

    print(
        f"\nSimulation Complete! Dataset exported to '{output_csv}' ({run_id} rows)."
    )


if __name__ == "__main__":
    # Execute 30 Monte Carlo replications per parameter condition
    execute_factorial_sweep(output_csv="simulation_results.csv", replications=30)