import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Set global compute device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_nx_graph(pt_filename: str = "facebook_tensor.pt") -> nx.Graph:
    """Loads preprocessed PyTorch graph tensor into a NetworkX graph."""
    if not os.path.exists(pt_filename):
        raise FileNotFoundError(f"File '{pt_filename}' not found. Run preprocessing first.")

    edge_index, num_nodes = torch.load(pt_filename)
    edges = edge_index.t().cpu().numpy()

    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(edges)
    return G


def extract_visual_subgraph(G: nx.Graph, max_nodes: int = 180) -> tuple[nx.Graph, dict]:
    """
    Extracts a connected 200-node neighborhood around top hubs
    and precomputes 2D layout coordinates for consistent rendering.
    """
    # Find top degree nodes
    degrees = dict(G.degree())
    top_hubs = sorted(degrees, key=degrees.get, reverse=True)[:5]

    # Collect immediate ego-neighborhoods of top hubs
    sub_nodes = set(top_hubs)
    for hub in top_hubs:
        neighbors = list(G.neighbors(hub))
        sub_nodes.update(neighbors[:35])
        if len(sub_nodes) >= max_nodes:
            break

    G_sub = G.subgraph(list(sub_nodes)[:max_nodes]).copy()

    # Precompute fixed 2D layout so nodes stay stationary during animation
    pos = nx.spring_layout(G_sub, seed=42, k=0.25)
    return G_sub, pos


def run_subgraph_simulation(
        G_sub: nx.Graph,
        strategy: str,
        alpha: float = 0.50,
        mean_theta: float = 0.25,
        max_ticks: int = 20
) -> list[dict[int, str]]:
    """
    Runs the Network Cellular Automaton on the visual subgraph.
    Returns a tick-by-tick list of node color mapping states.
    """
    degrees = dict(G_sub.degree())
    nodes = list(G_sub.nodes())
    num_nodes = len(nodes)

    # Identify Mega seed (highest degree node in subgraph)
    mega_seed = max(degrees, key=degrees.get)
    target_budget = degrees[mega_seed]

    # Select seeds
    if strategy == "Mega":
        seeds = {mega_seed}
    else:  # Micro strategy
        # Select micro-influencers (medium-high degree) until budget is met
        sorted_by_degree = sorted(degrees.items(), key=lambda x: x[1], reverse=True)
        micro_candidates = [n for n, d in sorted_by_degree[5:40]]

        seeds = set()
        accumulated_reach = 0
        for node in micro_candidates:
            seeds.add(node)
            accumulated_reach += degrees[node]
            if accumulated_reach >= target_budget:
                break

    # State tracking: 0 = Unaware, 1 = Adopted
    states = {node: (1 if node in seeds else 0) for node in nodes}

    # Thresholds and weights
    import numpy as np
    np.random.seed(42)
    thresholds = {node: np.clip(np.random.normal(mean_theta, 0.08), 0.05, 0.95) for node in nodes}
    weights = {node: 1.0 / (1.0 + alpha * np.log10(degrees[node] + 1.0)) for node in nodes}

    # Color mapping history
    # Colors: Unaware = Light Gray, Seed = Gold, Adopter = Teal/Green
    color_history = []

    def get_colors():
        colors = []
        for n in G_sub.nodes():
            if n in seeds:
                colors.append("#FFD700")  # Gold for initial seeds
            elif states[n] == 1:
                colors.append("#00B4D8")  # Teal for converted adopters
            else:
                colors.append("#E0E0E0")  # Light gray for unaware
        return colors

    color_history.append(get_colors())

    for tick in range(1, max_ticks + 1):
        new_states = states.copy()

        for node in G_sub.nodes():
            if states[node] == 1:
                continue

            neighbors = list(G_sub.neighbors(node))
            if not neighbors:
                continue

            active_weight = sum(weights[nbr] for nbr in neighbors if states[nbr] == 1)
            total_capacity = sum(weights[nbr] for nbr in neighbors)

            ratio = active_weight / total_capacity if total_capacity > 0 else 0
            if ratio >= thresholds[node]:
                new_states[node] = 1

        # Check for convergence
        if sum(new_states.values()) == sum(states.values()):
            # Pad remaining frames if converged early
            color_history.extend([color_history[-1]] * (max_ticks - tick + 1))
            break

        states = new_states
        color_history.append(get_colors())

    return color_history


def create_animated_comparison(
        output_gif: str = "campaign_spread_comparison.gif",
        max_ticks: int = 15
):
    """Generates and saves a side-by-side animated GIF of Mega vs. Micro campaigns."""
    print("Loading graph data for visualization...")
    G = load_nx_graph("facebook_tensor.pt")

    print("Extracting connected neighborhood and precomputing 2D layout...")
    G_sub, pos = extract_visual_subgraph(G, max_nodes=180)

    print("Running simulation step-by-step for visualization frames...")
    history_mega = run_subgraph_simulation(G_sub, strategy="Mega", max_ticks=max_ticks)
    history_micro = run_subgraph_simulation(G_sub, strategy="Micro", max_ticks=max_ticks)

    frames = min(len(history_mega), len(history_micro))

    # Setup Matplotlib Figure (Dark Theme / High Contrast for Slides)
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(
        "Adoption Cascade Dynamics: Mega-Influencer vs. Micro-Influencer Portfolio",
        fontsize=16,
        fontweight='bold',
        y=0.98
    )

    def update(frame):
        ax1.clear()
        ax2.clear()

        # Left Plot: Mega Strategy
        ax1.set_title(f"Mega-Influencer Campaign (Tick {frame})", fontsize=13, color='#FFD700')
        nx.draw_networkx_edges(G_sub, pos, ax=ax1, alpha=0.15, edge_color='#888888')
        nx.draw_networkx_nodes(
            G_sub, pos, ax=ax1,
            node_color=history_mega[frame],
            node_size=60,
            linewidths=0.5,
            edgecolors='#111111'
        )
        ax1.axis('off')

        # Right Plot: Micro Portfolio Strategy
        ax2.set_title(f"Micro-Influencer Portfolio (Tick {frame})", fontsize=13, color='#00B4D8')
        nx.draw_networkx_edges(G_sub, pos, ax=ax2, alpha=0.15, edge_color='#888888')
        nx.draw_networkx_nodes(
            G_sub, pos, ax=ax2,
            node_color=history_micro[frame],
            node_size=60,
            linewidths=0.5,
            edgecolors='#111111'
        )
        ax2.axis('off')

    print("Rendering animation frames...")
    anim = FuncAnimation(fig, update, frames=frames, interval=600, repeat=True)

    # Save as animated GIF using Pillow writer
    writer = PillowWriter(fps=2)
    anim.save(output_gif, writer=writer)
    plt.close()

    print(f"\nAnimation successfully exported to '{output_gif}'!")


if __name__ == "__main__":
    create_animated_comparison(output_gif="campaign_spread_comparison.gif", max_ticks=15)