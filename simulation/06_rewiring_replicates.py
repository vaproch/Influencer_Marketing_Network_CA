"""
Stage 6 (optional) - independent realisations of the randomised network.
Output: data/results/06_rewiring_replicates.csv

In stages 4 and 5 each randomisation level is a single randomised network
(seed REWIRE_SEED). Replications vary the seeds and thresholds on that network
only, so they do not capture differences between randomised networks. This
script repeats the partial-randomisation experiment on several independently
randomised networks, which separates variation between network realisations
from Monte Carlo variation within one network.

The model code of stage 5 (05_partial_rewiring.py) is imported unchanged; only
the randomisation seed (REWIRE_SEED) differs between network instances.
Instance 0 uses the original seed, so its runs reproduce the matching rows of
data/results/05_partial_rewiring.csv exactly, which the script checks.

Default design (no fatigue, 30 replications per cell):
    randomised share  : 25% and 50%
    network instances : 5 per share (instance 0 = the network used in stage 5)
    mean threshold    : 0.15, 0.20
    campaign size K   : 1, 3, 10
    strategies        : Mega, Micro, Niche, Random
    -> 2 x 5 x 2 x 3 x 4 x 30 = 7,200 runs

Columns are those of stage 5 plus `network_instance` and `rewire_seed`.

Usage
-----
    python simulation/06_rewiring_replicates.py
    python simulation/06_rewiring_replicates.py --instances 10 --shares 0.25 0.5
"""

import argparse
import csv
import importlib.util
import os
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE5 = HERE / "05_partial_rewiring.py"


def load_stage5():
    """Import 05_partial_rewiring.py as a module (its name starts with a digit)."""
    spec = importlib.util.spec_from_file_location("stage5", STAGE5)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description="Independent realisations of the randomised network.")
    ap.add_argument("--shares", type=float, nargs="+", default=[0.25, 0.50],
                    help="shares of edges to randomise (default 0.25 0.50)")
    ap.add_argument("--instances", type=int, default=5,
                    help="independent randomised networks per share (default 5)")
    ap.add_argument("--replications", type=int, default=30)
    ap.add_argument("--thetas", type=float, nargs="+", default=[0.15, 0.20],
                    help="mean adoption thresholds (default 0.15 0.20)")
    ap.add_argument("--output", default=None,
                    help="output CSV (default data/results/06_rewiring_replicates.csv)")
    args = ap.parse_args()

    s5 = load_stage5()
    output = Path(args.output) if args.output else s5.RESULTS_DIR / "06_rewiring_replicates.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    base_seed = s5.REWIRE_SEED

    header_written = False
    with open(output, "w", newline="") as out_f:
        writer = csv.writer(out_f)
        for instance in range(args.instances):
            seed = base_seed + 1000 * instance          # instance 0 = stage-5 network
            s5.REWIRE_SEED = seed                        # read by Graph() at construction
            variants = [{"name": f"rewire{round(100 * sh)}", "rewire": sh, "gamma": 1.0}
                        for sh in args.shares]
            print(f"\n##### Network instance {instance} (rewire seed {seed}) #####")

            fd, tmp = tempfile.mkstemp(suffix=".csv")
            os.close(fd)
            try:
                s5.execute_robustness(
                    output_csv=tmp,
                    replications=args.replications,
                    alphas=(0.0,),
                    mean_thetas=tuple(args.thetas),
                    variants=variants,
                )
                with open(tmp, newline="") as in_f:
                    reader = csv.reader(in_f)
                    header = next(reader)
                    if not header_written:
                        writer.writerow(["network_instance", "rewire_seed"] + header)
                        header_written = True
                    for row in reader:
                        writer.writerow([instance, seed] + row)
                out_f.flush()
            finally:
                os.remove(tmp)

    print(f"\nDone. Results written to '{output}'.")


if __name__ == "__main__":
    main()
