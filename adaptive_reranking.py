import argparse
import torch
import numpy as np
from glob import glob
from pathlib import Path
from tqdm import tqdm

from util import get_list_distances_from_preds

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--preds-dir", type=str, required=True,
                        help="Directory with retrieval predictions (.txt)")
    parser.add_argument("--inliers-dir", type=str, required=True,
                        help="Directory with image matching results (.torch)")
    parser.add_argument("--taus", type=int, nargs="+", required=True,
                        help="List of inlier thresholds (e.g. 5 10 20 30)")
    parser.add_argument("--num-preds", type=int, default=20)
    parser.add_argument("--positive-dist-threshold", type=int, default=25)
    parser.add_argument("--avg-full-rerank-time", type=float, required=True,
                        help="Average time per query for FULL re-ranking (seconds)")

    return parser.parse_args()

# This script evaluates an adaptive re-ranking strategy:
# the refinement step with reranking to be applied only when the top-1 match has
# fewer than τ inliers, trading off accuracy vs computation time.

def main(args):
    preds_folder = args.preds_dir
    inliers_folder = Path(args.inliers_dir)

    txt_files = glob(str(Path(preds_folder) / "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    total_queries = len(txt_files)

    full_total_time = total_queries * args.avg_full_rerank_time

    print(f"Total queries: {total_queries}")

    # Full re-ranking baseline
    full_rerank_correct = 0

    # Retrieval only baseline 
    retrieval_correct = 0

    # Per-threshold stats 
    stats = {tau: {"correct": 0, "reranked": 0} for tau in args.taus}

    # Make sure thresholds sorted
    sorted_taus = sorted(args.taus)

    for txt_file in tqdm(txt_files):
        geo_dists = torch.tensor(
            get_list_distances_from_preds(txt_file)
        )[:args.num_preds]

        torch_file = inliers_folder / Path(txt_file).name.replace("txt", "torch")
        data = torch.load(torch_file, weights_only=False)

        inliers = torch.tensor(
            [r["num_inliers"] for r in data["results"][:args.num_preds]]
        )

        # Retrieval-only (top-1)
        if geo_dists[0] <= args.positive_dist_threshold:
            retrieval_correct += 1

        # Full re-ranking
        _, idx_full = torch.sort(inliers, descending=True)
        if torch.any(geo_dists[idx_full][0:1] <= args.positive_dist_threshold):
            full_rerank_correct += 1

        # Adaptive
        inliers_top1 = inliers[0].item()

        for tau in args.taus:
            if inliers_top1 < tau:
                # Hard
                stats[tau]["reranked"] += 1
                idx = idx_full
            else:
                #No reranking
                idx = torch.arange(len(geo_dists))

            if torch.any(geo_dists[idx][0:1] <= args.positive_dist_threshold):
                stats[tau]["correct"] += 1


    #For plotting
    r1_values = []
    rerank_fracs = []
    saved_percents = []
    

    for tau in args.taus:
        rerank_fracs.append(stats[tau]["reranked"] / total_queries)
        r1_values.append(stats[tau]["correct"] / total_queries * 100)
        saved_percents.append((1 - stats[tau]["reranked"] / total_queries) * 100)


    # We select the smallest τ that achieves at least 99%
    # of the full re-ranking performance
    target_r1 = 0.99 * full_rerank_correct / total_queries * 100
    optimal_tau = None
    optimal_r1 = None
    optimal_saving = None

    # Find the first tau that satisfies the condition
    for i, r1 in enumerate(r1_values):
        if r1 >= target_r1:
            optimal_tau = sorted_taus[i]
            optimal_r1 = r1
            optimal_saving = saved_percents[i]
            break
    

    # Results
    print("\nBASELINES")
    print(f"Retrieval-only R@1: {retrieval_correct / total_queries * 100:.2f}")
    print(f"Full re-ranking R@1: {full_rerank_correct / total_queries * 100:.2f}")

    print("\nADAPTIVE RE-RANKING")
    for tau in sorted_taus:
        reranked_frac = stats[tau]["reranked"] / total_queries
        adaptive_r1 = stats[tau]["correct"] / total_queries * 100

        avg_time = reranked_frac * args.avg_full_rerank_time
        adaptive_total_time = reranked_frac * full_total_time
        saved_time = full_total_time - adaptive_total_time
        saved_percent = (1 - reranked_frac) * 100

        marker = ""
        if optimal_tau is not None and tau == optimal_tau:
            marker = " <--- [99% performance reached]"

        print(
            f"τ={tau:>3d} | "
            f"reranked={reranked_frac*100:5.1f}% | "
            f"R@1={adaptive_r1:5.2f} | "
            f"avg_time/query={avg_time:.3f}s | "
            f"saved_total={saved_time:.1f}s ({saved_percent:.1f}%)"
            f"{marker}"
        )

    #Plotting

    fig, ax1 = plt.subplots(figsize=(8, 6))

    # R@1
    ax1.plot(args.taus, r1_values, marker='o', label='Adaptive R@1')
    ax1.axhline(
        y=full_rerank_correct / total_queries * 100,
        linestyle='--',
        label='Full re-ranking'
    )
    ax1.axhline(
        y=retrieval_correct / total_queries * 100,
        linestyle=':',
        color='gray',
        label='Retrieval-only'
    )

    ax1.set_xlabel("Inliers threshold τ")
    ax1.set_ylabel("Recall@1 (%)")
    ax1.grid(alpha=0.3)

    # Cost saving axis
    ax2 = ax1.twinx()
    ax2.plot(args.taus, saved_percents, marker='s', color='r', linestyle='--', label='Cost saving')
    ax2.set_ylabel("Cost saving (%)")

    # To highlight 99% point
    if optimal_tau is not None:

        ax1.scatter([optimal_tau], [optimal_r1], color='gold', edgecolor='black', s=200, zorder=10, marker='*', label='≥99% Full perf.')
        
        annotation_text = f"τ={optimal_tau}\nSaving={optimal_saving:.1f}%"
        ax1.annotate(annotation_text,
                     xy=(optimal_tau, optimal_r1),
                     xytext=(optimal_tau, optimal_r1 - (full_rerank_correct / total_queries * 100 * 0.05)),
                     ha='center',
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3"))
        
       # Vertical line to see where it falls on the X-axis
        ax1.axvline(x=optimal_tau, color='gold', linestyle='-.', alpha=0.5)

    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.title("Adaptive Re-ranking: Performance vs Cost")
    plt.tight_layout()
    plt.xticks(args.taus[::1])  # show all passed taus, or use [::2] for every 2 values

    output_dir = Path("plots_adaptive_reranking")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"r1_vs_tau_{Path(preds_folder).parent.name}.png"
    save_path = output_dir / file_name

    plt.savefig(save_path, dpi=200)
    plt.savefig(f"r1_vs_tau_{Path(preds_folder).parent.name}.png", dpi=200)
    plt.close()




if __name__ == "__main__":
    args = parse_args()
    main(args)
