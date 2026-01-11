import argparse
import torch
from glob import glob
from pathlib import Path
from tqdm import tqdm

from util import get_list_distances_from_preds


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--preds-dir", type=str, required=True)
    parser.add_argument("--inliers-dir", type=str, required=True)
    parser.add_argument("--tau", type=int, required=True,
                        help="Fixed inliers threshold τ")
    parser.add_argument("--num-preds", type=int, default=20)
    parser.add_argument("--positive-dist-threshold", type=int, default=25)
    parser.add_argument("--avg-full-rerank-time", type=float, required=True,
                        help="Avg time per query for FULL re-ranking (seconds)")

    return parser.parse_args()


# This script evaluates an adaptive re-ranking strategy using a fixed
# inlier threshold τ. For each query, the refinement step with reranking to be applied
# only if the top-1 retrieved result has fewer than τ inliers.

def main(args):
    preds_folder = args.preds_dir
    inliers_folder = Path(args.inliers_dir)

    txt_files = sorted(
        glob(str(Path(preds_folder) / "*.txt")),
        key=lambda x: int(Path(x).stem)
    )

    total_queries = len(txt_files)
    full_total_time = total_queries * args.avg_full_rerank_time

    retrieval_correct = 0
    full_correct = 0
    adaptive_correct = 0
    adaptive_reranked = 0

    for txt_file in tqdm(txt_files):
        geo_dists = torch.tensor(
            get_list_distances_from_preds(txt_file)
        )[:args.num_preds]

        torch_file = inliers_folder / Path(txt_file).name.replace("txt", "torch")
        data = torch.load(torch_file, weights_only=False)

        inliers = torch.tensor(
            [r["num_inliers"] for r in data["results"][:args.num_preds]]
        )

        # Retrieval-only
        if geo_dists[0] <= args.positive_dist_threshold:
            retrieval_correct += 1

        # Full re-ranking
        _, idx_full = torch.sort(inliers, descending=True)
        if geo_dists[idx_full][0] <= args.positive_dist_threshold:
            full_correct += 1

        # Adaptive
        inliers_top1 = inliers[0].item()
        if inliers_top1 < args.tau:
            adaptive_reranked += 1
            idx = idx_full
        else:
            idx = torch.arange(len(geo_dists))

        if geo_dists[idx][0] <= args.positive_dist_threshold:
            adaptive_correct += 1

    # Metrics
    retrieval_r1 = retrieval_correct / total_queries * 100
    full_r1 = full_correct / total_queries * 100
    adaptive_r1 = adaptive_correct / total_queries * 100

    rerank_frac = adaptive_reranked / total_queries
    adaptive_avg_time = rerank_frac * args.avg_full_rerank_time
    adaptive_total_time = rerank_frac * full_total_time
    saved_time = full_total_time - adaptive_total_time
    saved_percent = (1 - rerank_frac) * 100

    # Print summary
    print("\nEVALUATION RESULTS")
    print(f"τ = {args.tau}")
    print(f"Total queries: {total_queries}")

    print("\nAccuracy:")
    print(f"Retrieval-only R@1: {retrieval_r1:.2f}")
    print(f"Full re-ranking R@1: {full_r1:.2f}")
    print(f"Adaptive R@1: {adaptive_r1:.2f}")

    print("\nCost:")
    print(f"Adaptive reranked queries: {rerank_frac*100:.1f}%")
    print(f"Avg time / query (adaptive): {adaptive_avg_time:.3f}s")
    print(f"Total time saved: {saved_time:.1f}s ({saved_percent:.1f}%)")


if __name__ == "__main__":
    args = parse_args()
    main(args)
