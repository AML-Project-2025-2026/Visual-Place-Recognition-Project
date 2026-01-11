#This script was modified to compute also the average processing time per query

import time


import os
import sys
import argparse
import torch
from glob import glob
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy

from util import read_file_preds

sys.path.append(str(Path(__file__).parent.joinpath("image-matching-models")))

from matching import get_matcher, available_models
from matching.utils import get_default_device

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--preds-dir", type=str, help="directory with predictions of a VPR model")
    parser.add_argument("--out-dir", type=str, default=None, help="output directory of image matching results")
    # Choose matcher
    parser.add_argument(
        "--matcher",
        type=str,
        default="sift-lg",
        choices=available_models,
        help="choose your matcher",
    )
    parser.add_argument("--device", type=str, default=get_default_device(), choices=["cpu", "cuda"])
    parser.add_argument("--im-size", type=int, default=512, help="resize img to im_size x im_size")
    parser.add_argument("--num-preds", type=int, default=100, help="number of predictions to match")
    parser.add_argument("--start-query", type=int, default=-1, help="query to start from")
    parser.add_argument("--num-queries", type=int, default=-1, help="number of queries")

    return parser.parse_args()

def main(args):
    device = args.device
    matcher_name = args.matcher
    img_size = args.im_size
    num_preds = args.num_preds
    matcher = get_matcher(matcher_name, device=device)
    preds_folder = args.preds_dir
    start_query = args.start_query
    num_queries = args.num_queries

    output_folder = Path(preds_folder + f"_{matcher_name}") if args.out_dir is None else Path(args.out_dir)
    output_folder.mkdir(exist_ok=True)
    
    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    start_query = start_query if start_query >= 0 else 0
    num_queries = num_queries if num_queries >= 0 else len(txt_files)

    for txt_file in tqdm(txt_files[start_query : start_query + num_queries]):
        q_num = Path(txt_file).stem
        out_file = output_folder.joinpath(f"{q_num}.torch")
        if out_file.exists():
            continue
        results = []
        q_path, pred_paths = read_file_preds(txt_file)
        img0 = matcher.load_image(q_path, resize=img_size)

        #
        query_time = 0.0
        for pred_path in pred_paths[:num_preds]:
            img1 = matcher.load_image(pred_path, resize=img_size)
            #
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = matcher(deepcopy(img0), img1)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            #op
            query_time += (end - start)

            result["all_desc0"] = result["all_desc1"] = None
            results.append(result)
        
        #
        avg_time_per_match = query_time / num_preds
        result_summary = {
            "avg_time_per_match": avg_time_per_match,
            "total_query_time": query_time,
        }
        #torch.save(results, out_file)
        torch.save(
            {"results": results, "timing": result_summary},
            out_file
        )

if __name__ == "__main__":
    args = parse_arguments()
    main(args)

