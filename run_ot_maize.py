import copy
import os
import sys
import time
import json
import numpy as np
import open3d
import ot
from scipy.spatial.distance import cdist

from organ_matching.organ_matching_lr import match_organ_two_days
from p2p_matching_in_organ.p2p_matching_seg_icp import form_maize_org, form_tomato_org
from visualize_p2p import visualize_pcd_registration_series
import argparse
from ot.unbalanced import sinkhorn_unbalanced



parser = argparse.ArgumentParser()
parser.add_argument("--type",
                    help="specify the type of plant to run the registration from tomato, maize and arabidopsis, " +
                         "default maize", default="maize")
args = parser.parse_args()
if args.type:
    assert args.type in ["tomato", "maize", "arabidopsis"]
    if args.type == "arabidopsis":
        dataset = "lyon2"
    else:
        dataset = args.type
else:
    dataset = "maize"


score_path = "scores/"
ot_save_path = "ot_results/"
if not os.path.exists(ot_save_path):
    os.mkdir(ot_save_path)


if __name__ == "__main__":
    if dataset == "maize":
        days = ["03-13_AM", "03-14_AM", "03-15_AM", "03-16_AM", "03-17_AM", "03-18_AM", "03-20_AM"]
    else:
        print("This script is designed for maize dataset.")
        sys.exit()

    options_path = "hyper_parameters/{}.json".format(dataset)
    with open(options_path, "r") as json_file:
        options = json.load(json_file)

    t_0 = time.time()
    for i in range(len(days) - 1):
        day1 = days[i]
        day2 = days[i + 1]
        
        print("Running optimal transport for maize in {} and {}".format(day1, day2))
        t_start = time.time()

        org1 = form_maize_org(day1)
        org2 = form_maize_org(day2)

        pcd1 = org1['pcd']
        pcd2 = org2['pcd']

        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)

        print(f"Loaded point cloud 1 with {len(points1)} points.")
        print(f"Loaded point cloud 2 with {len(points2)} points.")

        print("Computing cost matrix...")
        cost_matrix = cdist(points1, points2, 'sqeuclidean')
        cost_matrix /= cost_matrix.max()

        print("Solving for Optimal Transport plan...")
        n1, n2 = len(points1), len(points2)
        mu = np.ones(n1) / n1
        nu = np.ones(n2) / n2

        reg = 1e-1
        reg_m = 1.0 

        transport_plan = sinkhorn_unbalanced(mu, nu, cost_matrix, reg, reg_m)
        
        save_folder = os.path.join(ot_save_path, f"{day1}_to_{day2}")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        
        np.savetxt(os.path.join(save_folder, "transport_plan.csv"), transport_plan, delimiter=",")
        
        t_end = time.time()
        print("Optimal transport finished, used ", (t_end - t_start), " s")
        print(64 * "-")
        print("")

    t_end = time.time()
    print("")
    print(128 * "=")
    print("All pair of plants optimal transport finished, used {} s".format((t_end - t_0)))
