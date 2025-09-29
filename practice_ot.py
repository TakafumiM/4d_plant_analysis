import numpy as np
import open3d as o3d
import ot
import argparse
import os
from scipy.spatial.distance import cdist


def visualize_correspondences(pcd1, pcd2, transport_plan, offset_value=20):
    """
    Visualizes correspondences from an OT plan between two point clouds.

    Args:
        pcd1 (open3d.geometry.PointCloud): The first point cloud.
        pcd2 (open3d.geometry.PointCloud): The second point cloud.
        transport_plan (np.ndarray): The OT matrix.
        offset_value (float): How far to move the second cloud for visualization.
    """
    pcd1_vis = o3d.geometry.PointCloud(pcd1)
    pcd2_vis = o3d.geometry.PointCloud(pcd2)

    # Apply a color and a spatial offset to the second cloud for clarity
    pcd1_vis.paint_uniform_color([1, 0.706, 0])  # Gold
    pcd2_vis.paint_uniform_color([0, 0.651, 0.929]) # Blue
    pcd2_vis.translate([offset_value, 0, 0])

    # Get points from the translated clouds
    points1 = np.asarray(pcd1_vis.points)
    points2 = np.asarray(pcd2_vis.points)

    # Find the most likely correspondences from the transport plan
    matches_indices = np.argmax(transport_plan, axis=1)
    
    # Create a (N, 2) array of matching indices
    correspondences = np.column_stack([np.arange(len(points1)), matches_indices])

    # Create a line set for the correspondences
    lines = []
    points = []
    for i, j in correspondences:
        points.append(points1[i])
        points.append(points2[j])
        lines.append([len(points) - 2, len(points) - 1])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    print("Displaying visualization. Press 'Q' to close.")
    o3d.visualization.draw_geometries([pcd1_vis, pcd2_vis, line_set])


def main(args):
    """
    Main function to run Optimal Transport on registered point cloud segments.
    """
    # 1. Load two registered point cloud segments
    save_path = args.path_format.format(args.dataset, args.day1, args.day2)
    
    file1 = os.path.join(save_path, f"{args.day1}_{args.segment_index}.csv")
    file2 = os.path.join(save_path, f"{args.day2}_{args.segment_index}.csv")

    if not os.path.exists(file1) or not os.path.exists(file2):
        print(f"Error: Could not find files for segment {args.segment_index}.")
        print(f"""Checked for:
 - {file1}
 - {file2}""")
        return

    points1 = np.loadtxt(file1)
    points2 = np.loadtxt(file2)

    # Ensure point clouds have the same number of points for ot.emd
    # This can be done by padding or resampling, here we simply truncate
    min_points = min(len(points1), len(points2))
    points1 = points1[:min_points]
    points2 = points2[:min_points]

    pcd1 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points1))
    pcd2 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points2))

    print(f"Loaded segment {args.segment_index} for {args.day1} and {args.day2}.")
    print(f"Point cloud 1 shape: {points1.shape}")
    print(f"Point cloud 2 shape: {points2.shape}")

    # 2. Compute the cost matrix (squared Euclidean distance)
    print("Computing cost matrix...")
    cost_matrix = cdist(points1, points2, 'sqeuclidean')
    cost_matrix /= cost_matrix.max() # Normalize cost matrix

    # 3. Define weights and compute the Optimal Transport plan using EMD
    print("Solving for Optimal Transport plan with EMD...")
    n_points = len(points1)
    a = np.ones(n_points) / n_points
    b = np.ones(n_points) / n_points

    # Earth Mover's Distance is a linear program, no regularization needed
    transport_plan = ot.emd(a, b, cost_matrix)

    ot_cost = np.sum(transport_plan * cost_matrix)
    print(f"Optimal Transport cost (Earth Mover's Distance): {ot_cost:.4f}")

    # 4. Visualize the results
    print("Visualizing correspondences...")
    visualize_correspondences(pcd1, pcd2, transport_plan)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Optimal Transport on registered plant segments.")
    parser.add_argument("--dataset", type=str, default="tomato", help="Dataset name (e.g., 'tomato', 'maize').")
    parser.add_argument("--day1", type=str, default="03-05_AM", help="First day/timestamp.")
    parser.add_argument("--day2", type=str, default="03-06_AM", help="Second day/timestamp.")
    parser.add_argument("--segment_index", type=int, default=0, help="Index of the segment to analyze.")
    parser.add_argument("--path_format", type=str, 
                        default="data/{}/registration_result/{}_to_{}/",
                        help="Format string for the path to saved segments.")
    
    args = parser.parse_args()
    main(args)