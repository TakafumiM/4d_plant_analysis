import numpy as np
import open3d as o3d
import ot
import sys
from scipy.spatial.distance import cdist

sys.path.insert(0, "./p2p_matching_in_organ")
from p2p_matching_seg_icp import form_maize_org
from ot.unbalanced import sinkhorn_unbalanced


def visualize_correspondences(pcd1, pcd2, matches):
    """
    Visualizes correspondences between two point clouds.

    Args:
        pcd1 (open3d.geometry.PointCloud): The first point cloud.
        pcd2 (open3d.geometry.PointCloud): The second point cloud.
        matches (np.ndarray): A (N, 2) array of matching indices.
    """
    pcd1_temp = o3d.geometry.PointCloud(pcd1)
    pcd2_temp = o3d.geometry.PointCloud(pcd2)

    pcd1_temp.paint_uniform_color([1, 0.706, 0])  # Gold
    pcd2_temp.paint_uniform_color([0, 0.651, 0.929]) # Blue

    # Create a line set for the correspondences
    lines = []
    points = []
    for i, j in matches:
        points.append(pcd1.points[i])
        points.append(pcd2.points[j])
        lines.append([len(points) - 2, len(points) - 1])

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )

    o3d.visualization.draw_geometries([pcd1_temp, pcd2_temp, line_set])


if __name__ == "__main__":
    # 1. Load two point clouds
    day1 = "03-13_AM"
    day2 = "03-14_AM"

    org1 = form_maize_org(day1)
    org2 = form_maize_org(day2)

    pcd1 = org1['pcd']
    pcd2 = org2['pcd']

    # For simplicity, let's work with NumPy arrays
    points1 = np.asarray(pcd1.points)
    points2 = np.asarray(pcd2.points)

    # Optional: Downsample for faster computation during testing
    # points1 = points1[::10]
    # points2 = points2[::10]

    print(f"Loaded point cloud 1 with {len(points1)} points.")
    print(f"Loaded point cloud 2 with {len(points2)} points.")

    # 2. Compute the cost matrix (squared Euclidean distance)
    print("Computing cost matrix...")
    cost_matrix = cdist(points1, points2, 'sqeuclidean')
    cost_matrix /= cost_matrix.max()

    # 3. Compute the Optimal Transport plan
    print("Solving for Optimal Transport plan...")
    # Create uniform distributions on the point clouds
    n1, n2 = len(points1), len(points2)
    mu = np.ones(n1) / n1
    nu = np.ones(n2) / n2

    # Regularization parameters
    reg = 1e-1  # Entropy regularization
    reg_m = 1.0 # Marginal relaxation penalty

    # Solve with Unbalanced Sinkhorn algorithm
    transport_plan = sinkhorn_unbalanced(mu, nu, cost_matrix, reg, reg_m)

    # 4. Extract correspondences
    # For each point in pcd1, find the point in pcd2 it's most likely mapped to
    matches_indices = np.argmax(transport_plan, axis=1)
    
    # Create a (N, 2) array of matching indices
    correspondences = np.column_stack([np.arange(len(points1)), matches_indices])

    print(f"Found {len(correspondences)} correspondences.")

    # 5. Visualize the results
    print("Visualizing correspondences...")
    visualize_correspondences(pcd1, pcd2, correspondences)

    print("Done.")