import os
import numpy as np
import open3d as o3d
import sys
sys.path.insert(0, "./p2p_matching_in_organ")
from p2p_matching_in_organ.p2p_matching_seg_icp import form_maize_org

def visualize_ot_all_days():
    days = ["03-13_AM", "03-14_AM", "03-15_AM", "03-16_AM", "03-17_AM", "03-18_AM", "03-20_AM"]
    ot_results_path = "ot_results/"
    geometries = []
    x_offset = 0

    # A color palette for the point clouds
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1], [0.5, 0.5, 0.5]]

    pcds = []
    for i, day in enumerate(days):
        print(f"Loading point cloud for {day}...")
        org = form_maize_org(day)
        pcd = org['pcd']
        
        # Translate the point cloud
        pcd.translate((x_offset, 0, 0))
        x_offset += 100 # Adjust this offset if necessary
        
        pcd.paint_uniform_color(colors[i % len(colors)])
        pcds.append(pcd)
        geometries.append(pcd)

    for i in range(len(days) - 1):
        day1 = days[i]
        day2 = days[i+1]
        print(f"Processing transport from {day1} to {day2}...")

        transport_plan_path = os.path.join(ot_results_path, f"{day1}_to_{day2}", "transport_plan.csv")
        if not os.path.exists(transport_plan_path):
            print(f"Transport plan not found for {day1} to {day2}. Skipping.")
            continue

        transport_plan = np.loadtxt(transport_plan_path, delimiter=",")

        pcd1 = pcds[i]
        pcd2 = pcds[i+1]

        points1 = np.asarray(pcd1.points)
        points2 = np.asarray(pcd2.points)

        # Get the strongest correspondences
        matches_indices = np.argmax(transport_plan, axis=1)
        correspondences = np.column_stack([np.arange(len(points1)), matches_indices])

        # Create lineset
        lines = []
        points_for_lines = []
        for k, (src_idx, tgt_idx) in enumerate(correspondences):
            # Add a line every 5 correspondences to avoid clutter
            if k % 5 == 0:
                points_for_lines.append(points1[src_idx])
                points_for_lines.append(points2[tgt_idx])
                lines.append([len(points_for_lines) - 2, len(points_for_lines) - 1])

        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(points_for_lines),
            lines=o3d.utility.Vector2iVector(lines),
        )
        # Color lines based on the first day's color
        line_set.paint_uniform_color(colors[i % len(colors)])
        geometries.append(line_set)

    print("Visualizing...")
    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    # Set environment variable for WSL
    if 'WSL_DISTRO_NAME' in os.environ:
        os.environ['XDG_SESSION_TYPE'] = 'x11'
    visualize_ot_all_days()
