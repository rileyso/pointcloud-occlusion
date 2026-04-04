import argparse
import numpy as np
import json

from mixedsignals.mixed_signals import MixedSignalsExplorer
from mixedsignals.utils.mixed_signals_utils import AGENT_COLOR as agent_color
from mixedsignals.utils.mixed_signals_utils import VIEW_POINT as view_point
from mixedsignals.utils.geometry import apply_se3_, apply_se3
from mixedsignals.utils.o3d_viz_utils import PointPainter
import open3d as o3d
from mixedsignals.utils.bbox_utils import get_boxes_vertices_coord


def make_lineset_from_vertices(vertices, color=(1.0, 0.0, 0.0)):
    lines = [
        [0,1], [1,2], [2,3], [3,0],
        [4,5], [5,6], [6,7], [7,4],
        [0,4], [1,5], [2,6], [3,7]
    ]

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(vertices)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(color), (len(lines), 1))
    )
    return line_set

def main(chosen_sequence_index = 30, labeled_frame_idx = 100):
    
    print(f'showing ego pc of sequence {chosen_sequence_index} in TOP frame')
    # msig = MixedSignalsExplorer('/mnt/d/Datasets/mixed-signals-mini', verbose = False)
    msig = MixedSignalsExplorer('/mnt/d/Datasets/Mixed_signals', verbose = False)
    # seq_exist_cavs = msig.return_name_cavs_in_seq(chosen_sequence_index)
    # print(f"name CAVs in sequence {chosen_sequence_index}: {seq_exist_cavs}")
    
    seq_labeled_sync_time_ids = msig.return_labeled_sync_time_ids_of_seq(chosen_sequence_index)

    sync_time_idx = seq_labeled_sync_time_ids[labeled_frame_idx]
    
    top_se3_map = msig.top_se3_map
    # ['laser','top','dome']
    painter = PointPainter()
    # ===
    for agent_name in ['laser']:
        # get point cloud in agent's body frame
        agent_pc, agent_timestamp = \
            msig.return_agent_point_cloud(chosen_sequence_index, agent_name, sync_time_idx)
        # transform point cloud from agent's body frame to `TOP`
        map_se3_agent = msig.return_map_se3_agent(chosen_sequence_index, 
                                                    agent_name, 
                                                    agent_timestamp)
        apply_se3_(top_se3_map @ map_se3_agent, agent_pc)
        # add point cloud to visualization
        painter.add_pointclouds_(agent_pc, agent_color[agent_name], voxel_size=0.1)
        
        # add Ego Vehicle to Visualization
        if agent_name == 'laser':
            # convert index to timestamp
            timestamp = msig.return_timestamp_for_query_gt(chosen_sequence_index, sync_time_idx)
            # print(f'timestamp: {timestamp}')
            saved_laser_pc = np.array(agent_pc, copy=True)
            car_box = np.array([
                [0.0, 0.0, -2.2, 0.7, 0.7, 0.7, 0.0]
            ], dtype=float)
            # xyz lwh yaw
            
            # transform box from laser frame -> map frame
            apply_se3_(top_se3_map @ map_se3_agent, car_box)
            painter.add_boxes_(
                car_box,
                colors=np.array([1.0, 0.0, 1.0]),
                edge_thickness=0.06
            )
            
        # 0-coord to visualisation
        painter.add_boxes_(
            np.array([
            [0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0]
        ], dtype=float),
            colors=np.array([1.0, 0.0, 0.0]),
            edge_thickness=0.06
        )
                    
    # === 
    # get annotation in `MAP` frame
    gt_boxes_in_map = msig.return_gt_boxes_in_map(chosen_sequence_index, sync_time_idx)
    gt_boxes_in_top = apply_se3(top_se3_map, boxes_=gt_boxes_in_map)

    # add gt_boxes to visualization
    # print(gt_boxes_in_top[0])
    # print(f'number of gt boxes: {len(gt_boxes_in_top)}')
    
    # painter.add_boxes_(gt_boxes_in_top, np.zeros(3))
    
    # box masks
    def points_in_box_top_frame(points_xyz: np.ndarray, box: np.ndarray) -> np.ndarray:
        """
        points_xyz: (N, 3) points already in TOP frame
        box: [cx, cy, cz, dx, dy, dz, yaw, ...] also already in TOP frame
        
        returns: boolean mask of shape (N,)
        """
        cx, cy, cz, dx, dy, dz, yaw = box[:7]

        # move box centre to origin
        shifted = points_xyz - np.array([cx, cy, cz])

        # rotate points into the box's local axes
        c = np.cos(yaw)
        s = np.sin(yaw)

        x_local =  c * shifted[:, 0] + s * shifted[:, 1]
        y_local = -s * shifted[:, 0] + c * shifted[:, 1]
        z_local = shifted[:, 2]

        mask = (
            (np.abs(x_local) <= dx / 2) &
            (np.abs(y_local) <= dy / 2) &
            (np.abs(z_local) <= dz / 2)
        )
        return mask
    
    
    points_xyz = saved_laser_pc[:, :3]
    colour = np.zeros(3)
    # uncovered = []
    ego_points = np.array([])
    output = {
        "timestamp": float(timestamp),
        "frame_index": int(labeled_frame_idx),
        "gt_boxes": []
    }
    for i, box in enumerate(gt_boxes_in_top):
        mask = points_in_box_top_frame(points_xyz, box)
        print(f"box {i}, class {box[7]}, points inside: {mask.sum()}")
        ego_points = np.append(ego_points, mask.sum())
        # if mask.sum() == 0:
        #     uncovered.append(i)
        output["gt_boxes"].append({
            "gt_box_index": int(i),
            "ego_point_cloud": int(mask.sum()),
            "position": [float(x) for x in box[:3]],
            "dimensions": [float(x) for x in box[3:6]],
        })
    

    # gt_boxes_in_top = np.delete(gt_boxes_in_top, uncovered, axis=0)
    # ego_points = np.delete(ego_points, uncovered, axis=0)
    painter.add_boxes_(gt_boxes_in_top, np.zeros(3), edge_thickness=0.03, ego_points=ego_points)
    
    json_string = json.dumps(output)
    print(json_string)
    # ===
    painter.show(view_points=view_point)

        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualise ego vehicle')
    parser.add_argument('--seq_idx', type=int, default=20)
    parser.add_argument('--time_idx', type=int, default=0)
    
    args = parser.parse_args()
    main(args.seq_idx, args.time_idx)