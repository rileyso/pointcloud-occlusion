import argparse
import numpy as np

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
    msig = MixedSignalsExplorer('/mnt/d/Datasets/Mixed_signals', verbose = False)
    # seq_exist_cavs = msig.return_name_cavs_in_seq(chosen_sequence_index)
    # print(f"name CAVs in sequence {chosen_sequence_index}: {seq_exist_cavs}")
    
    seq_labeled_sync_time_ids = msig.return_labeled_sync_time_ids_of_seq(chosen_sequence_index)

    sync_time_idx = seq_labeled_sync_time_ids[labeled_frame_idx]
    
    top_se3_map = msig.top_se3_map
    
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
        print(agent_pc[:5])
        print(f'number of points: {len(agent_pc)}')            
        apply_se3_(top_se3_map @ map_se3_agent, agent_pc)
        print(agent_pc[:5])
        print(f'number of points: {len(agent_pc)}')  
        # add point cloud to visualization
        painter.add_pointclouds_(agent_pc, agent_color[agent_name], voxel_size=0.1)
        
        # add Ego Vehicle to Visualization
        if agent_name == 'laser':
            # convert index to timestamp
            timestamp = msig.return_timestamp_for_query_gt(chosen_sequence_index, sync_time_idx)
            print(f'timestamp: {timestamp}')

            car_box = np.array([
                [0.0, 0.0, -2.2, 1.8, 1.8, 1.8, 0.0]
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
    painter.add_boxes_(gt_boxes_in_top, np.zeros(3))
    
    print(f'number of gt boxes: {len(gt_boxes_in_top)}')

    # ===
    # painter.show(view_points=view_point)
    
    # print(msig.name_agents)
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualise ego vehicle')
    parser.add_argument('--seq_idx', type=int, default=20)
    parser.add_argument('--time_idx', type=int, default=0)
    
    args = parser.parse_args()
    main(args.seq_idx, args.time_idx)
