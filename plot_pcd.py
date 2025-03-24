import open3d as o3d
import argparse

def visualize_pcd(pcd_path):
    '''
    Visualize a point cloud
    pcd_path: path to pcd file
    '''
    pcd = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([pcd])
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd_path', type=str, help='path to pcd file')
    pcd_path = parser.parse_args().pcd_path
    visualize_pcd(pcd_path)