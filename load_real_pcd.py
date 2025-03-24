import open3d as o3d
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import argparse

def load_real_pcd(pcd_path, num_points=5000):
    '''
    Load a real point cloud from a pcd file
    pcd_path: path to pcd file
    num_points: number of points to sample from the point cloud
    '''
    pcd = o3d.io.read_point_cloud(pcd_path)
    return np.asarray(pcd.points)

def transform_pcd(pcd_points, angle, translation):
    '''
    Transform a point cloud by rotating it by an angle and translating it
    pcd_points: point cloud
    angle: angle to rotate the point cloud by
    translation: translation vector
    '''
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])
    pcd_points = pcd_points @ R.T + translation
    return pcd_points
    

def save_pcd_to_csv(pcd_points, save_path):
    '''
    Save a point cloud to a csv file
    pcd_points: point cloud
    save_path: path to save the csv file
    '''
    df = pd.DataFrame(pcd_points, columns=['x', 'y', 'z'])
    df.to_csv(save_path, index=False, header=False)

def main(pcd_path):
    pcd_points = load_real_pcd(pcd_path)
    transformed_pcd_points = transform_pcd(pcd_points, angle=np.pi/4, translation=np.array([10, 24, 32]))
    save_pcd_to_csv(pcd_points, pcd_path.replace('.pcd', '.csv'))
    save_pcd_to_csv(transformed_pcd_points, pcd_path.replace('.pcd', '_transformed.csv'))
    print('pcd converted to csv at path:', pcd_path.replace('.pcd', '.csv'))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcd_path', type=str, help='path to pcd file')
    pcd_path = parser.parse_args().pcd_path
    main(pcd_path)