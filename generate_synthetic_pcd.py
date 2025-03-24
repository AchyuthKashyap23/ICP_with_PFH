import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse

def generate_synthetic_pcd(num_points=1000):
    '''
    Generate a synthetic point cloud
    num_points: number of points in the point cloud
    '''
    pc = []
    for i in range(0,num_points):
        x = np.random.uniform(-1,1)
        y = np.random.uniform(-1,1)
        z = np.random.uniform(-1,1)
        z = x + y + z
        pc.append(np.array([x,y,z]))
    return pc

def save_pcd_to_csv(pcd_points, save_path):
    '''
    Save a point cloud to a csv file
    pcd_points: point cloud
    save_path: path to save the csv
    '''
    df = pd.DataFrame(pcd_points, columns=['x', 'y', 'z'])
    df.to_csv(save_path, index=False, header=False)

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

def main(save_path):
    pcd_points = generate_synthetic_pcd()
    transformed_pcd_points = transform_pcd(pcd_points, angle=np.pi/2, translation=np.array([0.2, 0.1, 0.2]))
    save_pcd_to_csv(pcd_points, save_path)
    save_pcd_to_csv(transformed_pcd_points, save_path.replace('.csv', '_transformed.csv'))
    print('pcd converted to csv at path:', save_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, help='path to save csv file')
    save_path = parser.parse_args().save_path
    main(save_path)