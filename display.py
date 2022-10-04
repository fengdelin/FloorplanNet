#! /usr/bin/env python3
#
from pathlib import Path
import os
import argparse
from pickle import FALSE
from re import M
import cv2
import matplotlib.cm as cm
import torch
import numpy as np
from load_graph import SparseDataset
from models.utils import (make_matching_plot,read_image_modified,
                          estimate_homo, estimation_one_area)
from models.superglue import SuperGlue
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue graph test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--path', type=str, default='/public/home/fengdl/p300/project/iros22/dataset/show',
        help='real time localization path')
    
    opt = parser.parse_args()
    print(opt)
    
    dir_path = Path(opt.path)
    
    sample_img_path= os.path.join(dir_path,"sample_image/")
    img_path = os.path.join(dir_path,"image/")
    floor_plan_name = '00000.png'
    floor_plan  = cv2.imread(img_path+floor_plan_name)
    tmp = cv2.cvtColor(floor_plan, cv2.COLOR_BGR2GRAY) 
    _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
    b, g, r = cv2.split(floor_plan)
    rgba = [b, g, r, alpha]
    dst = cv2.merge(rgba, 4)
    fig = plt.gcf()
    # fig.set_size_inches()
    plt.axis('off')
    # sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    # dst = cv2.filter2D(dst, cv2.CV_32F, sharpen_op)
    plt.imshow(dst)
    plt.savefig("trajectory_fp_0.png",dpi=200)
    plt.imshow(dst,alpha=0.5)
    plt.imshow(dst,alpha=0.5)
    plt.imshow(dst,alpha=0.5)
    
    
    i=0
    colors = [6,8,10,7]
    ccs = ['y','m','c','b']
    for f in os.listdir(sample_img_path):
        print(colors[i])
        
        file_name = f
        T_fp_map_file = 'T_map_fp_'+ file_name + ".csv"
        T_fp_map_file = os.path.join(dir_path, T_fp_map_file)
        T_robot_fp_flie = 'T_robot_fp_' + file_name + '.csv'
        T_robot_fp_flie = os.path.join(dir_path, T_robot_fp_flie)
        trj_file = 'trj_' + file_name +'.csv'
        trj_file = os.path.join(dir_path, trj_file)
        
        image_file = sample_img_path + f
        sample_image = cv2.imread(image_file)#, cv2.IMREAD_GRAYSCALE
        image_name = f.split('_')[0]
        
        T_fp_map = np.loadtxt(T_fp_map_file)
        T_robot_fp = np.loadtxt(T_robot_fp_flie)
        trj = np.loadtxt(trj_file)
        T_map_fp = np.linalg.inv(T_fp_map)
        
        # map_in_fp = np.dot(T_fp_map[:2,:2],sample_image)+ T_map_fp[:2,2]
        if(image_name != '00000'):
            symmetry = np.array([[-1,0,1024],[0,-1,1024],[0,0,1]])
            T_robot_fp = np.dot(symmetry, T_robot_fp)
            T_map_fp = np.dot(symmetry,T_map_fp)
            print(image_name)
        else:
            print(image_name)
            
        map_in_fp = cv2.warpAffine(sample_image,T_map_fp[:2,:],(1024,1024))
        tmp = cv2.cvtColor(map_in_fp, cv2.COLOR_BGR2GRAY) 
        _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
        c_ = cv2.applyColorMap(map_in_fp,colors[i])
        b, g, r = cv2.split(c_)
        rgba = [b, g, r, alpha]
        dst = cv2.merge(rgba, 4)
        cv2.imwrite("ttt.png",dst)
        # plt.clf()
        plt.imshow(dst,cmap='RdBu')#,alpha=0.8)           
            
        # flip_y = np.array([[0,1,0],[1,0,0],[0,0,1]])
        # T_robot_map = np.dot(flip_y, T_robot_map)
        # T_robot_fp = np.dot(np.linalg.inv(T_map_fp), T_robot_map)
        # trj_robot_map = np.dot(T_robot_map, trj.T).T
    
        trj_fp = np.dot(T_robot_fp, trj.T).T
        # plt.clf()
        plt.scatter(trj_fp[:,0],trj_fp[:,1],c=ccs[i],s=0.1)
        
        i = i+1
        plt.savefig("trajectory_fp_%d.png"%i,dpi=200)

    
    
    plt.savefig("trajectory_fp.png",dpi=200)