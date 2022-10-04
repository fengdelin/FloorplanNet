import numpy as np
import torch
import os
import cv2
import math
import datetime

from scipy.spatial.distance import cdist
from torch.utils.data import Dataset
import os.path

class SparseDataset(Dataset):
    """Sparse correspondences dataset."""

    def __init__(self, train_path):
        sample_img_path= os.path.join(train_path,"sample_image/")
        img_path = os.path.join(train_path,"image/")
        graph_path = os.path.join(train_path,"graph/") 
        sample_graph_path = os.path.join(train_path, "sample_graph/")
        self.images = []
        self.graph = []
        self.features = []
        self.sample_images = []
        self.sample_graph = []
        self.sample_features = []
        self.matches = []
        self.srt = []
        for f in os.listdir(sample_img_path):
            self.sample_images += [sample_img_path + f]
            sample_image_name = os.path.splitext(f)[0]
            image_name = sample_image_name.split('_')[0]
            images_file = image_name + ".png"
            self.images += [img_path + images_file]
            graph_file = image_name + ".csv"
            self.graph += [graph_path + graph_file]
            feature_file = image_name + "_feature.csv"
            self.features += [graph_path + feature_file]
            # sample_images_file = sample_image_name + ".png"
            # self.sample_images += [img_path + sample_images_file]
            sample_graph_file = sample_image_name + ".csv"
            self.sample_graph += [sample_graph_path + sample_graph_file]
            sample_feature_file = sample_image_name + "_feature.csv"
            self.sample_features += [sample_graph_path + sample_feature_file]
            match_file = sample_image_name + "_match.csv"
            self.matches += [sample_graph_path + match_file]
            transformation_file = sample_image_name + "_transform.csv"
            self.srt += [sample_graph_path + transformation_file]

    def __len__(self):
        return len(self.sample_images)

    def __getitem__(self, idx):
        sample_image_name = self.sample_images[idx]
        # print("sample graph is:", sample_image_name)
        sample_image = cv2.imread(sample_image_name, cv2.IMREAD_GRAYSCALE) 
        image_name = self.images[idx]
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE) 
        width, height = sample_image.shape[:2]
        # print("image_size",image.shape[:2])
        
        ##get the edge (self attention)
        edge_prob1 = np.array(np.genfromtxt(self.graph[idx], delimiter=','))
        edge_prob2 = np.array(np.genfromtxt(self.sample_graph[idx], delimiter=','))
        if(edge_prob2.ndim==0):
            edge_prob2=np.array([1])
        # edge_prob1 = np.array(np.genfromtxt(self.graph[idx], delimiter=','),dtype=np.bool)
        # edge_prob2 = np.array(np.genfromtxt(self.sample_graph[idx], delimiter=','),dtype=np.bool)
        # if(edge_prob2.ndim==0):
        #     edge_prob2=np.array([True],dtype=np.bool)
        
        transform_name = self.srt[idx]
        # get the corresponding warped image
        M = np.array(np.genfromtxt(transform_name, delimiter=','))
        # print(M)
        features1 = np.array(np.genfromtxt(self.features[idx], delimiter=',')).reshape(-1,29)
        features2 = np.array(np.genfromtxt(self.sample_features[idx], delimiter=',')).reshape(-1,29)
        kp1, descs1 = features1[:,:2],features1[:,3:29]
        kp2, descs2 = features2[:,:2],features2[:,3:29]
        # features1 = np.array(np.genfromtxt(self.features[idx], delimiter=',')).reshape(-1,629)
        # features2 = np.array(np.genfromtxt(self.sample_features[idx], delimiter=',')).reshape(-1,629)
        # kp1, descs1 = features1[:,:2],np.hstack((features1[:,3:27],features1[:,627:]))
        # kp2, descs2 = features2[:,:2],np.hstack((features2[:,3:27],features2[:,627:]))

        # print("kp1.shape", kp1.shape) ##(n) (tuple(x,y))
        # print("kp2.shape", kp2.shape) ##(n) (tuple(x,y))
        

        # skip this image pair if no keypoints detected in image
        if kp1.shape[0] <1 or kp2.shape[0] <1:
            return{
                'keypoints0': torch.zeros([0, 0, 2], dtype=torch.double),
                'keypoints1': torch.zeros([0, 0, 2], dtype=torch.double),
                'descriptors0': torch.zeros([0, 2], dtype=torch.double),
                'descriptors1': torch.zeros([0, 2], dtype=torch.double),
                'image0': image,
                'image1': sample_image,
                'file_name': sample_image_name
            } 

        # confidence of each key point
        scores1_np = features1[:,2]
        scores2_np = features2[:,2]
        # print("score1_np.shape",scores1_np.shape)
        
        matches = self.matches[idx]
        
        mn = np.array(np.genfromtxt(matches, delimiter=','),dtype=np.int)
        if(mn.ndim==1):
            MN = mn.reshape(-1,1)
        else:
            MN = mn
        mn1 = MN[0,:]
        mn2 = MN[1,:]       
        # obtain the matching matrix of the image pair
        missing1 = np.setdiff1d(np.arange(kp1.shape[0]), mn1)
        missing2 = np.setdiff1d(np.arange(kp2.shape[0]), mn2)
        # print(missing2)
        MN2 = np.concatenate([missing1[np.newaxis, :], (kp2.shape[0]) * np.ones((1, len(missing1)), dtype=np.int64)])
        # print("MN2.shape:", MN2.shape)
        MN3 = np.concatenate([(len(kp1)) * np.ones((1, len(missing2)), dtype=np.int64), missing2[np.newaxis, :]])
        # print("MN3.shape:", MN3)
        all_matches = np.concatenate([MN, MN2, MN3], axis=1)
        # print("matches:", all_matches)
        

        kp1_np = kp1.reshape(1,-1,2)
        kp2_np = kp2.reshape(1,-1,2)
        # print("reshape kp1:", kp1_np.shape) ###(b=1,n,2)
        descs1 = np.transpose(descs1 / 256.)
        descs2 = np.transpose(descs2 / 256.)
        # print("reshape des1:", descs1.shape) ###(128,n)
        # print("reshape des2:", descs2.shape) ###(128,m)
        
        # print('keypoints0', kp1_np.shape, 'keypoints0',kp2_np.shape,"new image size:", image.shape[:2],"all_matches", all_matches.shape)

        image = torch.from_numpy(image/255.).double()[None].cuda()
        sample_image = torch.from_numpy(sample_image/255.).double()[None].cuda()

        return{
            'keypoints0': list(kp1_np),
            'keypoints1': list(kp2_np),
            'descriptors0': list(descs1),
            'descriptors1': list(descs2),
            'scores0': list(scores1_np),
            'scores1': list(scores2_np),
            'edge_prob1': list(edge_prob1),
            'edge_prob2': list(edge_prob2),
            'image0': image,
            'image1': sample_image,
            'all_matches': list(all_matches),
            'file_name': sample_image_name,
            'transform': M,
            'matches': MN
        } 
