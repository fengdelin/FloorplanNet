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

torch.set_grad_enabled(False)

def read_graph_pairs(path, file_name):
    sample_img_path= os.path.join(path,"sample_image/")
    img_path = os.path.join(path,"image/")
    graph_path = os.path.join(path,"graph/") 
    sample_graph_path = os.path.join(path, "sample_graph/")
    sample_image = sample_img_path + file_name
    sample_image_name = os.path.splitext(file_name)[0]
    image_name = sample_image_name.split('_')[0]
    images_file = image_name + ".png"
    images= img_path + images_file
    graph_file = image_name + ".csv"
    graph = graph_path + graph_file
    feature_file = image_name + "_feature.csv"
    features = graph_path + feature_file
    sample_graph_file = sample_image_name + ".csv"
    sample_graph = sample_graph_path + sample_graph_file
    sample_feature_file = sample_image_name + "_feature.csv"
    sample_features = sample_graph_path + sample_feature_file
    match_file = sample_image_name + "_match.csv"
    matches = sample_graph_path + match_file
    transformation_file = sample_image_name + ".txt"
    srt = sample_graph_path + transformation_file
    trj_file = sample_img_path + "F3.txt"
    trj_time_file = sample_img_path + "time.txt"
    
    # print("sample graph is:", sample_image_name)
    sample_image = cv2.imread(sample_image, cv2.IMREAD_GRAYSCALE) 
    image = cv2.imread(images, cv2.IMREAD_GRAYSCALE) 
    width, height = sample_image.shape[:2]
    # print("image_size",image.shape[:2])
        
    ##get the edge (self attention)
    edge_prob1 = np.array(np.genfromtxt(graph, delimiter=','))
    edge_prob2 = np.array(np.genfromtxt(sample_graph, delimiter=','))
    if(edge_prob2.ndim==0):
        edge_prob2=np.array([1])
        
    edge_prob1 = np.expand_dims(edge_prob1,0)
    edge_prob2 = np.expand_dims(edge_prob2,0)
    transform_name = srt
    # get the corresponding warped image
    # M = np.array(np.genfromtxt(transform_name, delimiter=','))
    
    ### Read the transformation robot to map
    origin = []
    size = []
    file_srt = open(transform_name,'r')
    lines = file_srt.readlines()
    resolution_ = lines[0]
    resolution = float(resolution_.split(': ')[1])
    # print(resolution)
    origin_ = lines[1]
    origin_ = origin_.split(': ')[1][:-1]
    origin_ = origin_.split(' ')
    for xyz in origin_:
        origin.append(float(xyz))
    origin = np.array(origin)
    size_ = lines[2]
    size_ = size_.split(': ')[1][:-1]
    size_ = size_.split(' ')
    for xyz in size_:
        size.append(float(xyz))
    size = np.array(size)
    M = np.eye(3)   
    M[:,2] = M[:,2] - origin
    M = M / resolution * 2.064516129#2.06036217  #1.66775244
    M[2,2] = 1
    # print(M)
    
    ## read trajectory of different time
    trj_ = np.array(np.genfromtxt(trj_file, delimiter=' '))[:,:3]
    init_time = trj_[0,0]
    trj_time_ = open(trj_time_file,'r')
    lines = trj_time_.readlines()
    trj_idx_ = sample_image_name.split('_')[1]
    trj_idx = int(trj_idx_)
    if trj_idx == 0:
        last_time = init_time
    else:
        last_idx = trj_idx-1
        trj_time_l = lines[last_idx]
        trj_time_l = trj_time_l.split('  ')[0]
        trj_time_l = trj_time_l.split(': ')[1]
        trj_time_l = float(trj_time_l)
        last_time = init_time + trj_time_l
    trj_time_ = lines[trj_idx]
    trj_time_ = trj_time_.split('  ')[0]
    trj_time_ = trj_time_.split(': ')[1]
    trj_time = float(trj_time_)
    end_time = init_time + trj_time
    # print(end_time)
    start_idx = np.argwhere(trj_[:,0]>= last_time)[0][0]
    end_idx = np.argwhere(trj_[:,0]< end_time)[-1][0]
    print(start_idx, end_idx)
    
    trj = trj_[start_idx:end_idx,1:3]
    
    
    
    features1 = np.array(np.genfromtxt(features, delimiter=',')).reshape(-1,29)
    features2 = np.array(np.genfromtxt(sample_features, delimiter=',')).reshape(-1,29)
    kp1, descs1 = features1[:,:2],features1[:,3:29]
    kp2, descs2 = features2[:,:2],features2[:,3:29]

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
    scores1_np = features1[:,2].reshape(1,-1)
    scores2_np = features2[:,2].reshape(1,-1)
    # print("score1_np.shape",scores1_np.shape)      
        

    kp1_np = kp1.reshape(1,-1,2)
    kp2_np = kp2.reshape(1,-1,2)
    # print("reshape kp1:", kp1_np.shape) ###(b=1,n,2)
    descs1 = np.transpose(descs1 / 256.).reshape(1,26,-1)
    descs2 = np.transpose(descs2 / 256.).reshape(1,26,-1)
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
        'file_name': sample_image_name,
        'transform': M,
        'trajectory': trj
        } 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue graph test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--path', type=str, default='/public/home/fengdl/p300/project/iros22/dataset/test_loc',
        help='real time localization path')
    parser.add_argument(
        '--file_name', type=str, default='00000_0.png',
        help='submap image name')
    parser.add_argument(
        '--output_dir', type=str, default='trajectory_nc22',
        help='Directory where to write output frames (If None, no output)')
    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[1024, 1024],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')
    parser.add_argument(
        '--superglue', type=str, default='l9_ee_ep100_m3a_g05_t025_1',
        help='SuperGlue graph weights')
    parser.add_argument(
        '--scores_scale', type=float, default=1.0,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.3,
        help='SuperGlue match threshold')
    parser.add_argument(
    '--fast_viz', action='store_true',
    help='Use faster image visualization based on OpenCV instead of Matplotlib')
    parser.add_argument(
    '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
    help='Visualization file extension. Use pdf for highest-quality.')
    parser.add_argument(
    '--opencv_display', action='store_true',
    help='Visualize via OpenCV before saving output images')
    parser.add_argument(
        '--show_keypoints', action='store_true',
        help='Show the detected keypoints')
    parser.add_argument(
        '--no_display', action='store_true',
        help='Do not display images to screen. Useful if running remotely')
    parser.add_argument(
        '--force_cpu', action='store_true',
        help='Force pytorch to run in CPU mode.')
    parser.add_argument(
    '--eval_method', type=str, default='ours', choices=['ours', 'nn', 'all'],
    help='Visualization file extension. Use pdf for highest-quality.')

    opt = parser.parse_args()
    print(opt)

    if len(opt.resize) == 2 and opt.resize[1] == -1:
        opt.resize = opt.resize[0:1]
    if len(opt.resize) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            opt.resize[0], opt.resize[1]))
    elif len(opt.resize) == 1 and opt.resize[0] > 0:
        print('Will resize max dimension to {}'.format(opt.resize[0]))
    elif len(opt.resize) == 1:
        print('Will not resize images')
    else:
        raise ValueError('Cannot specify more than two integers for --resize')

    if opt.output_dir is not None:
        Path(opt.output_dir).mkdir(exist_ok=True)
        print('==> Will write outputs to {}'.format(opt.output_dir))
        

    # Create a window to display the demo.
    if not opt.no_display:
        cv2.namedWindow('SuperGlue matches', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('SuperGlue matches', (640*2, 480))
    else:
        print('Skipping visualization, will not show a GUI.')

    # Print the keyboard help menu.
    print('==> Keyboard control:\n'
          '\tn: select the current frame as the anchor\n'
          '\te/r: increase/decrease the keypoint confidence threshold\n'
          '\td/f: increase/decrease the match filtering threshold\n'
          '\tk: toggle the visualization of keypoints\n'
          '\tq: quit')
    
    config = {
        'graph': {
            'kpts_score': opt.scores_scale,
        },
        'superglue': {
            'weights': opt.superglue,
            'mode': 'test',
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
        }
    }
    eval_output_dir = Path(opt.output_dir)
    # load data
    pred = read_graph_pairs(opt.path, opt.file_name)

    superglue = SuperGlue(config.get('superglue', {}))

    if torch.cuda.is_available():
        superglue.double().cuda() # make sure it trains on GPU
    else:
        print("### CUDA not available ###")    
    

    for k in pred:
        if k=='image0' or k=='image1':
            pred[k] = pred[k].unsqueeze(0)
            # print(k,pred[k].size())
        elif k != 'file_name' and k!='transform' and k != 'trajectory':
            if type(pred[k]) != torch.Tensor:
                pred[k]= torch.from_numpy(pred[k][0]).cuda()
                if k == 'keypoints0' or k == 'keypoints1':
                    pred[k] = pred[k].unsqueeze(0)
                    pred[k] = pred[k].unsqueeze(0)
                else:
                    pred[k] = pred[k].unsqueeze(1)
                # pred[k] = torch.stack(pred[k]).cuda()
            # print(k,pred[k].shape)


    data = superglue(pred)
    for k, v in pred.items():
        if k=="transform" or k =="trajectory":
            pred[k] = v
        elif k=='descriptors0' or k=='descriptors1':
            pred[k]= v[:,0,:]
        else:
            pred[k] = v[0]
            
    pred = {**pred, **data}
        
    superglue.eval()
    image0, image1 = pred['image0'].cpu().numpy()[0]*255., pred['image1'].cpu().numpy()[0]*255.
    kpts0, kpts1 = pred['keypoints0'].cpu().numpy()[0], pred['keypoints1'].cpu().numpy()[0]
    matches, conf = pred['matches0'].cpu().detach().numpy(), pred['matching_scores0'].cpu().detach().numpy()
    desc0, desc1 = pred['descriptors0'].cpu().detach().numpy(), pred['descriptors1'].cpu().detach().numpy()
    # print(desc0.shape)#(dim,n)
    print(matches)
    image0 = read_image_modified(image0, opt.resize, opt.resize_float)
    image1 = read_image_modified(image1, opt.resize, opt.resize_float)
    # matches = np.array([-1,-6,-5, -10, -5,-10,-5, -10,5,3,-1,-1,-10, -6, 2,-1,-5,-1,-1,-10,-1,-1,-1,-1,-1,-1,-10,-1,-1,-1,1,-1,-1,-10, 0, -10,-1, 4, -1, -1,-1,-1,-1,-5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-10,-1,-1,-1,-1,-1,-1])
    # matches = np.ones(60)
    #matches = np.array([0,-1,1,2,-8,-1,6,-1,-1,-1,-4,-5,-1,-1,-6,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,8])

    # matches = np.array([0,-1,1,2,-8,-1,6,-1,-1,-1,4,5,-1,-1,6,-7,-1,7,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,8,-1,-1,-1,-1,-1])
    # print(matches)
    # matches[11]=4
    valid = matches > -1
    mkpts0 = kpts0[valid]##(N,2)
    mkpts1 = kpts1[matches[valid]]
    mconf = conf[valid]
    viz_path = eval_output_dir / '{}_matches.{}'.format(opt.file_name, opt.viz_extension)
    color = cm.jet(mconf)
    stem = pred['file_name']
    stem =""
    T_robot_map = pred['transform']
    # print("relative transfomation:", T_local)
    m_thresh = superglue.config['match_threshold']
    trj = pred['trajectory']
    trj = trj[::30,:]
    print(trj.shape) #(N,2)
    homo = np.ones(trj.shape[0]).reshape(-1,1)
    trj = np.hstack((trj,homo))
    
    flip_y = np.array([[0,1,0],[1,0,0],[0,0,1]])
    T_robot_map = np.dot(flip_y, T_robot_map)
    trj_robot_map = np.dot(T_robot_map, trj.T).T
    
    #plot trajectory in map
    plt.imshow(image1)
    # plt.gca().invert_yaxis()
    plt.scatter(trj_robot_map[:,0],trj_robot_map[:,1],0.1)
    plt.savefig("map.png")
    
    ## ours method eval
    text = [
        'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
        'Matches: {}'.format(len(mkpts0))
    ]

    if mkpts1.shape[0]<=1:
        H = np.array([0])
    else:
        H, _ = estimate_homo(mkpts1, mkpts0)

            # image0[image0==0]= 200
    if H.shape[0]!=1:
        H_inv = np.linalg.inv(np.vstack((H,np.array([[0,0,1]]))))[:2,:]
            # image1[image1<245]= image1[image1<245]-50
            # image1[image1<=10] = 10
    else:
        H_inv = np.array([0])
        image0[image0==0]= 200

    # print(mkpts0)
    # print(mkpts1)
    # if H is not None:
    make_matching_plot(
        image1, image0, kpts1, kpts0, mkpts1, mkpts0, H, color,
        text, viz_path, stem, stem, opt.show_keypoints,
        opt.fast_viz, opt.opencv_display, 'Matches')
    # print("mkpts shape:",mkpts0.shape,mkpts1.shape)
    # print(H) ##（2，3）
    
    # ##test nearest neighbor matcher
    # pairs, pts0, pts1, dist = nearest_ransac(kpts0,kpts1,desc0,desc1,1)
    # # print(pairs)
    # text_nn = [
    #     'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
    #     'Matches: {}'.format(len(pts0))
    # ]
    # M, precision_n, matching_score_n, success_match_nn, text_nn, err_R_nn, err_t_nn, err_R_auc_nn = record_nn_error(pts0,pts1,kpts1,T_gt,success_match_nn,text_nn)
    # pose_error_nn = np.maximum(err_R_auc_nn, err_t_nn)
    # pose_errors_nn.append(pose_error_nn)
    # precision_nn.append(precision_n) 
    # matching_score_nn.append(matching_score_n)
    # # print(dist)
        # color_nn = cm.jet(10/dist)
        # make_matching_plot(
        #     image0, image1, kpts0, kpts1, pts0, pts1, M, color_nn,
        #     text_nn, viz_nn_path, stem, stem, opt.show_keypoints,
        #     opt.fast_viz, opt.opencv_display, 'Matches')
    #### localization visualization
    if H.shape[0]==1:
        T_map_fp = estimation_one_area(kpts0, kpts1, desc0, desc1, matches)
    else:
        T_map_fp = np.vstack((H,np.array([0,0,1])))
    print(T_map_fp)
    np.savetxt(eval_output_dir /'T_map_fp_{}.csv'.format(opt.file_name),T_map_fp)
    T_robot_fp = np.dot(np.linalg.inv(T_map_fp), T_robot_map)
    print(T_robot_fp)
    np.savetxt(eval_output_dir /'T_robot_fp_{}.csv'.format(opt.file_name),T_robot_fp)
    
    trj_fp = np.dot(T_robot_fp, trj.T).T
    plt.clf()
    plt.imshow(image0)
    # plt.gca().invert_yaxis()
    plt.scatter(trj_fp[:,0],trj_fp[:,1],0.1)
    plt.savefig("trajectory_fp.png")
    # plt.figure(1,figsize=(6,6))
    # position = np.dot(T_robot_fp, np.array([0,0,1]))
    # position_x = np.dot(T_robot_fp, np.array([1,0,1]))
    # position_y = np.dot(T_robot_fp, np.array([0,1,1]))
    
    # # plt.xlim(-3,3)
    # # plt.ylim(-3,3)
    # plt.quiver(position[0], position[1], position_x[0]-position[0], position_x[1]-position[1],
    #            angles='xy',scale_units='xy',scale=0.5)
    # plt.quiver(position[0], position[1], position_y[0]-position[0], position_y[1]-position[1],
    #            angles='xy',scale_units='xy',scale=0.5)
    
    
    