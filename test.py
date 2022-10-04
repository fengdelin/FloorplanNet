#! /usr/bin/env python3
#
from pathlib import Path
import argparse
from pickle import FALSE
import cv2
import matplotlib.cm as cm
import torch
import numpy as np
from load_graph import SparseDataset
from models.utils import (compute_match_score, compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, read_image_modified,
                          compute_geometry_match_score, estimate_homo,
                          nearest_ransac,record_error,record_nn_error)
from models.superglue import SuperGlue
torch.set_grad_enabled(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SuperGlue graph test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--test_path', type=str, default='/public/home/fengdl/p300/project/iros22/dataset/test_door1',
        help='test input file path ')
    parser.add_argument(
        '--output_dir', type=str, default='test_result',
        help='Directory where to write output frames (If None, no output)')

    parser.add_argument(
        '--image_glob', type=str, nargs='+', default=['*.png', '*.jpg', '*.jpeg'],
        help='Glob if a directory of images is specified')
    parser.add_argument(
        '--skip', type=int, default=1,
        help='Images to skip if input is a movie or directory')
    parser.add_argument(
        '--max_length', type=int, default=1000000,
        help='Maximum length if input is a movie or directory')
    parser.add_argument(
        '--resize', type=int, nargs='+', default=[1024, 1024],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--resize_float', action='store_true',
        help='Resize the image after casting uint8 to float')
    parser.add_argument(
        '--superglue', type=str, default='v1',
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
    # load training data
    test_set = SparseDataset(opt.test_path)
    # train_set = SparseDataset(opt.train_path, opt.max_keypoints)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=False, batch_size=1, drop_last=True)

    superglue = SuperGlue(config.get('superglue', {}))

    if torch.cuda.is_available():
        superglue.double().cuda() # make sure it trains on GPU
    else:
        print("### CUDA not available ###")
        
    ##average precision
    precision_average = []
    matching_score_average = []
    success_match = []
    pose_errors = []
    
    num_keypoints0 = []
    num_keypoints1 = []
    num_match = []
    
    precision_nn = []
    matching_score_nn = []
    success_match_nn = []
    pose_errors_nn = []
    

    for i, pred in enumerate(test_loader):
        for k in pred:
            if k != 'file_name' and k!='image0' and k!='image1' and k!='transform'and k!='matches':
            # if k != 'file_name' and k!='image0' and k!='image1':
                if type(pred[k]) != torch.Tensor:
                    # print(pred['matches'])
                    pred[k] = torch.stack(pred[k]).cuda()

        data = superglue(pred)
        for k, v in pred.items():
            if k=="transform" or k=='matches':
                pred[k] = v
            elif k=='descriptors0' or k=='descriptors1':
                pred[k]= v[:,0,:]
            else:
                pred[k] = v[0]
        # print(pred['matches'])
            
        pred = {**pred, **data}

        if pred['skip_train'] == True: # image has no keypoint
            continue
        
        print("This is %d-th graph"%i)
        superglue.eval()
        image0, image1 = pred['image0'].cpu().numpy()[0]*255., pred['image1'].cpu().numpy()[0]*255.
        kpts0, kpts1 = pred['keypoints0'].cpu().numpy()[0], pred['keypoints1'].cpu().numpy()[0]
        matches, conf = pred['matches0'].cpu().detach().numpy(), pred['matching_scores0'].cpu().detach().numpy()
        desc0, desc1 = pred['descriptors0'].cpu().detach().numpy(), pred['descriptors1'].cpu().detach().numpy()
        # print(desc0.shape)#(dim,n)
        image0 = read_image_modified(image0, opt.resize, opt.resize_float)
        image1 = read_image_modified(image1, opt.resize, opt.resize_float)
        matches = np.array([-1,-6,-5, -10, -5,-10,-5, -10,5,3,-1,-1,-10, -6, 2,-1,-5,-1,-1,-10,-1,-1,-1,-1,-1,-1,-10,-1,-1,-1,1,-1,-1,-10, 0, -10,-1, 4, -1, -1,-1,-1,-1,-5,-1,-1,-1,-1,-1,-1,-1,-1,-1,-10,-1,-1,-1,-1,-1,-1])

        valid = matches > -1
        mkpts0 = kpts0[valid]##(N,2)
        mkpts1 = kpts1[matches[valid]]
        mconf = conf[valid]
        viz_path = eval_output_dir / '{}_matches.{}'.format(str(i), opt.viz_extension)
        eval_output_nn_dir = Path(opt.output_dir + '_NN')
        viz_nn_path =  eval_output_nn_dir / '{}_matches.{}'.format(str(i), opt.viz_extension)
        color = cm.jet(mconf)
        # stem = pred['file_name']
        stem = ""
        T_gt = pred['transform'][0].numpy()
        # print("transfomation ground truth:", T_gt)
        matches_gt = pred['matches'][0].numpy()
        # print("transfomation ground truth:", matches_gt)
        matches1 = pred['matches1'].cpu().detach().numpy()
        # matches1_valid=matches1[matches1 > -1]
        m_thresh = superglue.config['match_threshold']
        num_keypoints0.append(kpts0.shape[0])
        num_keypoints1.append(kpts1.shape[0])
        num_match.append(mkpts1.shape[0])
        
        ## ours method eval
        text = [
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(mkpts0))
        ]

        H, precision, matching, success_match, text, err_R, err_t, err_R_auc = record_error(mkpts0,mkpts1,kpts1,matches1,matches_gt,T_gt,success_match,text)
        pose_error = np.maximum(err_R_auc, err_t)
        pose_errors.append(pose_error)
        precision_average.append(precision)
        matching_score_average.append(matching)
        
        # image0[image0==0]= 200
        if H.shape[0]!=1:
            H_inv = np.linalg.inv(np.vstack((H,np.array([[0,0,1]]))))[:2,:]
            # image1[image1<245]= image1[image1<245]-50
            # image1[image1<=10] = 10
        else:
            H_inv = np.array([0])
            image0[image0==0]= 200
            
        make_matching_plot(
        image1, image0, kpts1, kpts0, mkpts1, mkpts0, H_inv, color,
            text, viz_path, stem, stem, opt.show_keypoints,
            opt.fast_viz, opt.opencv_display, 'Matches')
        # # if H is not None:
        # make_matching_plot(
        #     image0, image1, kpts0, kpts1, mkpts0, mkpts1, H, color,
        #     text, viz_path, stem, stem, opt.show_keypoints,
        #     opt.fast_viz, opt.opencv_display, 'Matches')
        # # print("mkpts shape:",mkpts0.shape,mkpts1.shape)
        # # print("precision update:",np.mean(precision_average),"matching_score update", np.mean(matching_score_average))
        # # print("graph match sucessus rate:",np.mean(success_match))
        
        
        ##test nearest neighbor matcher
        pairs, pts0, pts1, dist = nearest_ransac(kpts0,kpts1,desc0,desc1,1)
        # print(pairs)
        text_nn = [
            'Keypoints: {}:{}'.format(len(kpts0), len(kpts1)),
            'Matches: {}'.format(len(pts0))
        ]
        M, precision_n, matching_score_n, success_match_nn, text_nn, err_R_nn, err_t_nn, err_R_auc_nn = record_nn_error(pts0,pts1,kpts1,T_gt,success_match_nn,text_nn)
        pose_error_nn = np.maximum(err_R_auc_nn, err_t_nn)
        pose_errors_nn.append(pose_error_nn)
        precision_nn.append(precision_n) 
        matching_score_nn.append(matching_score_n)
        # print(dist)
        color_nn = cm.jet(10/dist)
        # make_matching_plot(
        #     image0, image1, kpts0, kpts1, pts0, pts1, M, color_nn,
        #     text_nn, viz_nn_path, stem, stem, opt.show_keypoints,
        #     opt.fast_viz, opt.opencv_display, 'Matches')
    
    thresholds = [1, 3, 5, 10, 20]
    aucs = pose_auc(pose_errors, thresholds)
    aucs_nn = pose_auc(pose_errors_nn, thresholds)
    # print(np.array(num_keypoints).shape,np.array(precision_average).shape,np.array(success_match).shape)
    print("precision update:",np.mean(precision_average),"matching_score update", np.mean(matching_score_average))
    print("graph match sucessus rate:",np.mean(success_match))
    result=np.vstack((np.array(num_keypoints0),np.array(num_keypoints1),np.array(num_match),np.array(precision_average),np.array(success_match)))
    np.savetxt("%s.csv"%opt.superglue,result)
    print("auc of ours:", aucs)
    print("auc of ours:", aucs, "acu of nn:", aucs_nn)
    
    print("precision NN update:",np.mean(precision_nn),"matching_score NN update", np.mean(matching_score_nn))
    print("graph match NN sucessus rate:",np.mean(success_match_nn))