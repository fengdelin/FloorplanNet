# %BANNER_BEGIN%
# ---------------------------------------------------------------------
# %COPYRIGHT_BEGIN%
#
#  Magic Leap, Inc. ("COMPANY") CONFIDENTIAL
#
#  Unpublished Copyright (c) 2020
#  Magic Leap, Inc., All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains the property
# of COMPANY. The intellectual and technical concepts contained herein
# are proprietary to COMPANY and may be covered by U.S. and Foreign
# Patents, patents in process, and are protected by trade secret or
# copyright law.  Dissemination of this information or reproduction of
# this material is strictly forbidden unless prior written permission is
# obtained from COMPANY.  Access to the source code contained herein is
# hereby forbidden to anyone except current COMPANY employees, managers
# or contractors who have executed Confidentiality and Non-disclosure
# agreements explicitly covering such access.
#
# The copyright notice above does not evidence any actual or intended
# publication or disclosure  of  this source code, which includes
# information that is confidential and/or proprietary, and is a trade
# secret, of  COMPANY.   ANY REPRODUCTION, MODIFICATION, DISTRIBUTION,
# PUBLIC  PERFORMANCE, OR PUBLIC DISPLAY OF OR THROUGH USE  OF THIS
# SOURCE CODE  WITHOUT THE EXPRESS WRITTEN CONSENT OF COMPANY IS
# STRICTLY PROHIBITED, AND IN VIOLATION OF APPLICABLE LAWS AND
# INTERNATIONAL TREATIES.  THE RECEIPT OR POSSESSION OF  THIS SOURCE
# CODE AND/OR RELATED INFORMATION DOES NOT CONVEY OR IMPLY ANY RIGHTS
# TO REPRODUCE, DISCLOSE OR DISTRIBUTE ITS CONTENTS, OR TO MANUFACTURE,
# USE, OR SELL ANYTHING THAT IT  MAY DESCRIBE, IN WHOLE OR IN PART.
#
# %COPYRIGHT_END%
# ----------------------------------------------------------------------
# %AUTHORS_BEGIN%
#
#  Originating Authors: Paul-Edouard Sarlin
#                       Daniel DeTone
#                       Tomasz Malisiewicz
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from hashlib import sha1
from pathlib import Path
import time
from collections import OrderedDict
from threading import Thread
from cv2 import RANSAC
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
import matplotlib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pandas as pd
matplotlib.use('Agg')


class AverageTimer:
    """ Class to help manage printing simple timing of code execution. """

    def __init__(self, smoothing=0.3, newline=False):
        self.smoothing = smoothing
        self.newline = newline
        self.times = OrderedDict()
        self.will_print = OrderedDict()
        self.reset()

    def reset(self):
        now = time.time()
        self.start = now
        self.last_time = now
        for name in self.will_print:
            self.will_print[name] = False

    def update(self, name='default'):
        now = time.time()
        dt = now - self.last_time
        if name in self.times:
            dt = self.smoothing * dt + (1 - self.smoothing) * self.times[name]
        self.times[name] = dt
        self.will_print[name] = True
        self.last_time = now

    def print(self, text='Timer'):
        total = 0.
        print('[{}]'.format(text), end=' ')
        for key in self.times:
            val = self.times[key]
            if self.will_print[key]:
                print('%s=%.3f' % (key, val), end=' ')
                total += val
        print('total=%.3f sec {%.1f FPS}' % (total, 1./total), end=' ')
        if self.newline:
            print(flush=True)
        else:
            print(end='\r', flush=True)
        self.reset()


# --- PREPROCESSING ---

def process_resize(w, h, resize):
    assert(len(resize) > 0 and len(resize) <= 2)
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w*scale)), int(round(h*scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print('Warning: input resolution is very small, results may vary')
    elif max(w_new, h_new) > 2000:
        print('Warning: input resolution is very large, results may vary')

    return w_new, h_new


def frame2tensor(frame):
    return torch.from_numpy(frame/255.).float()[None, None].cuda()


def read_image(path, resize, rotation, resize_float):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]

    inp = frame2tensor(image)
    return image, inp, scales



def read_image_modified(image, resize, resize_float):
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')
    return image
# --- GEOMETRY ---


def estimate_pose(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    if len(kpts0) < 3:
        return None

    f_mean = np.mean([K0[0, 0], K1[1, 1], K0[0, 0], K1[1, 1]])
    norm_thresh = thresh / f_mean

    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]

    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, np.eye(3), threshold=norm_thresh, prob=conf,
        method=cv2.RANSAC)

    assert E is not None

    best_num_inliers = 0
    ret = None
    for _E in np.split(E, len(E) / 3):
        n, R, t, mask_new = cv2.recoverPose(
            _E, kpts0, kpts1, np.eye(3), 1e9, mask=mask)
        if n > best_num_inliers:
            best_num_inliers = n
            ret = (R, t[:, 0], mask.ravel() > 0)
    return ret


def rotate_intrinsics(K, image_shape, rot):
    """image_shape is the shape of the image after rotation"""
    assert rot <= 3
    h, w = image_shape[:2][::-1 if (rot % 2) else 1]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    rot = rot % 4
    if rot == 1:
        return np.array([[fy, 0., cy],
                         [0., fx, w-1-cx],
                         [0., 0., 1.]], dtype=K.dtype)
    elif rot == 2:
        return np.array([[fx, 0., w-1-cx],
                         [0., fy, h-1-cy],
                         [0., 0., 1.]], dtype=K.dtype)
    else:  # if rot == 3:
        return np.array([[fy, 0., h-1-cy],
                         [0., fx, cx],
                         [0., 0., 1.]], dtype=K.dtype)


def rotate_pose_inplane(i_T_w, rot):
    rotation_matrices = [
        np.array([[np.cos(r), -np.sin(r), 0., 0.],
                  [np.sin(r), np.cos(r), 0., 0.],
                  [0., 0., 1., 0.],
                  [0., 0., 0., 1.]], dtype=np.float32)
        for r in [np.deg2rad(d) for d in (0, 270, 180, 90)]
    ]
    return np.dot(rotation_matrices[rot], i_T_w)


def scale_intrinsics(K, scales):
    scales = np.diag([1./scales[0], 1./scales[1], 1.])
    return np.dot(scales, K)


def to_homogeneous(points):
    return np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)


def compute_epipolar_error(kpts0, kpts1, T_0to1, K0, K1):
    kpts0 = (kpts0 - K0[[0, 1], [2, 2]][None]) / K0[[0, 1], [0, 1]][None]
    kpts1 = (kpts1 - K1[[0, 1], [2, 2]][None]) / K1[[0, 1], [0, 1]][None]
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)

    t0, t1, t2 = T_0to1[:3, 3]
    t_skew = np.array([
        [0, -t2, t1],
        [t2, 0, -t0],
        [-t1, t0, 0]
    ])
    E = t_skew @ T_0to1[:3, :3]

    Ep0 = kpts0 @ E.T  # N x 3
    p1Ep0 = np.sum(kpts1 * Ep0, -1)  # N
    Etp1 = kpts1 @ E  # N x 3
    d = p1Ep0**2 * (1.0 / (Ep0[:, 0]**2 + Ep0[:, 1]**2)
                    + 1.0 / (Etp1[:, 0]**2 + Etp1[:, 1]**2))
    return d


def deg_error_mat(R1, R2):
    cos = (np.trace(np.dot(R2.T, R1)) - 1) / 2
    cos = np.clip(cos, -1., 1.)  # numercial errors can make it out of bounds
    return np.rad2deg(np.abs(np.arccos(cos)))


# def angle_error_vec(v1, v2):
#     n = np.linalg.norm(v1) * np.linalg.norm(v2)
#     return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

# def compute_pose_error(T_0to1, T):
#     R = T[:2,:2]
#     t = T[:2,2]
#     alpha, beta = R[0,0],R[0,1]
#     R_gt = T_0to1[:2, :2]
#     t_gt = T_0to1[:2, 2]
#     alpha_gt, beta_gt = R_gt[0,0],R_gt[0,1]
#     #compute s, r, t
#     s=alpha/np.cos(np.arctan2(beta, alpha))
#     s_gt=alpha_gt/np.cos(np.arctan2(beta_gt, alpha_gt))
#     error_t = angle_error_vec(t, t_gt)
#     error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
#     error_R = angle_error_mat(R/s, R_gt/s_gt)
#     error_sacle = abs(s-s_gt)/s_gt
#     # print(R/s, R_gt/s_gt)
#     return error_t, error_R , error_sacle

def angle_error_mat(R1, R2):    
    error_matrix = np.dot(R1.T, R2)-np.eye(2)
    return np.linalg.norm(error_matrix)

def angle_error_vec(v1, v2):
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0)))

def compute_scale(M):
    a, b, c, d = M[0,0], M[0, 1], M[1,0], M[1, 1]
    s = np.sqrt((a * a) + (c * c))
    return s
    

def compute_pose_error(T_0to1, T):
    R = T[:2,:2]
    t = T[:2,2]
    # print(T)
    # print(T_0to1)
    alpha, beta = R[0,0],R[0,1]
    R_gt = T_0to1[:2, :2]
    t_gt = T_0to1[:2, 2]
    alpha_gt, beta_gt = R_gt[0,0],R_gt[0,1]
    #compute s, r, t
    s=alpha/np.cos(np.arctan2(beta, alpha))
    s_gt=alpha_gt/np.cos(np.arctan2(beta_gt, alpha_gt))
    # print('s1= ', s, s_gt)
    s = compute_scale(R)
    s_gt = compute_scale(R_gt)
    R_homo = np.vstack((R/s,np.array([0,0])))
    R_homo = np.hstack((R_homo,np.array([[0],[0],[1]])))
    R_gt_homo = np.vstack((R_gt/s_gt,np.array([0,0])))
    R_gt_homo = np.hstack((R_gt_homo,np.array([[0],[0],[1]])))
    error_t = angle_error_vec(t, t_gt)
    error_t = np.minimum(error_t, 180 - error_t)  # ambiguity of E estimation
    error_R = angle_error_mat(R/s, R_gt/s_gt)
    error_sacle = abs(s-s_gt)/s_gt
    error_R_deg = deg_error_mat(R_homo,R_gt_homo)
    # print(R/s, R_gt/s_gt)
    return error_t, error_R , error_sacle, error_R_deg

def pose_auc(errors, thresholds):
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0., errors]
    recall = np.r_[0., recall]
    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t)
        r = np.r_[recall[:last_index], recall[last_index-1]]
        e = np.r_[errors[:last_index], t]
        aucs.append(np.trapz(r, x=e)/t)
    return aucs


# --- VISUALIZATION ---


def plot_image_pair(imgs, dpi=200, size=6, pad=.5):
    n = len(imgs)
    assert n == 2, 'number of images must be two'
    figsize = (size*n, size*3/4) if size is not None else None #ori 3/4
    # _, ax = plt.subplots(1, n, figsize=figsize, dpi=dpi,)
    _, ax = plt.subplots(1, n, dpi=dpi,)
    
    for i in range(n):
        ax[i].imshow(imgs[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=255)
        # ax[i].get_yaxis().set_ticks([])
        # ax[i].get_xaxis().set_ticks([])
        ax[i].axis('off')
        for spine in ax[i].spines.values():  # remove frame
            spine.set_visible(False)
    plt.subplots_adjust(wspace=0,hspace=0)
    # plt.tight_layout(pad=pad)
    # plt.tight_layout()


def plot_keypoints(kpts0, kpts1, color='w', ps=2):
    ax = plt.gcf().axes
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)


def plot_matches(kpts0, kpts1, color, lw=1.5, ps=4):
    fig = plt.gcf()
    ax = fig.axes
    plt.axis('off')
    fig.canvas.draw()

    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(ax[0].transData.transform(kpts0))
    fkpts1 = transFigure.transform(ax[1].transData.transform(kpts1))

    fig.lines = [matplotlib.lines.Line2D(
        (fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]), zorder=1,
        transform=fig.transFigure, c=color[i], linewidth=lw)
                 for i in range(len(kpts0))]
    ax[0].scatter(kpts0[:, 0], kpts0[:, 1], c=color, s=ps)
    ax[1].scatter(kpts1[:, 0], kpts1[:, 1], c=color, s=ps)

def plot_result_images(image, T, path):
    fig = plt.gcf()
    ax = fig.axes
    # file_name = os.path.splitext(str(path))[0] + "_comp.png"
    # image_t = cv2.warpAffine(image,T,(1024,1024))
    image_t = cv2.warpAffine(image,T,image.shape[:2],borderValue=255) ##new
    # image_t = cv2.warpAffine(image,T,image.shape[:2],borderValue=0)
    
    # ax[1].imshow(image_t,cmap=plt.get_cmap('PuBuGn'),alpha=0.3)
    ax[1].imshow(image_t,cmap=plt.get_cmap('gray'),alpha=0.85)
    # cv2.imwrite(file_name,image_t)

def make_matching_plot(image0, image1, kpts0, kpts1, mkpts0, mkpts1, H,
                       color, text, path, name0, name1, show_keypoints=False,
                       fast_viz=False, opencv_display=False, opencv_title='matches'):

    if fast_viz:
        make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0, mkpts1,
                                color, text, path, show_keypoints, 10,
                                opencv_display, opencv_title)
        return

    plot_image_pair([image0, image1])
    if show_keypoints:
        plot_keypoints(kpts0, kpts1, color='k', ps=4)
        plot_keypoints(kpts0, kpts1, color='w', ps=2)
    plot_matches(mkpts0, mkpts1, color)
    if H.shape[0]!=1:
        plot_result_images(image0,H,path)

    fig = plt.gcf()
    txt_color = 'k' if image0[:100, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='bottom', ha='left', color=txt_color)

    txt_color = 'k' if image0[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, name0, transform=fig.axes[0].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    txt_color = 'k' if image1[-100:, :150].mean() > 200 else 'w'
    fig.text(
        0.01, 0.01, name1, transform=fig.axes[1].transAxes,
        fontsize=5, va='bottom', ha='left', color=txt_color)

    plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
    plt.close()


def make_matching_plot_fast(image0, image1, kpts0, kpts1, mkpts0,
                            mkpts1, color, text, path=None,
                            show_keypoints=False, margin=10,
                            opencv_display=False, opencv_title=''):
    H0, W0 = image0.shape
    H1, W1 = image1.shape
    H, W = max(H0, H1), W0 + W1 + margin

    out = 255*np.ones((H, W), np.uint8)
    out[:H0, :W0] = image0
    out[:H1, W0+margin:] = image1
    out = np.stack([out]*3, -1)

    if show_keypoints:
        kpts0, kpts1 = np.round(kpts0).astype(int), np.round(kpts1).astype(int)
        white = (255, 255, 255)
        black = (0, 0, 0)
        for x, y in kpts0:
            cv2.circle(out, (x, y), 2, black, -1, lineType=cv2.LINE_AA)
            cv2.circle(out, (x, y), 1, white, -1, lineType=cv2.LINE_AA)
        for x, y in kpts1:
            cv2.circle(out, (x + margin + W0, y), 2, black, -1,
                       lineType=cv2.LINE_AA)
            cv2.circle(out, (x + margin + W0, y), 1, white, -1,
                       lineType=cv2.LINE_AA)

    mkpts0, mkpts1 = np.round(mkpts0).astype(int), np.round(mkpts1).astype(int)
    color = (np.array(color[:, :3])*255).astype(int)[:, ::-1]
    for (x0, y0), (x1, y1), c in zip(mkpts0, mkpts1, color):
        c = c.tolist()
        cv2.line(out, (x0, y0), (x1 + margin + W0, y1),
                 color=c, thickness=1, lineType=cv2.LINE_AA)
        # display line end-points as circles
        cv2.circle(out, (x0, y0), 2, c, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, (x1 + margin + W0, y1), 2, c, -1,
                   lineType=cv2.LINE_AA)

    Ht = int(H * 30 / 480)  # text height
    txt_color_fg = (255, 255, 255)
    txt_color_bg = (0, 0, 0)
    for i, t in enumerate(text):
        cv2.putText(out, t, (10, Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    H*1.0/480, txt_color_bg, 2, cv2.LINE_AA)
        cv2.putText(out, t, (10, Ht*(i+1)), cv2.FONT_HERSHEY_DUPLEX,
                    H*1.0/480, txt_color_fg, 1, cv2.LINE_AA)

    if path is not None:
        cv2.imwrite(str(path), out)

    if opencv_display:
        cv2.imshow(opencv_title, out)
        cv2.waitKey(1)

    return out


def error_colormap(x):
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)], -1), 0, 1)

def estimate_homo(kpts0, kpts1):
    # if kpts1.shape[0]>3:
    #     H, mask=cv2.findHomography(kpts0,kpts1,cv2.RANSAC,5.0)
    # else:
    #     H = cv2.getAffineTransform(kpts0,kpts1)
    H = cv2.estimateAffinePartial2D(kpts0, kpts1, method=cv2.RANSAC, ransacReprojThreshold=8.0,confidence=0.99)
        
        
    return H

def compute_geometry_match_score(kpts0, kpts1, T):
    kpts0 = to_homogeneous(kpts0)
    kpts1 = to_homogeneous(kpts1)
    # print("homo kpts:", kpts0.T.shape)
    kpts_t = np.dot(T,kpts0.T)
    # print("kpts:", kpts_t.shape)
    diff= kpts_t.T-kpts1
    dist = np.linalg.norm(diff[:,:2], axis=1)
    # print("error:",dist)
    correct = dist < 10
    num_correct = np.sum(correct)
    precision = np.mean(correct) if len(correct) > 0 else 0
    matching_score = num_correct / len(kpts0) if len(kpts0) > 0 else 0
    # print("precision:",precision,"geometry matching_score", matching_score)
    return precision, matching_score

def compute_match_score(matches,matches_gt):
    # print("supergraph matches:",matches)
    # print("supergraph matches ground truth:",matches_gt)
    num_correct=0
    num_error=0
    sum=matches.shape[0]
    sum=sum-np.sum(matches<0)
    for i in range(matches.shape[0]):
        idx = np.argwhere(matches_gt[1,:]==i)
        if matches[i]==matches_gt[0,idx]:
            num_correct +=1
        else:
            num_error +=1
    if sum!=0:
        match_score = num_correct/sum
    else:
        match_score = 0
    # print("match_scores:",match_score)
    return match_score

def EuclideanDistance(t1,t2):
    B,N,_=t1.size()
    _,M,_=t2.size()
    dist = -2 * torch.matmul(t1, t2.permute(0, 2, 1))
    dist += torch.sum(t1 ** 2, -1).view(B, N, 1)
    dist += torch.sum(t2 ** 2, -1).view(B, 1, M)
    dist[dist<0]=1
    dist=torch.sqrt(dist)
    return dist

def nearest_ransac(kpts0,kpts1,desc0,desc1,ratio=1):
    B = np.hstack((kpts0, desc0.T))
    A = np.hstack((kpts1, desc1.T))
    knn = NearestNeighbors(n_neighbors=1).fit(B)
    distances, indices = knn.kneighbors(A)
    # print("distance.shape",distances.shape)
    # print("indice.shape",indices)

    matched = []
    pairs = []
    # for indexA, candidatesB in enumerate(indices):
    #     for indexB in candidatesB:
    #         if indexB not in matched:
    #             matched.append(indexB)
    #             pairs.append([indexA, indexB])
    #             break
    for indexA, candidatesB in enumerate(indices):
        for indexB in candidatesB:
            if indexB not in matched:
                matched.append(indexB)
                pairs.append([indexA, indexB])
            else:
                temp = np.argwhere(np.array(pairs)[:,1]==indexB)[0][0]
                index_past = pairs[temp][0]
                if distances[indexA,:] <= distances[index_past,:]:
                    pairs[temp][0] = indexA
            break
    pairs = np.array(pairs)
    pts0 = kpts0[pairs[:,1]]
    pts1 = kpts1[pairs[:,0]]
    conf = distances[pairs[:,0]].reshape(-1)  
    # print(conf.shape) ##(m,1)
    return pairs, pts0, pts1, conf
# matches = pd.DataFrame(pairs, columns=['SetA', 'SetB'])


def record_error(mkpts0,mkpts1,kpts1,matches1,matches_gt,T_gt,success_match,text):
            ##comput error
    if mkpts1.shape[0]<=1:
        H = np.array([0])
    else:
        H, _ = estimate_homo(mkpts0, mkpts1)
    # print("transfomation from match:",H)
    precision, matching_score = compute_geometry_match_score(mkpts0,mkpts1,T_gt)
    matching = compute_match_score(matches1,matches_gt)
    if H is None or H.shape[0]==1:
        err_t, err_R, err_s, err_R_deg = np.inf, np.inf, np.inf, np.inf
    else:
        err_t, err_R, err_s, err_R_deg = compute_pose_error(T_gt, H)
    text += ['MS: {}'.format(matching)]
    if abs(err_t) < 20 and err_R < 0.08 and abs(err_s)<0.07:
        # print("This graph matches sucessusfully")
        text += ['Success']
        success_match.append(1)
    elif(kpts1.shape[0]==1):
        if(matching==1):
            text += ['Success']
            success_match.append(1)
        else:
            text += ['False']
            success_match.append(0)
    else: 
        text += ['False']
        success_match.append(0)
        
    print("err_t:",err_t,"err_R:",err_R, "err_s:", err_s, "err_R_auc", err_R_deg)
    return H, precision, matching, success_match, text, err_R, err_t, err_R_deg

def record_nn_error(mkpts0,mkpts1,kpts1,T_gt,success_match,text):
    ##comput error
    if mkpts1.shape[0]<=1:
        H = np.array([0])
    else:
        H, mask = estimate_homo(mkpts0, mkpts1)
    # print("transfomation from match:",H)
    precision, matching_score = compute_geometry_match_score(mkpts0,mkpts1,T_gt)
    if H is None or H.shape[0]==1:
        err_t, err_R, err_s, err_R_deg = np.inf, np.inf, np.inf, np.inf
    else:
        err_t, err_R, err_s, err_R_deg = compute_pose_error(T_gt, H)
    text += ['MS: {}'.format(precision)]
    if abs(err_t) < 20 and err_R < 0.08 and abs(err_s)<0.07:
        # print("This graph matches sucessusfully")
        text += ['Success']
        success_match.append(1)
    elif(kpts1.shape[0]==1):
        if(matching_score==1):
            text += ['Success']
            success_match.append(1)
        else:
            text += ['False']
            success_match.append(0)
    else: 
        text += ['False']
        success_match.append(0)
        
    print("err_t:",err_t,"err_R:",err_R, "err_s:", err_s, "err_R_auc", err_R_deg)
    return H, precision, matching_score, success_match, text, err_R, err_t, err_R_deg


def estimation_one_area(kpts0, kpts1, desc0, desc1, matches):
    """
    Predict result: which room the robot is in now when we just have one node in graph
    Then we compute the transformation from area descriptors
    """
    return 0