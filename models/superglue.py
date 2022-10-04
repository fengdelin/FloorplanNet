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
#
# %AUTHORS_END%
# --------------------------------------------------------------------*/
# %BANNER_END%

from copy import deepcopy
from pathlib import Path
import torch
from torch import log_softmax, nn, unsqueeze
import numpy as np
from .utils import EuclideanDistance
import torch.nn.functional as F


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                # layers.append(nn.BatchNorm1d(channels[i]))
                layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def normalize_keypoints(kpts, image_shape):
    """ Normalize keypoints locations based on image image_shape"""
    # print(image_shape)
    _, _, height, width = image_shape
    one = kpts.new_tensor(1)
    size = torch.stack([one*width, one*height])[None]
    center = size / 2
    scaling = size.max(1, keepdim=True).values * 0.7
    return (kpts - center[:, None, :]) / scaling[:, None, :]


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, feature_dim, layers):
        super().__init__()
        self.encoder = MLP([3] + layers + [feature_dim])###original 3
        self.encoder1 = MLP([3] + layers + [feature_dim],False)###original 3
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        # print(torch.cat(inputs, dim=1).shape)
        # inputs = [kpts.transpose(1, 2)]
        if inputs[0].shape[2]==1:
            return self.encoder1(torch.cat(inputs, dim=1))
        else:
            return self.encoder(torch.cat(inputs, dim=1))


def attention(query, key, value, edge):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5
    prob_no = torch.nn.functional.softmax(scores, dim=-1)
    # print("prob_none:",prob_no.shape)
    prob = torch.zeros_like(prob_no)
    # edge=None;
    ##add edge propagation
    # print("alpha weight:", prob.shape)
    if(edge!= None):
        # print("edge:",edge.shape)
        for i in range(prob.size(0)):
            for j in range(prob.size(1)):
                # print("orign:",prob[i,j,:,:].detach().cpu().numpy())
                prob[i,j,:,:] = torch.mul(prob_no[i,j,:,:],edge[i,:,:])
    else:
        prob = prob_no
                
        
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value, edge=None):
        batch_dim = query.size(0)
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        # print("query.shape:", query.shape)
        x, prob = attention(query, key, value, edge)
        # print("message:",x.shape,prob.shape)
        # print("x:", torch.any(torch.isnan(x)))
        # print("prob:", torch.any(torch.isnan(prob)))
        self.prob.append(prob)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        self.mlp1 = MLP([feature_dim*2, feature_dim*2, feature_dim],False)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source, edge = None):
        
        message = self.attn(x, source, source, edge)
        # print("message:", torch.any(torch.isnan(message)))
        # print("message:",x.shape,message.shape)
        if(message.shape[-1]==1):
            return self.mlp1(torch.cat([x, message], dim=1))
        else:
            return self.mlp1(torch.cat([x, message], dim=1))
            


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionalPropagation(feature_dim, 2)##head
            for _ in range(len(layer_names))])
        self.names = layer_names

    def forward(self, desc0, desc1, edge0, edge1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'cross':
                src0, src1 = desc1, desc0
                delta0, delta1 = layer(desc0, src0, None), layer(desc1, src1, None)                
            else:  # if name == 'self':
                src0, src1 = desc0, desc1
                delta0, delta1 = layer(desc0, src0, edge0), layer(desc1, src1, edge1)
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.

    Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, and Andrew
    Rabinovich. SuperGlue: Learning Feature Matching with Graph Neural
    Networks. In CVPR, 2020. https://arxiv.org/abs/1911.11763

    """
    default_config = {
        'descriptor_dim': 26,
        'weights': 'indoor',
        'keypoint_encoder': [6, 13, 26],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 100,
        'match_threshold': 0.3,
        'mode': 'train',
    }

    def __init__(self, config):
        super().__init__()
        self.config = {**self.default_config, **config}

        self.kenc = KeypointEncoder(
            self.config['descriptor_dim'], self.config['keypoint_encoder'])

        self.gnn = AttentionalGNN(
            self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(
            self.config['descriptor_dim'], self.config['descriptor_dim'],
            kernel_size=1, bias=True)

        bin_score = torch.nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)
        
        if self.config['mode']=='test' or self.config['premodel']!='N':
            path = Path(__file__).parent
            path = path / 'weights/best_{}.pth'.format(self.config['weights'])
            print("checkpoint path:", path)
            self.load_state_dict(torch.load(path).state_dict())
            print('Loaded SuperGlue model (\"{}\" weights)'.format(
                self.config['weights']))

    def forward(self, data):
        """Run SuperGlue on a pair of keypoints and descriptors"""
        desc0, desc1 = data['descriptors0'].double(), data['descriptors1'].double()
        kpts0, kpts1 = data['keypoints0'].double(), data['keypoints1'].double()

        desc0 = desc0.transpose(0,1)
        desc1 = desc1.transpose(0,1)
        kpts0 = torch.reshape(kpts0, (1, -1, 2))
        kpts1 = torch.reshape(kpts1, (1, -1, 2))

    
        if kpts0.shape[1] == 0 or kpts1.shape[1] == 0:  # no keypoints
            shape0, shape1 = kpts0.shape[:-1], kpts1.shape[:-1]
            return {
                'matches0': kpts0.new_full(shape0, -1, dtype=torch.int)[0],
                'matches1': kpts1.new_full(shape1, -1, dtype=torch.int)[0],
                'matching_scores0': kpts0.new_zeros(shape0)[0],
                'matching_scores1': kpts1.new_zeros(shape1)[0],
                'skip_train': True
            }

        file_name = data['file_name']
        
        # Keypoint normalization.
        kpts0 = normalize_keypoints(kpts0, data['image0'].shape)
        kpts1 = normalize_keypoints(kpts1, data['image1'].shape)
        # print("des:",desc1.shape,"kpts0:",kpts1.shape,"score0:",data['scores1'].shape)

        # Keypoint MLP encoder.
        # print("des:",desc1.shape,"kpts0:",kpts1.shape,"score0:",data['scores1'].shape)
        desc0 = desc0 + self.kenc(kpts0, torch.transpose(data['scores0'], 0, 1))
        desc1 = desc1 + self.kenc(kpts1, torch.transpose(data['scores1'], 0, 1))
        # print("des:",desc1.shape)
        # print("desc0", torch.any(torch.isnan(desc1)))
        # print("desc1", torch.any(torch.isnan(desc0)))

        #get edge attention
        edge1 = data['edge_prob1'].double().transpose(0,1)
        edge2 = data['edge_prob2'].double().transpose(0,1)
        # print("edge size:",edge1.shape,edge2)
        
        if kpts1.shape[1]==1:
            edge2 = torch.unsqueeze(edge2,0)
        adjmatrix1 = edge1.clone()
        adjmatrix2 = edge2.clone() 
        edge1[edge1==1]=0.7
        edge2[edge2==1]=0.7
        # print("edge size:",edge1.shape,edge2.shape)
        
        #get spatial attention bias
        dist0 = EuclideanDistance(kpts0,kpts0) #(1,n,n)
        dist1 = EuclideanDistance(kpts1,kpts1) 
        dist0 = nn.functional.normalize(dist0)
        dist1 = nn.functional.normalize(dist1)
        # print("dist0:",dist0)
        # print("dist1:",dist1)
        spatial_pos0 = torch.mul(dist0,adjmatrix1)
        spatial_pos1 = torch.mul(dist1,adjmatrix2)
        # print("p0:",torch.any(torch.isnan(spatial_pos0)))
        spatial_pos0[spatial_pos0==0] = -1
        spatial_pos1[spatial_pos1==0] = -1
        diff0 = torch.eye(dist0.shape[1]).cuda()
        diff1 = torch.eye(dist1.shape[1]).cuda()
        spatial_pos0 += diff0.unsqueeze(0)
        spatial_pos1 += diff1.unsqueeze(0)
        spatial_pos0[spatial_pos0==0] = 5e-2
        spatial_pos1[spatial_pos1==0] = 5e-2
        ## print("sptial_pos0:",spatial_pos0)
        spatial_pos0 = 1./ spatial_pos0
        spatial_pos1 = 1./ spatial_pos1
        
        ### ie
        # spatial_pos0[spatial_pos0==0] = 1
        # spatial_pos1[spatial_pos1==0] = 1
        # spatial_pos0 =  spatial_pos0
        # spatial_pos1 =  spatial_pos1
        spatial_pos0[spatial_pos0<0] = 0.5
        spatial_pos1[spatial_pos1<0] = 0.5
        # print("sptial_pos1:",spatial_pos1)
        spatial_pos0 = nn.functional.normalize(spatial_pos0)
        spatial_pos1 = nn.functional.normalize(spatial_pos1)
        # print("sptial_pos1:",spatial_pos1)

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1, spatial_pos0, spatial_pos1)
        # desc0, desc1 = self.gnn(desc0, desc1, edge1, edge2)

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        # print("mdes0:",mdesc0[0])
        # print("mdes1:",mdesc1[0])

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5
        
        # Run the optimal transport.
        scores = log_optimal_transport(
            scores, self.bin_score,
            iters=self.config['sinkhorn_iterations'])
    

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        # print(mscores0.shape) #(1,n)
        if mscores0.shape[1]<=3:
            threshold = self.config['match_threshold']/2
        else:
            threshold = self.config['match_threshold']

        # threshold = self.config['match_threshold']
        valid0 = mutual0 & (mscores0 > threshold)
        # valid0 = valid0 & (mscores0 < 0.5)
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))
        # print("indices0:",indices0) #floor plan 所对应的map indices
        # print("indices1:",indices1)
        
        loss_graph = []
        
        # check if indexed correctly
        if self.config['mode']=='train':
        # if self.config['mode']=='train':
            all_matches = data['all_matches'].permute(1,2,0) # shape=torch.Size([1, 87, 2])
            # print("matches:", all_matches)
            for i in range(len(all_matches[0])):
                x = all_matches[0][i][0]
                y = all_matches[0][i][1]
                # print("each matches loss:",scores[0][x][y].exp().detach().cpu())
                # loss_graph = -torch.log( scores[0][x][y].exp()+ 1e-8 )
                loss_graph.append(-torch.log( scores[0][x][y].exp()+ 1e-8 )) # check batch size == 1 ?
                # for p0 in unmatched0:
                #     loss += -torch.log(scores[0][p0][-1])
                # for p1 in unmatched1:
                #     loss += -torch.log(scores[0][-1][p1])
            loss_graph = torch.mean(torch.stack(loss_graph))
            loss_graph= torch.reshape(loss_graph, (1, -1))

            dist = 0
            condition = indices1[0]>-1
            # # print("condition:",condition)
            num_valid = torch.sum(condition)
            
            
            # #### for new data
            # x_= kpts0.shape[1]
            # y_= kpts1.shape[1]
            # geo_score = torch.ones((x_+1,y_+1)) * 1.5
            # for i in range(len(all_matches[0])):
            #     x = all_matches[0][i][0]
            #     y = all_matches[0][i][1]
            #     if y!=y_ :
            #         if x!=x_:
            #             for j in range(x_):
            #                 geo_score[j][y]= torch.norm(kpts0[0][j]-kpts0[0][x])
            #             geo_score[-1][y]= torch.norm(-kpts0[0][x])
            #         else:
            #             for j in range(x_):
            #                 geo_score[j][y]= torch.norm(kpts0[0][j])
            #             geo_score[-1][y]= torch.tensor(0)
            #     else:
            #         geo_score[x][y]= torch.tensor(0)
                    
            # #### print(geo_score)
            
            
            # if num_valid > 0:
            #     for i in range(indices0.shape[1]):
            #         j = indices0[0][i]
            #         if j!= -1:
            #             dist += geo_score[i][j]
            #     loss_geo = dist/num_valid
            # else:
            #     loss_geo = torch.Tensor([1])
            # loss_geo = torch.reshape(loss_geo, (1, -1)).cuda().requires_grad_(True) 
            # # print("loss_geo:",loss_geo) 
            
            
            # #########
            # ####for kld loss, geometric loss
            # geo_dtb = torch.zeros((num_valid,2)) ##distribution 
            # geo_dtb_gt = torch.zeros((num_valid,2)) ##distribution 
            # if num_valid > 0:
            #     k = 0;
            #     for i in range(indices1.shape[1]):
            #         j = indices1[0][i]
            #         j_gt = all_matches[0][i][0]
            #         if j!= -1:
            #             geo_dtb[k][:]=kpts0[0][j]
            #             geo_dtb_gt[k][:] = kpts0[0][j_gt]
            #             k=k+1
            #             # print("j and j_gt:",j,j_gt)
                        
            #     loss_geo_kdl = F.kl_div(F.log_softmax(geo_dtb,dim=-1), F.softmax(geo_dtb_gt,dim=-1), reduction='sum')
            # else:
            #     loss_geo_kdl = torch.Tensor([1])
                
            # loss_geo_kdl = torch.reshape(loss_geo_kdl, (1, -1)).cuda().requires_grad_(True) 
            # # print("loss_geo_kdl:",loss_geo_kdl) 
            
                        
                        
                    
                    
                
            
            # ## x_= kpts0.shape[1]
            # # y_= kpts1.shape[1]
            # # mn1 = all_matches[0,:,0]
            # # mn2 = all_matches[0,:,1]
            # # leave2 = mn2[mn1!= x_]
            # # leave2 = leave2[leave2 != y_]
            # # print("leaving edge index:",leave2)
            # if num_valid>1:
            #     index1 = indices1[0][condition]
            #     M1 = torch.eye(num_valid).cuda()
            #     topo = adjmatrix2[0]
            #     M2 = topo[condition,:]
            #     M2 = M2[:,condition]
            #     # print("M2:",M2)
            #     for j in range(num_valid):
            #         for k in range(num_valid):
            #             m = index1[j]
            #             n = index1[k]
            #             M1[j][k]= adjmatrix1[0][m][n]
            #     # print("M1:",M1)
            #     loss_topo = 1-(torch.sum(M1==M2)/num_valid**2)
            # else:
            #     loss_topo = torch.Tensor([1])
            # # print("loss_topo:",loss_topo)
            # loss_topo = torch.reshape(loss_topo, (1, -1)).cuda().requires_grad_(True)        
                    
                    
            # ##### ####
            # # for old data(right version)
            # dist = 0
            # condition = indices1[0]>-1
            # num_valid = torch.sum(condition)
            # if num_valid>0:
            #     idx0 = indices1[0][condition]
            #     # print("idx1", idx0)
            #     # print("idx0", indices0[0][indices0[0]>-1])
            #     idx1 = condition.nonzero()
            #     # print("idx1", idx1)
            #     for i in range(num_valid):
            #         i1 = idx1[i][0]
            #         i0 = idx0[i]  ###need check!!!
            #         x0_gt = all_matches[0][i1][0]  ## because the all_matched[0][:][0] is the correct order of indices
            #         pt0_gt = kpts0[0][x0_gt]
            #         # print("i0",i0)
            #         # print("x0_gt",x0_gt)
            #         pt0 = kpts0[0][i0]
            #         # print("pt0_gt: ", pt0_gt, "pt0: ", pt0)
            #         dist += torch.norm(pt0_gt-pt0)
            #     loss_geo = dist/num_valid
            # else:
            #     loss_geo = torch.Tensor([1])
            
            # loss_geo = torch.reshape(loss_geo, (1, -1)).cuda().requires_grad_(True)
        
            # if num_valid>1:
            #     index1 = indices1[0][condition]
            #     M1 = torch.eye(num_valid).cuda()
            #     M2 = adjmatrix2[0]
            #     # print("M2:",M2.shape)
            #     M2 = M2[condition,:]
            #     M2 = M2[:,condition]
            #     for j in range(num_valid):
            #         for k in range(num_valid):
            #             m = index1[j]
            #             n = index1[k]
            #             M1[j][k]= adjmatrix1[0][m][n]
            #     # print("M1:",M1)
            #     loss_topo = 1-(torch.sum(M1==M2)/num_valid**2)
            # else:
            #     loss_topo = torch.Tensor([1])
            # # print("loss_topo:",loss_topo)
            # loss_topo = torch.reshape(loss_topo, (1, -1)).cuda().requires_grad_(True)
               
        
            ###compute projection error
        
        
            # dist = 0
            # condition = indices1[0]>-1
            # num_valid = torch.sum(condition)
            # if num_valid>0:
            #     idx0 = indices1[0][condition]
            #     print("idx0", idx0)
            #     idx1 = condition.nonzero()
            #     for i in range(num_valid):
            #         i1 = idx1[i][0]
            #         i0 = idx0[i]  ###need check!!!
            #         x0_gt = all_matches[0][i1][0]  ## because the all_matched[0][:][0] is the correct order of indices
            #         pt0_gt = kpts0[0][x0_gt]
            #         print("i0",i0)
            #         print("x0_gt",x0_gt)
            #         pt0 = kpts0[0][i0]
            #         print("pt0_gt: ", pt0_gt, "pt0: ", pt0)
            #         dist += torch.norm(pt0_gt-pt0)
            #     loss_geo = dist/num_valid
            # else:
            #     loss_geo = loss_topo = torch.Tensor([1])
            
            # loss_geo = torch.reshape(loss_geo, (1, -1)).cuda().requires_grad_(True)
        
                
            # if num_valid>1:
            #     index1 = indices1[0][condition]
            #     M1 = torch.eye(num_valid).cuda()
            #     M2 = adjmatrix2[0]
            #     M2 = M2[condition,condition]
            #     # print("M2:",M2)
            #     for j in range(num_valid):
            #         for k in range(num_valid):
            #             m = index1[j]
            #             n = index1[k]
            #             M1[j][k]= adjmatrix1[0][m][n]
            #     # print("M1:",M1)
            #     loss_topo = 1-(torch.sum(M1==M2)/num_valid**2)
            # else:
            #     loss_topo = torch.Tensor([1])
            # # print("loss_topo:",loss_topo)
            # loss_topo = torch.reshape(loss_topo, (1, -1)).cuda().requires_grad_(True)
        
            # ###compute projection error
            
            
            # loss_mean = 1 * loss_graph + 0.25 * loss_topo + 1.5 * loss_geo_kdl
            # loss_mean = 1 * loss_graph + 0.25 * loss_topo  + 0.5 * loss_geo
            # print("loss:",loss_graph[0].item(), "loss_topo:",loss_topo[0].item(),"loss_geo:",loss_geo[0].item())
            loss_mean = loss_graph
            # print("loss:",loss_graph[0].item(), "loss_topo:",loss_topo[0].item())
            # loss_mean = 1 * loss_graph + 1.5 * loss_geo_kdl
            # print("loss:",loss_graph[0].item(),"loss_geo:",loss_geo[0].item())
            # loss_mean = 1 * loss_graph + 0.5 * loss_topo
        else:
            loss_mean = torch.Tensor([-1])
            
        
        
        return {
            'matches0': indices0[0], # use -1 for invalid match
            'matches1': indices1[0], # use -1 for invalid match
            'matching_scores0': mscores0[0],
            'matching_scores1': mscores1[0],
            'loss': loss_mean[0],
            'skip_train': False
        }

        # scores big value or small value means confidence? log can't take neg value
