import numpy as np
from models.utils import pose_auc
errt = []
errR = []
pose_errors = []

with open("t_err.txt","r") as f:
    errt_file = f.readline()
    sample_ = errt_file.split(', [')
    pose_errt_ = sample_[1].split(', ')
    for pose in pose_errt_:
        pose=float(pose)
        if(pose== -1):
            pose=np.inf
        errt.append(pose)
with open("R_err.txt","r") as f:
    errt_file = f.readline()
    sample_ = errt_file.split(', [')
    pose_errt_ = sample_[1].split(', ')
    i=0
    for pose in pose_errt_:
        pose=float(pose)
        if(pose== -1):
            pose=np.inf
        pose = np.rad2deg(pose)
        errR.append(pose)
        i=i+1
print(i)

assert len(errt)==len(errR)
for i in range(len(errt)):
    pose_error = np.maximum(errR[i], errt[i])
    pose_errors.append(pose_error)
for j in range(147):{
    pose_errors.append(np.inf)
}
thresholds = [1, 3, 5, 10, 20]
aucs = pose_auc(pose_errors, thresholds)
print("auc of ours:", aucs)