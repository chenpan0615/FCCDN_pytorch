"""A simple inference code"""
import os
import torch
import torch.nn as nn
from torchvision import transforms as T
from tqdm import tqdm
import cv2
import numpy as np

from networks.FCCDN import FCCDN
print("**********warning**********")
print("We have updated the model by replacing the upsample mode from \'bilinear\' to \'nearest\'. ")
print("Please update to the latest code.")

test_in_path = "/nfs/project/netdisk/192.168.10.227/d/cp/LEVIR_CD/raw/test/"
test_out_path = "/nfs/project/netdisk/192.168.10.227/d/cp/LEVIR_CD/raw/out/FCCDN/"
pretrained_weights = "./pretrained/FCCDN_test_LEVIR_CD.pth"
mean_value = [0.37772245912313807, 0.4425350597897193, 0.4464795300397427]
std_value = [0.1762166286060892, 0.1917139949806914, 0.20443966020731438]

"make paths"
if not os.path.exists(test_out_path):
    os.makedirs(test_out_path)
if not os.path.exists(os.path.join(test_out_path, "change")):
    os.makedirs(os.path.join(test_out_path, "change"))
if not os.path.exists(os.path.join(test_out_path, "seg1")):
    os.makedirs(os.path.join(test_out_path, "seg1"))
if not os.path.exists(os.path.join(test_out_path, "seg2")):
    os.makedirs(os.path.join(test_out_path, "seg2"))

basename_list = []
files = os.listdir(os.path.join(test_in_path, "A"))
for file in files:
    if file[-3:] == "png":
        basename_list.append(file)

model = FCCDN(num_band=3, use_se=True)
pretrained_dict = torch.load(pretrained_weights, map_location="cpu")
module_model_state_dict = {}
for item, value in pretrained_dict['model_state_dict'].items():
    if item[0:7] == 'module.':
        item = item[7:]
    module_model_state_dict[item] = value
model.load_state_dict(module_model_state_dict, strict=True)
model.cuda()
model.eval()
normalize = T.Normalize(mean=mean_value, std=std_value)

"""This is a simple inference code. Users can speed up the inference with torch.utils.data.DataLoader"""
with tqdm(total=len(basename_list)) as pbar:
    pbar.set_description("Test")
    with torch.no_grad():
        for basename in basename_list:
            pre = cv2.imread(os.path.join(test_in_path, "A", basename))
            post = cv2.imread(os.path.join(test_in_path, "B", basename))
            pre = normalize(torch.Tensor(pre.transpose(2,0,1)/255))[None].cuda()
            post = normalize(torch.Tensor(post.transpose(2,0,1)/255))[None].cuda()
            pred = model([pre, post])
            # change_mask = torch.sigmoid(pred[0]).cpu().numpy()[0,0]
            out = torch.round(torch.sigmoid(pred[0])).cpu().numpy()
            seg1 = torch.round(torch.sigmoid(pred[1])).cpu().numpy()
            seg2 = torch.round(torch.sigmoid(pred[2])).cpu().numpy()
            cv2.imwrite(os.path.join(test_out_path, "change", basename), out[0,0].astype(np.uint8))
            cv2.imwrite(os.path.join(test_out_path, "seg1", basename), seg1[0,0]*255)
            cv2.imwrite(os.path.join(test_out_path, "seg2", basename), seg2[0,0]*255)
            pbar.update()
