import os
import sys
import cv2
import random
import colorsys
from io import BytesIO
import argparse


import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image

from utils.util import setup_seed, load_checkpoint
import utils.visualize_utils
from options import Option
from model.model import Model
from data_utils.utils import preprocess



# def sa_sk_feature(args, path):

#     # prepare model
#     model = Model(args)
#     model = model.half()

#     if args.load is not None:
#         checkpoint = load_checkpoint(args.load)

#     cur = model.state_dict()
#     new = {k: v for k, v in checkpoint['model'].items() if k in cur.keys()}
#     cur.update(new)
#     model.load_state_dict(cur)

#     if len(args.choose_cuda) > 1:
#         model = torch.nn.parallel.DataParallel(model.to('cuda'))
#     model = model.cuda()
#     model.eval()
#     torch.set_grad_enabled(False)


#     #extract sketch features
#     sketch = preprocess(path, 'sk').half()

#     sketch = sketch.unsqueeze(0).cuda()

#     sk, sk_idx = model(sketch, None, 'test', only_sa = True)

#     return sk

def plot_result(query, top, path): 
    top = top[:5]
    fig = plt.figure(figsize=(10, 7)) 
    rows = 1
    columns = 6
    query = cv2.imread(query)
    img_list = [cv2.imread(img) for img in top]

    fig.add_subplot(rows, columns, 1) 
    plt.imshow(query) 
    plt.axis('off') 
    plt.title("Query Sketch")
    
    for i in range(2, 7):
        fig.add_subplot(rows, columns, i)
        plt.imshow(img_list[i - 2]) 
        plt.axis('off') 
        plt.title("Top " + str(i - 1))
  
    fig.savefig(path)

def retrieve(device, query=None, gallery='Sketchy', model=None, lite=True, args=None, path=None, k=16):

    if query is None:
        raise ValueError("Please provide query image (path)!")
    
    query_img = preprocess(query, "sk").unsqueeze(0).to(device)

    sk, _ = model(query_img, None, 'test', only_sa = True)

    scores = np.array([])
    path_to_img = []
    gallery_path = None
    category_list = None

    if gallery == 'TUBerlin':
        gallery_path = os.path.join(args.data_path, "TUBerlin", "ImageResized_ready")
        feature_path = os.path.join(args.data_path, "TUBerlin", "256x256", "sa_features")
    elif gallery == 'Sketchy':
        gallery_path = os.path.join(args.data_path, "Sketchy", "EXTEND_image_sketchy_ready")
        feature_path = os.path.join(args.data_path, "Sketchy", "256x256", "sa_features")
    else:
        raise ValueError("Not Implement yet!")

    

    category_list = os.listdir(feature_path)
    for folder in category_list:
        category_path = os.path.join(feature_path, folder)
        feature_list = os.listdir(category_path)
        
        if lite:
            len_retrieval = 5
        else:
            len_retrieval = len(feature_list)
        
        category_feature = []

        for feature_file in feature_list[:len_retrieval]:
            img_feature = torch.from_numpy(np.load(os.path.join(category_path, feature_file)))

            category_feature.append(img_feature)
            path_to_img.append(os.path.join(gallery_path, folder, feature_file[:-4]))

        category_feature = torch.stack(category_feature, dim=0)
        _ , score = model(sk.repeat(len(category_feature), 1, 1), category_feature.to(device), 'test')
        scores = np.append(scores, score.cpu().numpy())
    
    # scores = np.array(scores)
    path_to_img = np.array(path_to_img)

    top_k = path_to_img[scores.argsort()[-1:-k-1:-1]]

    plot_result(query, top_k, path)
    return top_k
  


if __name__ == '__main__':
    args = Option().parse()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model(args)
    
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    model.to(device)
    if os.path.isfile(args.load):
        checkpoint = load_checkpoint(args.load)
        cur = model.state_dict()
        new = {k: v for k, v in checkpoint['model'].items() if k in cur.keys()}
        cur.update(new)
        model.load_state_dict(cur)  
    else:
        raise ImportError("Pre-trained weigths not found!")

    retrieve(device=device, query=args.sketch_path, gallery="Sketchy", model=model, args=args)


