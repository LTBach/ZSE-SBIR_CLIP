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
import pandas as pd
from PIL import Image

from utils.util import setup_seed, load_checkpoint
import utils.visualize_utils
from options import Option
from model.model import Model
from data_utils.utils import preprocess

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

def retrieve(device, embeded, df, centroid_feats, query=None, gallery='Sketchy', model=None, lite=False, args=None, path=None, k=16, t = 1):

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
    elif gallery == 'Sketchy':
        gallery_path = os.path.join(args.data_path, "Sketchy", "256x256", "photo", "tx_000000000000_ready")
    else:
        raise ValueError("Not Implement yet!")
    
    # calculate relevance between sketch and clusters 
    _ , centroid_scores = model(sk.repeat(len(centroid_feats), 1, 1), centroid_feats, 'test')
    centroid_scores = np.squeeze(centroid_scores.cpu().numpy())
    
    cluster_idx = np.argsort(centroid_scores)
    
    pos = -1
    
    index = df.index[df['cluster_idx']==cluster_idx[pos]]
    
    while len(index) < k:
        pos -= 1
        index = index.union(df.index[df['cluster_idx'] == cluster_idx[pos]])
        
    img_name = df.iloc[index]['image_name'].to_numpy()
    category_name = df.iloc[index]['class'].to_numpy()
    
    img_feats = torch.from_numpy(np.array([embeded[name] for name in img_name]))

    _ , score = model(sk.repeat(len(img_feats), 1, 1), img_feats.to(device), 'test')

    score = np.squeeze(score.cpu().numpy())

    top_k_index = score.argsort()[::-1][:k]
    top_k_name = img_name[top_k_index]
    top_k_category = category_name[top_k_index]
    
    #plot_result(query, top_k, path)
    res = [os.path.join(gallery_path, top_k_category[idx], top_k_name[idx][:-4]) for idx in range(k)]
    print('res:',res)
    return res
  


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
