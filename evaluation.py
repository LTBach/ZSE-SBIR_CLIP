import torch
import pandas as pd
import json as js
from PIL import Image

import os
import random

from utils.util import get_output_file, load_checkpoint
from options import Option
from model.model import Model
from retrieval import retrieve

import time

PATH = os.path.join('datasets', "Sketchy", "256x256", "photo", "tx_000000000000_ready")
query_PATH = os.path.join('datasets', "Sketchy", "256x256", "sketch", "tx_000000000000_ready")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

args = Option().parse()
    
    # Setup model
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
    
def AP_prec_recall(query, retrieved, total_rel):
    total_rel_found = 0
    AP = 0
    for i, item in enumerate(retrieved):
        if item.upper() == query.upper():
            total_rel_found += 1
            AP = AP + (total_rel_found / (i + 1))
    
    if total_rel_found == 0:
        return 0, 0, 0
    
    AP = AP / total_rel_found
    prec = total_rel_found / len(retrieved)
    recall = total_rel_found / total_rel
    
    return AP, prec, recall

def main(args):
    # Specify canvas parameters in application
    num_query_per_class = 3
    num_retrieve = 100
    
    query_list = os.listdir(query_PATH)
    
    log = []
    
    mAP = 0.0
    mPrec = 0.0
    mRecall = 0.0
    mExecute_time = 0.0
    cnt = 0
    
    eval_set = pd.read_csv('eval.csv')
    
    for i, query in enumerate(query_list):
        sk_path = os.path.join(query_PATH, query)
        sk_list = eval_set[query]
        num_rel = len(os.listdir(os.path.join(PATH, query)))
        
        for j, sk in enumerate(sk_list): 
            # Perform searching
            t0 = time.time()
            res = retrieve(device, os.path.join(sk_path, sk), "Sketchy", model, False, args, None, k=num_retrieve, t = 3)
            t1 = time.time()
            
            # evaluation
            execute_time = t1-t0
            
            AP, prec, recall = AP_prec_recall(query, res, num_rel)
            
            
            print(f'query {i * num_query_per_class + j} - AP@{len(res)} = {AP} - prec@{len(res)} = {prec} - recall@{len(res)} = {recall} - search time: {execute_time}')
            
            print(f'query: {query} - top_5 retrieve result: {res[:5]}')

            log.append([i * num_query_per_class + j, AP, prec, recall, execute_time])
            
            mAP = mAP + AP
            mPrec = mPrec + prec
            mRecall = mRecall + recall
            mExecute_time = mExecute_time + execute_time
            cnt += 1
    
    mAP = mAP / cnt
    mPrec = mPrec / cnt
    mRecall = mRecall / cnt
    mExecute_time = mExecute_time / cnt
    
    evaluation_result = {'mAP@100': mAP, 
                        'mPrec@100': mPrec,
                        'mRecall@100': mRecall,
                        'average_search_time': mExecute_time
                        }
    
    with open('../working/evaluation_result.csv', 'w') as f:
        js.dump(evaluation_result, f)
    
    cols = ['query', 'AP@100', 'prec@100', 'recall@100', 'search_time']
    log = pd.DataFrame(log, columns = cols)
    
    log.to_csv('../working/log.csv')

if __name__ == '__main__':
    
    # Run app
    main(args)
