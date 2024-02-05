import os
import sys
import shutil
import random
import errno

import torch
import numpy as np
from torch.optim import AdamW


def build_optimizer(args, model):

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    return optimizer


def load_checkpoint(model_file):
    if os.path.isfile(model_file):
        print("=> loading model '{}'".format(model_file))
        checkpoint = torch.load(model_file)
        return checkpoint
    else:
        print("=> no model found at '{}'".format(model_file))
        raise OSError(errno.ENOENT, os.strerror(errno.ENOENT), model_file)


def save_checkpoint(state, directory, file_name):
    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, file_name + '.pth')
    torch.save(state, checkpoint_file)


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dir(root_save_path):
    if os.path.exists(root_save_path):
        shutil.rmtree(root_save_path)  # delete output folder
    os.makedirs(root_save_path)  # make new output folder


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def get_output_file(path, extension="png"):
    """
    Assumes sketch file in path have suffix -{number}.
    Finds the highest number and return path for next sketch.
    """
    sketch_id = 0
    for file_name in os.listdir(path):
        try:
            id_on_file = int(file_name.split('-')[-1][:-4])
            if id_on_file > sketch_id:
                    sketch_id = id_on_file
        except:
            pass
    sketch_id += 1
    sketch_file = f"sketch-{sketch_id}.{extension}"

    return os.path.join(path, sketch_file), sketch_id
