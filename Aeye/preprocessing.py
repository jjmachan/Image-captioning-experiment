# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_Preprocessing.ipynb (unless otherwise specified).

__all__ = ['flickr8k_img_dir', 'annotation_file', 'dataset_train']

# Cell
from torchvision import models
from torchvision import transforms
import torch

# Cell
flickr8k_img_dir = '/home/jithin/datasets/imageCaptioning/flicker8k/Flicker8k_Dataset'
annotation_file = '/home/jithin/datasets/imageCaptioning/captions/dataset_flickr8k.json'

dataset_train = Flickr8k(flickr8k_img_dir, annotation_file, split="train", transform=transform)