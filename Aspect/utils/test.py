from image_pairs import ImagePairsFinder
from feature_matchers import LightGlueFeatureMatcher, LoFTRFeatureMatcher

from pathlib import Path
import os
import torch

img_dir = Path('SfM-Datasets/bird_images')
img_fnames = list(img_dir.glob('*.jpg'))
img_fnames_str = [f'{img_dir}/{fname}' for fname in os.listdir(img_dir)]
feature_dir = 'outputs/{}'.format(str(img_dir).split("\\")[-1])

if not os.path.exists(feature_dir):
    os.makedirs(feature_dir, exist_ok = True)

IPF = ImagePairsFinder(device = torch.device('cuda'))
index_pairs = IPF.get_image_pairs(img_fnames)

lg_feat_matcher = LightGlueFeatureMatcher(model_name = 'aliked', 
                                          feature_dir = feature_dir, 
                                          device = torch.device('cuda'))
t = lg_feat_matcher.detect_and_match(img_fnames_str, 
                                     index_pairs)

