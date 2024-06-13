from fastprogress import progress_bar
import torch
import h5py
from tqdm import tqdm
import numpy as np
import kornia as K
import kornia.feature as KF
import os
import gc
import cv2

from time import time, sleep
from lightglue import ALIKED, SuperPoint, DoGHardNet, DISK, SIFT

class LightGlueFeatureMatcher:
    def __init__(self, 
                model_name, 
                feature_dir,
                resize_to = 1024,
                detection_threshold = 0.001, 
                num_features = 8192, 
                min_matches = 15,
                device = torch.device('cpu')):
        self.model_name = model_name
        self.feature_dir = feature_dir
        self.resize_to = resize_to
        self.detection_threshold = detection_threshold
        self.num_features = num_features
        self.min_matches = min_matches
        self.device = device

        self.attach_name = f'matches_lightglue_{model_name}.h5'

        self.dict_model = {
            'aliked': ALIKED, 
            'superpoint': SuperPoint, 
            'doghardnet': DoGHardNet, 
            'disk' : DISK, 
            'sift' : SIFT, 
        }
        
        self.extractor_class = self.dict_model[model_name]
        self.dtype = torch.float32
        self.extractor = self.extractor_class(
            max_num_keypoints = num_features,
            detection_threshold = detection_threshold, 
            resize = resize_to
        ).eval().to(device, torch.float32)

        self.lg_matcher = KF.LightGlueMatcher(model_name, 
                                              {'width_confidence': -1, 'depth_confidence': -1,
                                               'mp': True if 'cuda' in str(self.device) else False}).eval().to(self.device)
        
    def _detect_features_(self, img_fnames):  
        if not os.path.isdir(self.feature_dir):
            os.makedirs(self.feature_dir)

        with h5py.File(f'{self.feature_dir}/keypoints_{self.model_name}.h5', mode = 'w') as f_kp, \
             h5py.File(f'{self.feature_dir}/descriptors_{self.model_name}.h5', mode = 'w') as f_desc:
        
            for img_path in tqdm(img_fnames):
                img_fname = img_path.split('/')[-1]
                key = img_fname
                with torch.inference_mode():
                    image0 = self.load_torch_image(img_path, device = self.device).to(torch.float32)
                    feats0 = self.extractor.extract(image0)
                    kpts = feats0['keypoints'].reshape(-1, 2).detach().cpu().numpy()
                    descs = feats0['descriptors'].reshape(len(kpts), -1).detach().cpu().numpy()
                    
                    f_kp[key] = kpts
                    f_desc[key] = descs
                    print(f'{self.model_name} > kpts.shape = {kpts.shape}, descs.shape = {descs.shape}')
        return
    
    def load_torch_image(self, img_path, device = torch.device('cpu')):
        """Loads an image and adds batch dimension"""
        img = K.io.load_image(img_path, K.io.ImageLoadType.RGB32, device = device)[None, ...]
        return img

    def _match_features_(self, img_fnames, index_pairs):
        cnt_pairs = 0
        with h5py.File(f'{self.feature_dir}/keypoints_{self.model_name}.h5', mode = 'r') as f_kp, \
             h5py.File(f'{self.feature_dir}/descriptors_{self.model_name}.h5', mode = 'r') as f_desc, \
             h5py.File(f'{self.feature_dir}/{self.attach_name}', mode = 'w') as f_match:
            for pair_idx in tqdm(index_pairs):
                idx1, idx2 = pair_idx
                fname1, fname2 = img_fnames[idx1], img_fnames[idx2]

                key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]
                kp1 = torch.from_numpy(f_kp[key1][...]).to(self.device)
                kp2 = torch.from_numpy(f_kp[key2][...]).to(self.device)
                desc1 = torch.from_numpy(f_desc[key1][...]).to(self.device)
                desc2 = torch.from_numpy(f_desc[key2][...]).to(self.device)
                with torch.inference_mode():
                    dists, idxs = self.lg_matcher(desc1, 
                                                  desc2, 
                                                  KF.laf_from_center_scale_ori(kp1[None]),
                                                  KF.laf_from_center_scale_ori(kp2[None]))
                if len(idxs) == 0:
                    continue
                n_matches = len(idxs)
                kp1 = kp1[idxs[:, 0], :].cpu().numpy().reshape(-1, 2).astype(np.float32)
                kp2 = kp2[idxs[:, 1], :].cpu().numpy().reshape(-1, 2).astype(np.float32)
                group = f_match.require_group(key1)

                if n_matches >= self.min_matches:
                    group.create_dataset(key2, data=np.concatenate([kp1, kp2], axis=1))
                    cnt_pairs += 1
                    print(f'{key1}-{key2}: {n_matches} matches @ {cnt_pairs}th pair({self.model_name}+lightglue)')
                else:
                    print(f'{key1}-{key2}: {n_matches} matches --> skipped')

    def detect_and_match(self, img_fnames, index_pairs):
        start_time = time()
        self._detect_features_(img_fnames)
        gc.collect()
        self._match_features_(img_fnames, index_pairs)
        end_time = time()
        print(f'Features matched in {end_time - start_time:.4f} sec ({self.model_name}+LightGlue)')