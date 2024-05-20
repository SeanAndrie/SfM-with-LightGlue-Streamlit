from kornia.feature import LoFTR
from torch.utils.data import DataLoader, Dataset
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

class LoFTRDataset(Dataset):
    def __init__(self, fnames1, fnames2, idxs1, idxs2, resize_small_edge_to, device):
        self.fnames1 = fnames1
        self.fnames2 = fnames2
        self.keys1 = [ fname.split('/')[-1] for fname in fnames1 ]
        self.keys2 = [ fname.split('/')[-1] for fname in fnames2 ]
        self.idxs1 = idxs1
        self.idxs2 = idxs2
        self.resize_small_edge_to = resize_small_edge_to
        self.device = device
        self.round_unit = 16
        
    def __len__(self):
        return len(self.fnames1)

    def load_torch_image(self, fname, device):
        img = cv2.imread(fname)
        original_shape = img.shape
        ratio = self.resize_small_edge_to / min([img.shape[0], img.shape[1]])
        w = int(img.shape[1] * ratio) # int( (img.shape[1] * ratio) // self.round_unit * self.round_unit )
        h = int(img.shape[0] * ratio) # int( (img.shape[0] * ratio) // self.round_unit * self.round_unit )
        img_resized = cv2.resize(img, (w, h))
        img_resized = K.image_to_tensor(img_resized, False).float() /255.
        img_resized = K.color.bgr_to_rgb(img_resized)
        img_resized = K.color.rgb_to_grayscale(img_resized)
        return img_resized.to(device), original_shape
    
    def __getitem__(self, idx):
        fname1 = self.fnames1[idx]
        fname2 = self.fnames2[idx]
        image1, ori_shape_1 = self.load_torch_image(fname1, self.device)
        image2, ori_shape_2 = self.load_torch_image(fname2, self.device)

        return image1, image2, self.keys1[idx], self.keys2[idx], self.idxs1[idx], self.idxs2[idx], ori_shape_1, ori_shape_2

class LoFTRFeatureMatcher:
    def __init__(self, 
                 pretrained, 
                 feature_dir, 
                 resize_small_edge_to = 750, 
                 min_matches = 15, 
                 batch_size = 1, 
                 num_workers = 2, 
                 device = torch.device('cpu')):
        self.feature_dir = feature_dir
        self.resize_small_edge_to = resize_small_edge_to
        self.min_matches = min_matches
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.matcher = LoFTR(pretrained = pretrained).to(device).eval()
        self.device = device
        
        self.attach_name = 'matches_loftr.h5'

    def get_loftr_dataset(self, images1, images2, idxs1, idxs2):
        dataset = LoFTRDataset(images1, images2, idxs1, idxs2, 
                               self.resize_small_edge_to, self.device)
        dataloader = DataLoader(
            dataset = dataset,
            shuffle = False,
            batch_size = self.batch_size,
            num_workers = self.num_workers,
            drop_last = False
        )
        return dataset
    
    def detect_and_match(self, img_fnames, index_pairs):
        start_time = time()

        fnames1, fnames2, idxs1, idxs2 = [], [], [], []
        for idx1, idx2 in index_pairs:
            fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            fnames1.append(fname1)
            fnames2.append(fname2)
            idxs1.append(idx1)
            idxs2.append(idx2)

        dataloader = self.get_loftr_dataset(fnames1, fnames2, idxs1, idxs2)
        cnt_pairs = 0

        with h5py.File(f'{self.feature_dir}/{self.attach_name}', mode = 'w') as f_match:
            for X in tqdm(dataloader):
                image1, image2, key1, key2, idx1, idx2, ori_shape_1, ori_shape_2 = X
                fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
            
            with torch.inference_mode():
                correspondences = self.matcher( {"image0": image1.to(self.device),
                                                 "image1": image2.to(self.device)} )
                mkpts1 = correspondences['keypoints0'].cpu().numpy()
                mkpts2 = correspondences['keypoints1'].cpu().numpy()
                mconf  = correspondences['confidence'].cpu().numpy()

            mkpts1[:,0] *= (float(ori_shape_1[1]) / float(image1.shape[3]))
            mkpts1[:,1] *= (float(ori_shape_1[0]) / float(image1.shape[2]))

            mkpts2[:,0] *= (float(ori_shape_2[1]) / float(image2.shape[3]))
            mkpts2[:,1] *= (float(ori_shape_2[0]) / float(image2.shape[2]))

            n_matches = mconf.shape[0]

            group  = f_match.require_group(key1)
            if n_matches >= self.min_matches:
                group.create_dataset(key2, data=np.concatenate([mkpts1, mkpts2], axis=1).astype(np.float32))
                cnt_pairs+=1
                print (f'{key1}-{key2}: {n_matches} matches @ {cnt_pairs}th pair(loftr)')
            else:
                print (f'{key1}-{key2}: {n_matches} matches --> skipped')
        gc.collect()
        t=time()-start_time 
        print(f'Features matched in  {t:.4f} sec')

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