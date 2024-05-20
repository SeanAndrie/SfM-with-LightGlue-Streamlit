import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
from PIL import Image
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from pathlib import Path
import numpy as np
import kagglehub

class ImagePairsFinder:
    def __init__(self, 
                 sim_th=0.6, 
                 min_pairs=20, 
                 exhaustive_if_less=20, 
                 device=torch.device('cpu')):
        self.sim_th = sim_th
        self.min_pairs = min_pairs
        self.exhaustive_if_less = exhaustive_if_less
        self.device = device

        model_path = kagglehub.model_download('timm/tf-efficientnet/pyTorch/tf-efficientnet-b7')
        model = timm.create_model(model_name = 'tf_efficientnet_b7', 
                                  checkpoint_path = f'{model_path}/tf_efficientnet_b7_ra-6c08e654.pth')
        self.model = model.eval().to(device)

    def get_global_descriptors(self, fnames):
        config = resolve_data_config({}, model=self.model)
        transform = create_transform(**config)
        global_descs_convnext = []
        for i, img_fname_full in tqdm(enumerate(fnames), total=len(fnames)):
            key = os.path.splitext(os.path.basename(img_fname_full))[0]
            img = Image.open(img_fname_full).convert('RGB')
            timg = transform(img).unsqueeze(0).to(self.device)
            with torch.inference_mode(), torch.cuda.amp.autocast():
                desc = self.model.forward_features(timg).mean(dim=(-1, 2))
                desc = desc.view(1, -1)
                desc_norm = F.normalize(desc, dim=1, p=2)
                global_descs_convnext.append(desc_norm.detach().cpu())
        global_descs_all = torch.cat(global_descs_convnext, dim=0)
        return global_descs_all.to(torch.float32)

    def _convert_1d_to_2d_(self, idx, num_images):
        idx1 = idx // num_images
        idx2 = idx % num_images  
        return (idx1, idx2)
    
    def _get_pairs_from_distancematrix_(self, mat):
        pairs = [self._convert_1d_to_2d_(idx, mat.shape[0]) for idx in np.argsort(mat.flatten())]
        pairs = [pair for pair in pairs if pair[0] < pair[1]]
        return pairs
    
    def _get_image_pairs_exhaustive_(self, img_fnames):
        descs = self.get_global_descriptors(img_fnames)
        dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
        matching_list = self._get_pairs_from_distancematrix_(dm)
        return matching_list
    
    def get_image_pairs(self, fnames):
        num_imgs = len(fnames)
        if num_imgs <= self.exhaustive_if_less:
            return self._get_image_pairs_exhaustive_(fnames)
        
        descs = self.get_global_descriptors(fnames)
        dm = torch.cdist(descs, descs, p=2).detach().cpu().numpy()
        mask = dm <= self.sim_th
        total = 0
        matching_list = []
        ar = np.arange(num_imgs)

        for st_idx in range(num_imgs - 1):
            mask_idx = mask[st_idx]
            to_match = ar[mask_idx]
            if len(to_match) < self.min_pairs:
                to_match = np.argsort(dm[st_idx])[:self.min_pairs]
            for idx in to_match:
                if st_idx == idx:
                    continue
                if dm[st_idx, idx] < 1000:
                    matching_list.append(tuple(sorted((st_idx, idx.item()))))
                    total += 1

        matching_list = sorted(list(set(matching_list)))
        return matching_list
    