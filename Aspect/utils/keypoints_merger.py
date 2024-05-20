import torch 
import numpy as np
from collections import defaultdict
from copy import deepcopy
from fastprogress import progress_bar
import h5py
class KeypointsMerger:
    def __init__(self, feature_dir):
        self.feature_dir = feature_dir
        
    def get_unique_idxs(self, A, dim=0):
        unique, idx, counts = torch.unique(A, dim=dim, sorted=True, return_inverse=True, return_counts=True)
        _, ind_sorted = torch.sort(idx, stable=True)
        cum_sum = counts.cumsum(0)
        cum_sum = torch.cat((torch.tensor([0], device=cum_sum.device), cum_sum[:-1]))
        first_indices = ind_sorted[cum_sum]
        return first_indices

    def get_keypoint_from_h5(self, fp, key1, key2):
        rc = -1
        try:
            kpts = np.array(fp[key1][key2])
            rc = 0
            return (rc, kpts)
        except:
            return (rc, None)
    
    def get_keypoint_from_multi_h5(self, fps, key1, key2):
        list_mkpts = []
        for fp in fps:
            rc, mkpts = self.get_keypoint_from_h5(fp, key1, key2)
            if rc == 0:
                list_mkpts.append(mkpts)
        
        if len(list_mkpts) > 0:
            list_mkpts = np.concatenate(list_mkpts, axis = 0)
        else:
            list_mkpts = None
        return list_mkpts
    
    def merge_keypoints(self, img_fnames, index_pairs, files_keypoints):
        fps = [h5py.File(file, mode="r") for file in files_keypoints]

        with h5py.File(f'{self.feature_dir}/merge_tmp.h5', mode='w') as f_match:
            counter = 0
            for pair_idx in progress_bar(index_pairs):
                idx1, idx2 = pair_idx
                fname1, fname2 = img_fnames[idx1], img_fnames[idx2]
                key1, key2 = fname1.split('/')[-1], fname2.split('/')[-1]

                mkpts = self.get_keypoint_from_multi_h5(fps, key1, key2)
                if mkpts is None:
                    print(f"skipped key1={key1}, key2={key2}")
                    continue

                print(f'{key1}-{key2}: {mkpts.shape[0]} matches')
                group = f_match.require_group(key1)
                group.create_dataset(key2, data=mkpts)
                counter += 1

        print(f"Ensembled pairs : {counter} pairs")

        for fp in fps:
            fp.close()

        kpts = defaultdict(list)
        match_indexes = defaultdict(dict)
        total_kpts = defaultdict(int)

        with h5py.File(f'{self.feature_dir}/merge_tmp.h5', mode='r') as f_match:
            for k1 in f_match.keys():
                group = f_match[k1]
                for k2 in group.keys():
                    matches = group[k2][...]
                    kpts[k1].append(matches[:, :2])
                    kpts[k2].append(matches[:, 2:])
                    current_match = torch.arange(len(matches)).reshape(-1, 1).repeat(1, 2)
                    current_match[:, 0] += total_kpts[k1]
                    current_match[:, 1] += total_kpts[k2]
                    total_kpts[k1] += len(matches)
                    total_kpts[k2] += len(matches)
                    match_indexes[k1][k2] = current_match

        for k in kpts.keys():
            kpts[k] = np.round(np.concatenate(kpts[k], axis=0))

        unique_kpts = {}
        unique_match_idxs = {}
        out_match = defaultdict(dict)

        for k in kpts.keys():
            uniq_kps, uniq_reverse_idxs = torch.unique(torch.from_numpy(kpts[k]), dim=0, return_inverse=True)
            unique_match_idxs[k] = uniq_reverse_idxs
            unique_kpts[k] = uniq_kps.numpy()

        for k1, group in match_indexes.items():
            for k2, m in group.items():
                m2 = deepcopy(m)
                m2[:, 0] = unique_match_idxs[k1][m2[:, 0]]
                m2[:, 1] = unique_match_idxs[k2][m2[:, 1]]
                mkpts = np.concatenate([unique_kpts[k1][m2[:, 0]],
                                        unique_kpts[k2][m2[:, 1]],
                                        ],
                                       axis=1)
                unique_idxs_current = self.get_unique_idxs(torch.from_numpy(mkpts), dim=0)
                m2_semiclean = m2[unique_idxs_current]
                unique_idxs_current1 = self.get_unique_idxs(m2_semiclean[:, 0], dim=0)
                m2_semiclean = m2_semiclean[unique_idxs_current1]
                unique_idxs_current2 = self.get_unique_idxs(m2_semiclean[:, 1], dim=0)
                m2_semiclean2 = m2_semiclean[unique_idxs_current2]
                out_match[k1][k2] = m2_semiclean2.numpy()

        with h5py.File(f'{self.feature_dir}/keypoints.h5', mode='w') as f_kp:
            for k, kpts1 in unique_kpts.items():
                f_kp[k] = kpts1

        with h5py.File(f'{self.feature_dir}/matches.h5', mode='w') as f_match:
            for k1, gr in out_match.items():
                group = f_match.require_group(k1)
                for k2, match in gr.items():
                    group[k2] = match