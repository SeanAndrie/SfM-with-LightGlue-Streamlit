from .image_pairs import ImagePairsFinder
from .feature_matchers import LightGlueFeatureMatcher, LoFTRFeatureMatcher
from .keypoints_merger import KeypointsMerger
from pathlib import Path
from .database import *
from .h5_to_db import *

import os
import torch
import pycolmap

class SfMPipeline:
    def __init__(self, 
                 img_fnames:list[Path], 
                 img_fnames_str:list[str], 
                 feature_dir:str, 
                 device: torch.device, 
                 model_name:str, 
                 pretrained:str):
        self.img_fnames = img_fnames
        self.img_fnames_str = img_fnames_str
        self.feature_dir = feature_dir
        self.device = device
        self.model_name = model_name
        self.pretrained = pretrained

        detector = 'lightglue_' if model_name in ['aliked', 'superpoint', 'doghardnet', 'sift', 'disk'] else ''
        self.file_keypoints = f'{self.feature_dir}/matches_{detector}{model_name}.h5'

    def Reconstruction(self, index_pairs):
        if not os.path.exists(self.feature_dir):
            os.makedirs(self.feature_dir, exist_ok=True)

        if self.pretrained not in ['outdoor', 'indoor']:
            raise ValueError("If using LoFTR Feature Matcher, parameter `pretrained` must either be 'outdoor' or 'indoor'.")
        
        if self.model_name in ['aliked', 'superpoint', 'doghardnet', 'sift', 'disk']:
            feat_matcher = LightGlueFeatureMatcher(model_name=self.model_name,
                                                   feature_dir=self.feature_dir,
                                                   device=self.device)
        # elif self.model_name == 'loftr':
        #     feat_matcher = LoFTRFeatureMatcher(pretrained=self.pretrained, 
        #                                        feature_dir=self.feature_dir,
        #                                        device=self.device)
        
        feat_matcher.detect_and_match(img_fnames=self.img_fnames_str, 
                                      index_pairs=index_pairs)

class PipelineManager:
    def __init__(self, 
                 img_dir:str,
                 infer_file_format:bool=True, 
                 file_format:str='.jpg', 
                 output_filename:str='outputs', 
                 database_name:str='colmap.db',
                 device:torch.device=torch.device('cpu'), 
                 debug:bool=False):
        self.img_dir = Path(img_dir)    
        self.img_fnames = list(self.img_dir.glob(f'*{file_format}'))
        self.img_fnames_str = [f'{img_dir}/{fname}' for fname in os.listdir(img_dir)]

        if infer_file_format:
            file_format = f".{self.img_fnames_str[0].split('.')[-1]}"

        self.feature_dir = f'{output_filename}/{img_dir}_rec' if debug else f'{img_dir}_rec/{output_filename}'
        
        self.database_path = f'{self.feature_dir}/{database_name}'
        self.device = device

        assert self.img_dir.exists(), 'Image Directory does not exist.'
        assert file_format in ['.png', '.jpg', '.tiff'], f'PipelineManager does not support {file_format}.'

        if debug and os.path.isfile(self.database_path):
            os.remove(self.database_path)

        self.instances = {}
        self.files_keypoints = []
    
    def view_instance_attributes(self, instance_id):
        if instance_id not in self.instances:
            raise ValueError(f"Instance with ID `{instance_id}` does not exist.")
        instance = self.instances[instance_id]
        return {'img_fnames' : instance.img_fnames,
                'img_fnames_str' : instance.img_fnames_str, 
                'feature_dir' : instance.feature_dir, 
                'device' : instance.device, 
                'model_name' : instance.model_name, 
                'pretrained' : instance.pretrained}

    def create_instance(self, instance_id, model_name='aliked', pretrained='outdoor'):
        if instance_id in self.instances:
            raise ValueError(f"Instance with ID `{instance_id}` already exists.")
        self.instances[instance_id] = SfMPipeline(self.img_fnames, self.img_fnames_str, self.feature_dir,
                                                  self.device, model_name, pretrained)
        print(f"Instance `{instance_id}` created successfully.")

    def get_instance(self, instance_id):
        if instance_id not in self.instances:
            raise ValueError(f"Instance with ID `{instance_id}` does not exist.")
        return self.instances[instance_id]

    def remove_instance(self, instance_id):
        if instance_id not in self.instances:
            raise ValueError(f"Instance with ID `{instance_id}` does not exist.")
        del self.instances[instance_id]
        print(f"Instance `{instance_id}` removed successfully.")

    def list_instances(self):
        return list(self.instances.keys())
    def run_reconstruction_for_all(self):
        index_pairs = ImagePairsFinder(device=self.device).get_image_pairs(fnames=self.img_fnames)

        for instance_id, instance in self.instances.items():
            print(f'Running Reconstruction for `{instance_id}` with {instance.model_name}.')

            instance.Reconstruction(index_pairs=index_pairs)
            self.files_keypoints.append(instance.file_keypoints)

        print('Reconstruction completed for all instances.')
        print(self.files_keypoints)

        KeypointsMerger(self.feature_dir).merge_keypoints(self.img_fnames_str, 
                                                          index_pairs, 
                                                          self.files_keypoints)
        
        db = import_into_colmap(self.img_dir, 
                           self.feature_dir, 
                           self.database_path)
        pycolmap.match_exhaustive(self.database_path) # RANSAC
        
        mapper_options = pycolmap.IncrementalPipelineOptions()
        mapper_options.min_model_size = 3
        output_path = f'{self.feature_dir}/colmap_rec'
        maps = pycolmap.incremental_mapping(database_path = self.database_path, 
                                            image_path = self.img_dir, 
                                            output_path = output_path, 
                                            options = mapper_options)
        reconstruction = pycolmap.Reconstruction(f'{output_path}/0')
        reconstruction.export_PLY(f'{output_path}/rec.ply')
        print('Reconstruction Complete.')
        db.close()

        return reconstruction, output_path
    
# if __name__ == '__main__':
#     manager = PipelineManager(img_dir = 'SfM-Datasets/beethoven_converted', 
#                               device = torch.device('cuda'))
#     manager.create_instance(instance_id = 'aliked-default')
#     manager.create_instance(instance_id = 'superpoint', model_name = 'superpoint')
#     manager.run_reconstruction_for_all()