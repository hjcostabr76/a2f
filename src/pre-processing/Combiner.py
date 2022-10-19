import os
import json
import paths
import numpy as np

class Combiner:
    
    def __init__(self, feature_dir: str, blendshape_dir: str):
        self.feature_dir = feature_dir
        self.blendshape_dir = blendshape_dir
    
    def combine(file_names: list) -> tuple[np.array, np.array]:
        '''
            TODO: 2022-10-11 - ADD Description
            PARAMS
            - file_names: List of file names (can be either blendshapes or features considering they're named in the standard way).
        '''

        for i in range(len(file_names)):
            
            base_name = file_names[i].split('.')[0]
            
            try:
                if i == 0:
                    feature, blendshape = self.__get_matrices(base_name=base_name)
                    continue
                
                feature_temp, blendshape_temp = self.__get_matrices(base_name=base_name)
                feature = np.concatenate((feature, feature_temp), 0)                    
                blendshape = np.concatenate((blendshape, blendshape_temp), 0)
            
            except AssertionError as error:
                print(error)
                print(f'failed files: "{base_name}"!')

        return feature, blendshape
    
    def __get_matrices(self, base_name: str) -> tuple[np.array, np.array]:
        '''
            Return the matrix numerical representations of blendshapes & audio feature files corresponding
            to one same audio / video track. 
        '''
    
        # Features
        path_feat = os.path.join(self.feature_dir, paths.feature_file(base_name))
        feature = np.load(path_feat)

        # Blendshapes
        path_blendshape = os.path.join(self.blendshape_dir, paths.blendshape_file(base_name))

        f = open(path_blendshape)
        data = json.load(f)
        blendshape = np.array(data['weightMat'])
        f.close()
        
        blendshape = self.__cut(feature, blendshape)
        
        return feature, blendshape

    def __cut(self, wav_feature: np.array, blendshape_target: np.array) -> list:
        '''
            TODO: 2022-10-11 - Check this shit out!
        '''
        n_audioframe, n_videoframe = len(wav_feature), len(blendshape_target)
        assert n_videoframe - n_audioframe in [31, 32, 33]
        start_videoframe = 16
        blendshape_target = blendshape_target[start_videoframe : start_videoframe+n_audioframe]
        return blendshape_target

    