import os
from config import dataroot 

__all__ = ['wav_file', 'blendshape_file', 'feature_file', 'combined_features', 'combined_blendshapes', 'checkpoint_file', 'model_file']

## ----------------------------------------

wav_dir = os.path.join(dataroot, 'wav')
blendshape_dir = os.path.join(dataroot, 'blendshape')
feature_dir = os.path.join(dataroot, 'feature')
dataset_dir = os.path.join(dataroot, 'dataset')
model_dir = os.path.join(dataroot, 'model')

## ----------------------------------------

def _assert_training_stage(stage: str) -> None:
    if type not in ['train', 'val', 'test']:
        raise ValueError(f"training stage type should be 'train', 'val', 'test' ('{type}' given)")

def _get_file_name(path: str) -> str:
    pathname, extension = os.path.splitext(path)
    return pathname.split('/')[-1]

def wav_file(file_name: str) -> str:
    file_name = _get_file_name(file_name) + '.wav'
    return os.path.join(wav_dir, file_name)

def blendshape_file(file_name: str) -> str:
    file_name = _get_file_name(file_name) + '.json'
    return os.path.join(blendshape_dir, file_name)

def feature_file(file_name: str) -> str:
    file_name = _get_file_name(file_name) + '-lpc.npy'
    return os.path.join(feature_dir, file_name)

def combined_features(type: str = 'train') -> None:
    _assert_training_stage(type)
    return os.path.join(paths.dataset_dir, type, 'combined-features.npy')

def combined_blendshapes(type: str = 'train') -> None:
    _assert_training_stage(type)
    return os.path.join(paths.dataset_dir, type, 'combined-blendshapes.npy')

def checkpoint_file(epoch: int, is_best = False) -> str:
    file_name = 'model-checkpoint-epoch-{:03}\r'.format(epoch + 1)
    if is_best: file_name += '-best'
    file_name += '.pth.tar'
    return os.path.join(model_dir, file_name)

def model_file(file_name: str) -> str:
    return os.path.join(model_dir, file_name)
