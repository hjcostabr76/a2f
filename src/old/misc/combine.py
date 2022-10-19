import numpy as np
import os
from tqdm import tqdm
import json


problems_too_short = [ # Files with less then 64 frames
    'Chapter_19-104',
    'Chapter_25-121',
    'Chapter_42-166',
]

problems_frame_diff = [ # Files whose wav frames count is not 32 frames more than blend shape frames count
    'Chapter_14-97',
    'Chapter_19-126',
    'Chapter_19-4',
    'Chapter_2-13',
    'Chapter_2-159',
    'Chapter_20-3',
    'Chapter_33-1',
    'Chapter_36-30',
    'Chapter_39-14',
    'Chapter_39-254',
    'Chapter_40-7',
    'Chapter_44-213',
    'Chapter_47-148',
    'Chapter_5-116',
    'Chapter_53-23',
    'Chapter_56-18',
]

test_files = [
    'Chapter_35-51',
    'Chapter_17-74',
    'Chapter_50-47',
    'Chapter_52-120',
    'Chapter_9-219',
    'Chapter_18-18',
    'Chapter_27-370',
    'Chapter_36-68',
    'Chapter_39-250'
]

skipped_files = test_files + problems_frame_diff + problems_too_short



dataset = 'emma'
dataroot = '../file'
wav_path = os.path.join(dataroot, 'emma-subset-16k')
feature_path = os.path.join(dataroot, 'feature-lpc')
target_path = os.path.join(dataroot, 'emma-a2f-json')
combine_path = os.path.join(dataroot, 'emma-combine')

# def combine(feature_path, target_path):
def combine(feature_files: list, blendshape_files: list) -> tuple:
    
    print(f'parsing {len(feature_files)} feature_files, {len(blendshape_files)} blendshape_files, ')

    for i in tqdm(range(len(feature_files))):
    # for i in range(len(problems)):
        
        # Skip bad files
        # base_name = problems[i]
        base_name = blendshape_files[i].split('.')[0]
        if base_name in skipped_files:
            continue
        
        file_blendshape = base_name + '.json'
        file_feat = base_name + '-lpc.npy'
        
        path_json = os.path.join(target_path, file_blendshape)
        path_feat = os.path.join(feature_path, file_feat)

        try:
            if i == 0:
                feature = np.load(path_feat)
                feature_combine_file = dataset + '_' + feature_path.split('/')[-2] + '.npy'

                # blendshape is shorter, need cut
                blendshape = loadjson(path_json)
                blendshape = cut(feature, blendshape)

                blendshape_combine_file = dataset + '_' + target_path.split('/')[-2] + '.txt'

            else:
                feature_temp = np.load(path_feat)
                feature = np.concatenate((feature, feature_temp), 0)

                # blendshape is shorter
                blendshape_temp = loadjson(path_json)
                blendshape_temp = cut(feature_temp, blendshape_temp)

                blendshape = np.concatenate((blendshape, blendshape_temp), 0)
        
        except AssertionError as error:
            print(error)
            print(f'failed files: "{file_blendshape}" / "{file_feat}"')

        # print(i, blendshape_files[i], feature.shape, blendshape.shape)

    # np.save(os.path.join(feature_path, feature_combine_file), feature)
    # np.savetxt(os.path.join(target_path, blendshape_combine_file), blendshape, fmt='%.8f')
    return feature, blendshape

def cut(wav_feature, blendshape_target):
    n_audioframe, n_videoframe = len(wav_feature), len(blendshape_target)
    # print('--------\n', 'Current dataset -- n_audioframe: {}, n_videoframe:{}'.format(n_audioframe, n_videoframe))
    # print(f'diff: {n_videoframe - n_audioframe}')
    assert n_videoframe - n_audioframe in [31, 32, 33]
    start_videoframe = 16
    blendshape_target = blendshape_target[start_videoframe : start_videoframe+n_audioframe]

    return blendshape_target

def loadjson(blendshape_file):
    f = open(blendshape_file)
    data = json.load(f)
    f.close()
    return np.array(data['weightMat'])

def main():
    
    feature_files = sorted(os.listdir(feature_path))
    blendshape_files = sorted(os.listdir(target_path))

    
    # Validation
    n_validation = round(.05 * len(feature_files))
    print(len(feature_files))
    print(f'n_validation: {n_validation}')
    return

    feat_files_val = feature_files[0:n_validation]
    blendshape_files_val = blendshape_files[0:n_validation]
    feat_val, blendshape_val = combine(feature_files=feat_files_val, blendshape_files=blendshape_files_val)
    
    path_val_feat = os.path.join(combine_path, 'val', 'feature-lpc.npy')
    path_val_blendshape = os.path.join(combine_path, 'val', 'blendshape.txt')
    np.save(path_val_feat, feat_val)
    np.savetxt(path_val_blendshape, blendshape_val, fmt='%.8f')

    # print(f'n_validation: {n_validation}')
    # Train
    feat_files_train = feature_files[n_validation:]
    blendshape_files_train = blendshape_files[n_validation:]
    feat_train, blendshape_train = combine(feature_files=feat_files_train, blendshape_files=blendshape_files_train)
    
    path_train_feat = os.path.join(combine_path, 'train', 'feature-lpc.npy')
    path_train_blendshape = os.path.join(combine_path, 'train', 'blendshape.txt')
    np.save(path_train_feat, feat_train)
    np.savetxt(path_train_blendshape, blendshape_train, fmt='%.8f')

if __name__ == '__main__':
    main()
