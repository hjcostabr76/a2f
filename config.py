import paths
import os

n_blendshapes = 46
n_workers = 6 # TODO: ADD desc

dataroot = './file/emma'

# ===============================================================
# -- Inference parameters

inference_model_path = paths.model_file('...')
inference_features_path = ''
inference_result_path = ''

# ===============================================================
# -- Training parameters

batch_size = 100
learning_rate = 0.0001
epochs = 500
best_loss = 10000000
val_rate = .05

print_freq = 20 # TODO: ADD Desc
checkpoint_freq = 100 # Iterations count between partial models (checkpoints) generation


# ===============================================================
# -- Set skipped files among the inputs

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
    'Chapter_17-74',
    'Chapter_18-18',
    'Chapter_27-370',
    'Chapter_35-51',
    'Chapter_36-68',
    'Chapter_39-250'
    'Chapter_50-47',
    'Chapter_52-120',
    'Chapter_9-219',
]

skipped_files = test_files + problems_frame_diff + problems_too_short