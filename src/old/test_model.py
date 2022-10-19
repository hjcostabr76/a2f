import torch
import torch.autograd as autograd

import os
import time
import numpy as np
import argparse
from scipy.signal import savgol_filter

from dataset import BlendshapeDataset
from models import A2BNet, NvidiaNet, LSTMNvidiaNet, FullyLSTM

# options
# parser = argparse.ArgumentParser(description="PyTorch testing of LSTM")
# parser.add_argument('model_best', type=str)
# parser.add_argument('--smooth', type=bool, default=False)
# parser.add_argument('--pad', type=bool, default=False)
# parser.add_argument('--epoch', type=int, default=None)
# parser.add_argument('--net', type=str, default='lstm')

# args = parser.parse_args()

# parameters
n_blendshape = 46
batch_size = 100

# data path
dataroot = 'file'
modelo_id = 'LSTMNvidiaNet'
# dataset = 'Emma'

data_path = os.path.join(dataroot, 'emma-test')
checkpoint_path = os.path.join(dataroot, 'emma-checkpoint', modelo_id)
val_path = os.path.join(dataroot, 'emma-combine', 'val')


result_path = os.path.join(dataroot, 'results')
result_file = os.path.join(result_path, f'{modelo_id}_hat.txt')

blend_shape_file = 'blendshape.txt'
feature_file = 'feature-lpc.npy'

ckp = 'checkpoint-model_best.pth.tar'
# if args.epoch != None:
#     ckp = 'checkpoint-epoch'+str(args.epoch)+'.pth.tar'
#     result_file = str(args.epoch)+'-'+result_file
# else:
#     ckp = args.ckp+'.pth.tar'

def pad_blendshape(blendshape):
    return np.pad(blendshape, [(16, 16), (0, 0)], mode='constant', constant_values=0.0)

model = LSTMNvidiaNet(num_blendshapes = n_blendshape)

# restore checkpoint model
checkpoint = torch.load(os.path.join(checkpoint_path, ckp))

print("=> loading checkpoint '{}'".format(ckp))
print("model epoch {} loss: {}".format(checkpoint['epoch'], checkpoint['eval_loss']))

model.load_state_dict(checkpoint['state_dict'])

# load data
val_loader = torch.utils.data.DataLoader(
    BlendshapeDataset(
        # feature_file=os.path.join(data_path, 'Emma_val_feature-lpc.npy'),
        feature_file=os.path.join(val_path, feature_file),
        target_file=os.path.join(val_path, blend_shape_file)
    ),
    batch_size=batch_size, shuffle=False, num_workers=2
)

if torch.cuda.is_available():
    model = model.cuda()
    print('cuda')
else:
    print('-'*10 + '!! NO CUDA !!' + '-'*10)

# run test features
model.eval()

start_time = time.time()
with torch.no_grad():
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda(non_blocking=True)
        input_var = autograd.Variable(input.float()).cuda()
        target_var = autograd.Variable(target.float())

        # compute output
        output = model(input_var)

        if i == 0:
            output_cat = output.data
        else:
            output_cat = torch.cat((output_cat, output.data), 0)
        # print(type(output_cat.cpu().numpy()), output_cat.cpu().numpy().shape)

# convert back *100
output_cat = output_cat.cpu().numpy()*100.0

# if args.smooth:
#     #smooth3--savgol_filter
#     win = 9; polyorder = 3
#     for i in range(n_blendshape):
#         power = output_cat[:,i]
#         power_smooth = savgol_filter(power, win, polyorder, mode='nearest')
#         output_cat[:, i] = power_smooth
#     result_file = 'smooth-' + result_file

# # padding to the same frames as input wav
# if args.pad:
#     output_cat = pad_blendshape(output_cat)
#     result_file = 'pad-' + result_file

# count time for testing
past_time = time.time() - start_time

with open(result_file, 'wb') as f:
    np.savetxt(f, output_cat, fmt='%.6f')

print("Test finished in {:.4f} sec! Saved in {}".format(past_time, result_file))