#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 12 12:53:37 2022

@author: jose
"""

import pandas as pd
import librosa
from tqdm import tqdm
import soundfile as sf

import warnings
warnings.filterwarnings("ignore")

# read
df = pd.read_csv('subset.csv', header = None)

# resample and write
subset_folder = 'subset'
resample_folder = 'subset_16k'

orig_sr = 22050
target_sr = 16000
for _, wav in tqdm(df.iterrows()):
    file = '{}/{}.wav'.format(subset_folder, wav[0])
    y, sr = librosa.load(file, sr = orig_sr)
    
    output = '{}/{}.wav'.format(resample_folder, wav[0])
    y_16k = librosa.resample(y, orig_sr = sr, target_sr = target_sr)
    # librosa.output.write_wav(output, y_16k, target_sr)
    sf.write(output, y_16k, samplerate = target_sr)
