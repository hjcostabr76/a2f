import os
import re
import json
import math

import pandas as pd

'''
==================================================================
    
    Build blendshape or meta human curve .csv files
    - Parse pre-processed .txt files holding blendshape weights or meta human curves;

==================================================================
'''

# path_root = './canto-01-long'
path_root = './canto-01-short'
path_blends_txt = os.path.join(path_root, '_weight_blends.txt')
path_blends_csv = os.path.join(path_root, '_weight_blends.csv')

path_input = path_blends_txt
path_output = path_blends_csv
max_lines = None


try:
        
    file_in = open(path_input, 'r')
    max_lines = max_lines or math.inf
    df: pd.DataFrame = None

    i = -1
    for line in file_in:
        if i == max_lines - 1: break
        i = i + 1

        line = re.sub(r"^([^\[]*(\d+\:)?[^\[]*)(\[.+\])(.*)$", r"\3", line)
        weights = json.loads(line)
        
        if i == 0:
            df = pd.DataFrame(columns=[col for col in range(len(weights))])

        df.loc[i] = weights
    
    df.to_csv(path_output, index=False)

finally:
    if file_in:
        file_in.close()