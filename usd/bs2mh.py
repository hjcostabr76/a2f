from sklearn.linear_model import LinearRegression

import joblib
import numpy as np
import pandas as pd


'''
==================================================================

    Convert a blendshpe weights .csv file into a meta human curves based .usda file.

==================================================================
'''

# ===================================
# Config paths

# path_root = './canto-01-long'
path_root = './canto-01-short'
path_blends_csv = os.path.join(path_root, '_weight_blends.csv')
path_usda = os.path.join(path_root, 'mh_out.usda')
path_model = os.path.join(path_root, 'regressors.joblib')

# ===================================
# Convert: Blend shapes --> Meta human

regressor = joblib.load(path_model) 
df_blends = pd.read_csv(path_blends_csv)
weights_mh = regressor.predict(df_blends.values)

# ===================================
# Write .usda file

n_frames, _ = weights_mh.shape
weights_mh_str = ''
for i in range(n_frames):
    row_weights = ', '.join([f'{w:.8f}' for w in weights_mh[i, :]])
    row = f'{i}: [{row_weights}],' 
    weights_mh_str += f'\n\t\t\t\t{row}'


prim_xform='World'
prim_skew_anim='anim_canto_primeiro_usd'
frame_rate=30

usda_str = f'''
#usda 1.0
(
    defaultPrim = "{prim_xform}"
    endTimeCode = {n_frames - 1} 
    framesPerSecond = {frame_rate} 
    metersPerUnit = 0.01
    startTimeCode = 0
    timeCodesPerSecond = {frame_rate} 
)

def Xform "World"
{{
    def SkelAnimation "{prim_skew_anim}"
    {{
        token[] custom:mh_curveNames = ["CTRL_expressions_browDownL", "CTRL_expressions_browDownR", "CTRL_expressions_browLateralL", "CTRL_expressions_browLateralR", "CTRL_expressions_browRaiseInL", "CTRL_expressions_browRaiseInR", "CTRL_expressions_browRaiseOuterL", "CTRL_expressions_browRaiseOuterR", "CTRL_expressions_eyeBlinkL", "CTRL_expressions_eyeBlinkR", "CTRL_expressions_eyeWidenL", "CTRL_expressions_eyeWidenR", "CTRL_expressions_eyeSquintInnerL", "CTRL_expressions_eyeSquintInnerR", "CTRL_expressions_eyeCheekRaiseL", "CTRL_expressions_eyeCheekRaiseR", "CTRL_expressions_eyeLookUpL", "CTRL_expressions_eyeLookUpR", "CTRL_expressions_eyeLookDownL", "CTRL_expressions_eyeLookDownR", "CTRL_expressions_eyeLookLeftL", "CTRL_expressions_eyeLookLeftR", "CTRL_expressions_eyeLookRightL", "CTRL_expressions_eyeLookRightR", "CTRL_expressions_noseWrinkleL", "CTRL_expressions_noseWrinkleR", "CTRL_expressions_mouthCheekBlowL", "CTRL_expressions_mouthCheekBlowR", "CTRL_expressions_mouthLipsBlowL", "CTRL_expressions_mouthLipsBlowR", "CTRL_expressions_mouthLeft", "CTRL_expressions_mouthRight", "CTRL_expressions_mouthUpperLipRaiseL", "CTRL_expressions_mouthUpperLipRaiseR", "CTRL_expressions_mouthLowerLipDepressL", "CTRL_expressions_mouthLowerLipDepressR", "CTRL_expressions_mouthCornerPullL", "CTRL_expressions_mouthCornerPullR", "CTRL_expressions_mouthStretchL", "CTRL_expressions_mouthStretchR", "CTRL_expressions_mouthDimpleL", "CTRL_expressions_mouthDimpleR", "CTRL_expressions_mouthCornerDepressL", "CTRL_expressions_mouthCornerDepressR", "CTRL_expressions_mouthPressUL", "CTRL_expressions_mouthPressUR", "CTRL_expressions_mouthPressDL", "CTRL_expressions_mouthPressDR", "CTRL_expressions_mouthLipsPurseUL", "CTRL_expressions_mouthLipsPurseUR", "CTRL_expressions_mouthLipsPurseDL", "CTRL_expressions_mouthLipsPurseDR", "CTRL_expressions_mouthLipsTowardsUL", "CTRL_expressions_mouthLipsTowardsUR", "CTRL_expressions_mouthLipsTowardsDL", "CTRL_expressions_mouthLipsTowardsDR", "CTRL_expressions_mouthFunnelUL", "CTRL_expressions_mouthFunnelUR", "CTRL_expressions_mouthFunnelDL", "CTRL_expressions_mouthFunnelDR", "CTRL_expressions_mouthLipsTogetherUL", "CTRL_expressions_mouthLipsTogetherUR", "CTRL_expressions_mouthLipsTogetherDL", "CTRL_expressions_mouthLipsTogetherDR", "CTRL_expressions_mouthUpperLipRollInL", "CTRL_expressions_mouthUpperLipRollInR", "CTRL_expressions_mouthLowerLipRollInL", "CTRL_expressions_mouthLowerLipRollInR", "CTRL_expressions_jawOpen", "CTRL_expressions_jawLeft", "CTRL_expressions_jawRight", "CTRL_expressions_jawFwd", "CTRL_expressions_jawChinRaiseDL", "CTRL_expressions_jawChinRaiseDR"]

        float[] custom:mh_curveValues.timeSamples = {{
            {weights_mh_str}
        }}
    }}
}}'''


f = open(path_usda, 'w')
f.write(usda_str)
f.close()