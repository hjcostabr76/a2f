import maya.cmds as mc
import json

with open(r'C:\tmp\a2f_bs_weights.json', "r") as f:
    facs_data = json.loads(f.read())
    facsNames = facs_data["facsNames"]
    numPoses = facs_data["numPoses"]
    numFrames = facs_data["numFrames"]    
    weightMat = facs_data["weightMat"]
    mc.playbackOptions(ast=0, min=0, max=numFrames-1, aet=numFrames-1)
    
    bsNode = 'blendShape1'
    for fr in range(numFrames):
        for i in range(numPoses):
            mc.setKeyframe(bsNode+'.'+facsNames[i], v=weightMat[fr][i], t=fr)