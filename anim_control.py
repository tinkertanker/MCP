#!/usr/bin/env python3

"""
Animation Control Module for Marina Central Project
"""

import numpy as np


_anims = [  ['./data/anim0_animData.txt', 51],       # 0. test grid
            ['./data/y_animData.txt', 51],           # 1. Y-pose
            ['./data/tri_animData.txt', 76],         # 2. tri
            ['./data/crane_animData.txt', 151],       # 3. crane
            ['./data/leftdab_animData.txt', 51],      # 4. left dab
            ['./data/rightdab_animData.txt', 51],      # 5. right dab
            ['./data/squat_animData.txt', 126],       # 6. squat 
            ['./data/idle_Pink_animData.txt', 76],    # 7. idle pink
            ['./data/leftsuperman_animData.txt', 141]  # 8. left superman
            #['./data/rightsuperman_animData.txt', 141]    # 9. right superman
        ]
_totalAnims = len(_anims)


# 1 is normal playback,
# 2 is frame stepping mode
#   , for prev frame  
#   . for next frame  
#   numbers for changing advanceByFrames
mode = 1    
# advanceByFrames = 1


class AnimLoader:
    def __init__(self, filepath, framecount, playbackMode):
        self.filepath = filepath
        self.framecount = framecount
        self.playbackMode = playbackMode

        self.frameCounter = 0
        self.active = False

        colorData = np.loadtxt(self.filepath, delimiter=",", dtype="int")
        colorData = colorData.reshape(self.framecount,185,3)  #179
        
        print("Modified shape = ",colorData.shape)
        print("data =", colorData[6,2,:])
        print("data =", colorData[7,2,:])

        self.colorData = colorData

    def getColorData(self,pos):
        return self.colorData[self.frameCounter, pos ,:]

    def getCurrentFrame(self):
        return self.frameCounter

    def advance(self, dir=1, step=1):
        self.frameCounter += step * dir

        if(self.frameCounter >= self.framecount):
            self.frameCounter = 0
            
            if(self.playbackMode == 1):
                self.active = False

        elif(self.frameCounter <0):
            self.frameCounter = self.framecount-1
    
    def isActive(self):
        return self.active

    def setActiveState(self, s):
        self.active = s


class AnimController:
    def __init__(self, playbackMode=1):
        self.playbackMode = playbackMode
        self.totalAnims = _totalAnims
        self.anims = [None] * self.totalAnims

        for i in range(0, _totalAnims):
            self.anims[i] = AnimLoader(_anims[i][0], _anims[i][1], playbackMode)

    def getTotalAnims(self):
        return self.totalAnims

    def getAnim(self, i):
        return self.anims[i]

    def getAnimState(self, i):
        return self.anims[i].isActive()

    def getCombinedColor(self, pos):
        # cd = [None] * self.getTotalAnims()
        final_cd = [0,0,0]
        activeAnims = 0

        for i in range(0, len(self.anims)):
            if(self.anims[i].isActive()):
                activeAnims += 1

        # print('activeAnims: ', activeAnims)

        if(activeAnims == 0):
            return [0,0,0]

        else :
            p = 1/activeAnims
            
            # print('p: ', p)

            for i in range(0, self.getTotalAnims()):
                if(self.anims[i].isActive()):
                    # print('self.anims[i].isActive: ', i)
                    cd = np.asarray( self.anims[i].getColorData(pos) )
                    # print('cd: ', cd)

                    final_cd += cd * p

            final_cd = [int(final_cd[0]), int(final_cd[1]), int(final_cd[2])]
            
            return final_cd

    def printState(self):
        for i in range(0, self.getTotalAnims()):
            # animst = str(self.getAnim(i).isActive())
            animfr = str(self.getAnim(i).getCurrentFrame())

            print('== Anim', i, ':', animfr, ' ==')

    def advanceFrame(self, advanceByFrames=1):
        if(self.playbackMode == 2):  # stepping mode
            print ('stepping mode disabled')
            # ipt = input()
            # if(ipt == ','):
            #     print('<- ', advanceByFrames)
            #     animStateCounter -= advanceByFrames

            # elif(ipt == '.'):
            #     print(advanceByFrames, '->')
            #     animStateCounter += advanceByFrames

            # elif(ipt.strip().isdigit()):
            #     advanceByFrames = int(ipt)
            #     print('advanceByFrames changed to:' , advanceByFrames)


            # if(animStateCounter>=animMaxFrames):
            #     animStateCounter = 0
            # elif(animStateCounter<0):
            #     animStateCounter = animMaxFrames-1

        else:   # normal playback mode
            for i in range(0, self.getTotalAnims()):
                if(self.getAnim(i).isActive()):
                    self.getAnim(i).advance()

