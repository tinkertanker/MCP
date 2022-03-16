#!/usr/bin/env python3

from light_control import LightControl
import time
import numpy as np


#global vars
animStateCounter = 0
animMaxFrames = 51
globalCounter = 1   # pls start at 1 !
_anims = [  ['./data/anim0_animData.txt',51],   # test grid
            ['./data/anim1_animData.txt',51],   # circle
            ['./data/anim2_animData.txt',51],
            ['./data/anim3_animData.txt',51],
            ['./data/anim4_animData.txt',101],  # idle
            ['./data/anim5_animData.txt',101],  # right dab
            ['./data/anim6_animData.txt',76]   # squat
        ]
_totalAnims = len(_anims)
framerate = 25
playbackFramerate = 1 / framerate


# 1 is normal playback,
# 2 is frame stepping mode
#   , for prev frame  
#   . for next frame  
#   numbers for changing advanceByFrames
mode = 1    
advanceByFrames = 1



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
    def __init__(self, playbackMode, anims):
        self.totalAnims = len(anims)
        self.anims = [None] * self.totalAnims

        for i in range(0, len(anims)):
            self.anims[i] = AnimLoader(anims[i][0], anims[i][1], playbackMode)

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



# light count 63 : 71 : 45 

if __name__ == '__main__':
    lc = LightControl(simulate=True)
    # lc = LightControl(simulate=False)
    ac = AnimController(mode, _anims)

    # ac.getAnim(1).setActiveState(True)
    

    while True:
        # here we make the different anims trigger at diff times...
        if(globalCounter % 1 == 0):
            ac.getAnim(5).setActiveState(True)

        # if(globalCounter % 51 == 0):
        #     ac.getAnim(6).setActiveState(True)

        # if(globalCounter % 71 == 0):
        #     ac.getAnim(1).setActiveState(True)
        
        # if(globalCounter % 123 == 0):
        #     ac.getAnim(3).setActiveState(True)


        lc.clear()

        # for seeing the top left and btm right boundaries of each panel
        # showPanelBoundaries(lc)
        

        # Printing some Diagnostic stuff
        print('\nglobalcounter: ', globalCounter)

        animst = [0] * _totalAnims # anim state
        animfr = [0] * _totalAnims # anim frame
        
        for i in range(0, _totalAnims):
            animst[i] = str(ac.getAnim(i).isActive())
            animfr[i] = str(ac.getAnim(i).getCurrentFrame())

            print('== Anim', i, ':', animfr[i], ' ==')



        offsets1 = 0

        for x in range(0,9):
            for y in range(0,7):
                # cd = ac.getAnim(0).getColorData(x + y*9)
                cd = ac.getCombinedColor(x + y*9)
                lc.set_color(1, x, y, [ cd[0], cd[1], cd[2] ])
                
                offsets1 += 1
        
        # print('offsets1: ', offsets1)
        offsets2 = offsets1

        for x in range(0,11):
            for y in range(0,7):
                # cd2 = ac.getAnim(0).getColorData(x + (y*11) + offsets1)
                cd2 = ac.getCombinedColor(x + (y*11) + offsets1)
                # print('cd2:', cd2)
                lc.set_color(2, x, y, [ cd2[0], cd2[1], cd2[2] ])
                
                offsets2 += 1

        # print('offsets2: ', offsets2)

        for x in range(0,9):
            for y in range(0,5):
                # cd3 = ac.getAnim(0).getColorData(x + (y*9) + offsets2)
                cd3 = ac.getCombinedColor(x + (y*9) + offsets2)
                lc.set_color(0, x, y, [ cd3[0], cd3[1], cd3[2] ])

        lc.show()
        time.sleep(playbackFramerate)

        
        if(mode == 2):  # stepping mode
            ipt = input()
            if(ipt == ','):
                print('<- ', advanceByFrames)
                animStateCounter -= advanceByFrames

            elif(ipt == '.'):
                print(advanceByFrames, '->')
                animStateCounter += advanceByFrames

            elif(ipt.strip().isdigit()):
                advanceByFrames = int(ipt)
                print('advanceByFrames changed to:' , advanceByFrames)


            if(animStateCounter>=animMaxFrames):
                animStateCounter = 0
            elif(animStateCounter<0):
                animStateCounter = animMaxFrames-1

        else:   # normal playback mode
            # ac.getAnim(0).advance()
            for i in range(0, ac.getTotalAnims()):
                if(ac.getAnim(i).isActive()):
                    ac.getAnim(i).advance()

            if(animStateCounter == animMaxFrames-1):
                animStateCounter = 0
            else:
                animStateCounter += 1


        globalCounter += 1