#!/usr/bin/env python3

from light_control import LightControl
import time
import numpy as np
from anim_control import AnimController


#   global vars
isSim = True
selectAnim = 6
framerate = 25

#   fixed global vars
playbackFramerate = 1 / framerate
globalCounter = 1   # pls start at 1 !



def setLightColor(xRange, yRange, animctrl, offset, lightctrl, panelNum):
    for x in range(0, xRange):
            for y in range(0, yRange):
                cd = animctrl.getCombinedColor(x + (y * xRange) + offset)
                lightctrl.set_color(panelNum, x, y, [ cd[0], cd[1], cd[2] ])


# light count 63 : 71 : 45 

if __name__ == '__main__':
    lc = LightControl(simulate=isSim)
    ac = AnimController()


    while True:
        # here we make the different anims trigger at diff times...
        if(globalCounter % 1 == 0):
            ac.getAnim(selectAnim).setActiveState(True)


        # Printing some Diagnostic stuff
        print('\nglobalcounter: ', globalCounter)

        ac.printState()
        

        lc.clear()
        setLightColor(9, 7, ac, 0, lc, 1)
        setLightColor(11, 7, ac, 63, lc, 2)
        setLightColor(9, 5, ac, 140, lc, 0)
        

        lc.show()
        time.sleep(playbackFramerate)

        ac.advanceFrame()
        
        globalCounter += 1