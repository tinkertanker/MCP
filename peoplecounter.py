#!/usr/bin/env python3
from pathlib import Path

import blobconverter
import cv2
import depthai as dai
import numpy as np
import argparse
from time import monotonic
import itertools

import colorsys
import math
import random
import signal
import time


from depthai_sdk import PipelineManager, NNetManager, PreviewManager
from depthai_sdk import cropToAspectRatio

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *args):
        self.kill_now = True

parentDir = Path(__file__).parent

#=====================================================================================
# To use a different NN, change `size` and `nnPath` here:
size = (544, 320)
nnPath = blobconverter.from_zoo("person-detection-retail-0013", shaves=8)
#=====================================================================================

# Labels
labelMap = ["background", "person"]
# Get argument first
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument('-nn', '--nn', type=str, help=".blob path") # unused?
parser.add_argument('-i', '--image', type=str,
                    help="Path to an image file to be used for inference (conflicts with -cam)")
parser.add_argument('-cam', '--camera', action="store_true",
                    help="Use DepthAI RGB camera for inference (conflicts with -vid)")
parser.add_argument('-f', '--peoplenum-filename', type=str,
                    help="Alternative path at which to write the number of people",
                    default='/dev/shm/millenia.txt')

args = parser.parse_args()
graceful_killer = GracefulKiller()

# Whether we want to use images from host or rgb camera
IMAGE = not args.camera
nnSource = "host" if IMAGE else "color"

# Start defining a pipeline
pm = PipelineManager()
if not IMAGE:
    pm.createColorCam(previewSize=size, xout=True)
    pv = PreviewManager(display=["color"], nnSource=nnSource, createWindows= False)

nm = NNetManager(inputSize=size, nnFamily="mobilenet", labels=labelMap, confidence=0.5)
nn = nm.createNN(pm.pipeline, pm.nodes, blobPath=nnPath, source=nnSource)
pm.setNnManager(nm)
pm.addNn(nn)

# Pipeline defined, now the device is connected to
with dai.Device(pm.pipeline) as device:
    nm.createQueues(device)

    if IMAGE:
        imgPaths = [args.image] if args.image else list(parentDir.glob('images/*.jpeg'))
        og_frames = itertools.cycle([cropToAspectRatio(cv2.imread(str(imgPath)), size) for imgPath in imgPaths])
    else:
        pv.createQueues
        pv.createQueues(device)
        pass

    while not graceful_killer.kill_now:
        try:
            nn_data = nm.decode(nm.outputQueue.get())
            with open(args.peoplenum_filename, 'w') as f:
                f.write(str(len(nn_data)))
                print(f'{len(nn_data)} people detected')
        except KeyboardInterrupt:
            break
        except:
            pass
        #nm.draw(frame, nn_data)
        #cv2.putText(frame, f"People count: {len(nn_data)}", (5, 30), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255))
        #cv2.imshow("color", frame)

        #if cv2.waitKey(3000 if IMAGE else 1) == ord('q'):
        #    break
