"""
/PoseRecognition.py
Uses the .blob output from MCP_pose_classification.py (run on Colab)
"""

import marshal
import numpy as np
import cv2
from collections import namedtuple
from pathlib import Path
from FPS import FPS
import depthai as dai
import time
from MovenetDepthaiEdge import Body

from math import gcd
from string import Template

SCRIPT_DIR = Path(__file__).resolve().parent
POSE_RECOGNITION_MODEL = SCRIPT_DIR / "models/pose_recognition.blob"

class PoseRecogniser:
    def __init__(self,
                model=None,
                body=None,
                device=None):

        self.model = str(POSE_RECOGNITION_MODEL)
        print(f"Using blob file: {self.model}")
    
    def infer(self, body):
      pipeline = dai.Pipeline()
      
      xin_nn = pipeline.create(dai.node.XLinkIn)
      xin_nn.setStreamName("pose_in")
      
      # create pose recognition neural network
      pr_nn = pipeline.create(dai.node.NeuralNetwork)
      pr_nn.setBlobPath(str(Path(self.model).resolve().absolute()))
      xin_nn.out.link(pr_nn.input)
      
      xout_nn = pipeline.create(dai.node.XLinkOut)
      xout_nn.setStreamName("pose_out")
      pr_nn.out.link(xout_nn.input)
      
      self.device = dai.Device(pipeline)
      q_pose_in = self.device.getInputQueue(name="pose_in")
      q_pose_in.send(body)
      q_pose_out = self.device.getOutputQueue(name="pose_out", maxSize=4, blocking=False)
      
      return q_pose_out