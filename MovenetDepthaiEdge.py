from distutils.log import debug
import marshal
import re
import numpy as np
import cv2
from collections import namedtuple
from pathlib import Path
from FPS import FPS
import depthai as dai
import time
import blobconverter
from models.class_names import CLASS_NAMES

from math import gcd
from string import Template

SCRIPT_DIR = Path(__file__).resolve().parent
MOVENET_LIGHTNING_MODEL = SCRIPT_DIR / "models/movenet_singlepose_lightning_U8_transpose.blob"
MOVENET_THUNDER_MODEL = SCRIPT_DIR / "models/movenet_singlepose_thunder_U8_transpose.blob"
POSE_RECOGNITION_MODEL_XML = SCRIPT_DIR / "models/pose_recognition.xml"
POSE_RECOGNITION_MODEL_BIN = SCRIPT_DIR / "models/pose_recognition.bin"

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

class Body:
    def __init__(self, scores=None, keypoints_norm=None, keypoints=None, score_thresh=None, crop_region=None, next_crop_region=None):
        """
        Attributes:
        scores : scores of the keypoints
        keypoints_norm : keypoints normalized ([0,1]) coordinates (x,y) in the squared cropped region
        keypoints : keypoints coordinates (x,y) in pixels in the source image
        score_thresh : score threshold used
        crop_region : cropped region on which the current body was inferred
        next_crop_region : cropping region calculated from the current body keypoints and that will be used on next frame
        """
        self.scores = scores 
        self.keypoints_norm = keypoints_norm 
        self.keypoints = keypoints
        self.score_thresh = score_thresh
        self.crop_region = crop_region
        self.next_crop_region = next_crop_region

    def print(self):
        attrs = vars(self)
        print('\n'.join("%s: %s" % item for item in attrs.items()))

CropRegion = namedtuple('CropRegion',['xmin', 'ymin', 'xmax',  'ymax', 'size']) # All values are in pixel. The region is a square of size 'size' pixels

def find_isp_scale_params(size, is_height=True):
    """
    Find closest valid size close to 'size' and and the corresponding parameters to setIspScale()
    This function is useful to work around a bug in depthai where ImageManip is scrambling images that have an invalid size
    is_height : boolean that indicates if the value is the height or the width of the image
    Returns: valid size, (numerator, denominator)
    """
    # We want size >= 288
    if size < 288:
        size = 288
    
    # We are looking for the list on integers that are divisible by 16 and
    # that can be written like n/d where n <= 16 and d <= 63
    if is_height:
        reference = 1080 
        other = 1920
    else:
        reference = 1920 
        other = 1080
    size_candidates = {}
    for s in range(288,reference,16):
        f = gcd(reference, s)
        n = s//f
        d = reference//f
        if n <= 16 and d <= 63 and int(round(other * n / d) % 2 == 0):
            size_candidates[s] = (n, d)
            
    # What is the candidate size closer to 'size' ?
    min_dist = -1
    for s in size_candidates:
        dist = abs(size - s)
        if min_dist == -1:
            min_dist = dist
            candidate = s
        else:
            if dist > min_dist: break
            candidate = s
            min_dist = dist
    return candidate, size_candidates[candidate]

    

class MovenetDepthai:
    """
    Movenet body pose detector
    Arguments:
    - input_src: frame source, 
                    - "rgb" or None: OAK* internal color camera,
                    - "rgb_laconic": same as "rgb" but without sending the frames to the host,
    - model: Movenet blob file,
                    - "thunder": the default thunder blob file (see variable MOVENET_THUNDER_MODEL),
                    - "lightning": the default lightning blob file (see variable MOVENET_LIGHTNING_MODEL),
                    - a path of a blob file. It is important that the filename contains 
                    the string "thunder" or "lightning" to identify the tyoe of the model.
    - score_thresh : confidence score to determine whether a keypoint prediction is reliable (a float between 0 and 1).
    - crop : boolean which indicates if systematic square cropping to the smaller side of 
                    the image is done or not,
    - smart_crop : boolen which indicates if cropping from previous frame detection is done or not,
    - internal_fps : when using the internal color camera as input source, set its FPS to this value (calling setFps()).
    - internal_frame_height : when using the internal color camera, set the frame height (calling setIspScale()).
                                The width is calculated accordingly to height and depends on value of 'crop'
    - stats : True or False, when True, display the global FPS when exiting.            
    """
    def __init__(self, input_src="rgb",
                model=None, 
                score_thresh=0.2,
                crop=False,
                smart_crop = True,
                internal_fps=None,
                internal_frame_height=640,
                stats=True):

        self.model = model 
        
        
        if model == "lightning":
            self.model = str(MOVENET_LIGHTNING_MODEL)
            self.pd_input_length = 192
        elif model == "thunder":
            self.model = str(MOVENET_THUNDER_MODEL)
            self.pd_input_length = 256
        else:
            self.model = model
            if "lightning" in str(model):
                self.pd_input_length = 192
            else: # Thunder
                self.pd_input_length = 256
        print(f"Using blob file : {self.model}")

        print(f"MoveNet imput size : {self.pd_input_length}x{self.pd_input_length}x3")

        self.score_thresh = score_thresh   
        
        self.crop = crop
        self.smart_crop = smart_crop
        self.internal_fps = internal_fps
        self.stats = stats
        
        if input_src is None or input_src == "rgb" or input_src == "rgb_laconic":
            self.input_type = "rgb" # OAK* internal color camera
            self.laconic = "laconic" in input_src # Camera frames are not sent to the host
            if internal_fps is None:
                if "thunder" in str(model):
                    self.internal_fps = 12
                else:
                    self.internal_fps = 26
            else:
                self.internal_fps = internal_fps
            print(f"Internal camera FPS set to: {self.internal_fps}")

            self.video_fps = self.internal_fps # Used when saving the output in a video file. Should be close to the real fps
            
            if self.crop:
                self.frame_size, self.scale_nd = find_isp_scale_params(internal_frame_height)
                self.img_h = self.img_w = self.frame_size
            else:
                width, self.scale_nd = find_isp_scale_params(internal_frame_height * 1920 / 1080, is_height=False)
                self.img_h = int(round(1080 * self.scale_nd[0] / self.scale_nd[1]))
                self.img_w = int(round(1920 * self.scale_nd[0] / self.scale_nd[1]))
                self.frame_size = self.img_w
            print(f"Internal camera image size: {self.img_w} x {self.img_h}")

        else:
            print(f"Input source '{input_src}' is not supported in edge mode !")
            print("Valid input sources: 'rgb', 'rgb_laconic'")
            import sys
            sys.exit()

        # Defines the default crop region (pads the full image from both sides to make it a square image) 
        # Used when the algorithm cannot reliably determine the crop region from the previous frame.
        box_size = max(self.img_w, self.img_h)
        x_min = (self.img_w - box_size) // 2
        y_min = (self.img_h - box_size) // 2
        self.init_crop_region = CropRegion(x_min, y_min, x_min+box_size, y_min+box_size, box_size)
        self.crop_region = self.init_crop_region
        
        self.device = dai.Device(self.create_pipeline())
        print("Pipeline started")

        # Define data queues 
        if not self.laconic:
            self.q_video = self.device.getOutputQueue(name="cam_out", maxSize=1, blocking=False)
        self.q_processing_out = self.device.getOutputQueue(name="processing_out", maxSize=4, blocking=False)
        self.q_pose_out = self.device.getOutputQueue(name="pose_out", maxSize=1, blocking=False)
        self.q_debug_out = self.device.getOutputQueue(name="debug_out", maxSize=1, blocking=False)
        # For debugging
        # self.q_manip_out = self.device.getOutputQueue(name="manip_out", maxSize=1, blocking=False)
   
        self.fps = FPS()

        self.nb_frames = 0
        self.nb_pd_inferences = 0


    def create_pipeline(self):
        print("Creating pipeline...")
        # Start defining a pipeline
        pipeline = dai.Pipeline()
        pipeline.setOpenVINOVersion(dai.OpenVINO.Version.VERSION_2021_4)

        # ColorCamera
        print("Creating Color Camera...")
        cam = pipeline.create(dai.node.ColorCamera) 
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setInterleaved(False)
        cam.setIspScale(self.scale_nd[0], self.scale_nd[1])
        cam.setFps(self.internal_fps)
        cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        if self.crop:
            cam.setVideoSize(self.frame_size, self.frame_size)
            cam.setPreviewSize(self.frame_size, self.frame_size)
        else: 
            cam.setVideoSize(self.img_w, self.img_h)
            cam.setPreviewSize(self.img_w, self.img_h)
        
        if not self.laconic:
            cam_out = pipeline.create(dai.node.XLinkOut)
            cam_out.setStreamName("cam_out")
            cam.video.link(cam_out.input)

        # ImageManip for cropping
        manip = pipeline.create(dai.node.ImageManip)
        manip.setMaxOutputFrameSize(self.pd_input_length*self.pd_input_length*3)
        manip.setWaitForConfigInput(True)
        manip.inputImage.setQueueSize(1)
        manip.inputImage.setBlocking(False)            

        cam.preview.link(manip.inputImage)

        # For debugging
        # manip_out = pipeline.createXLinkOut()
        # manip_out.setStreamName("manip_out")
        # manip.out.link(manip_out.input)

        # Define pose detection model
        print("Creating Pose Detection Neural Network...")
        pd_nn = pipeline.create(dai.node.NeuralNetwork)
        pd_nn.setBlobPath(str(Path(self.model).resolve().absolute()))
        # pd_nn.input.setQueueSize(1)
        # Increase threads for detection
        # pd_nn.setNumInferenceThreads(2)

        manip.out.link(pd_nn.input)

        # Define processing script
        processing_script = pipeline.create(dai.node.Script)
        processing_script.setScript(self.build_processing_script())
        
        pd_nn.out.link(processing_script.inputs['from_pd_nn'])
        processing_script.outputs['to_manip_cfg'].link(manip.inputConfig)
        
        # Define link to send result to host 
        processing_out = pipeline.create(dai.node.XLinkOut)
        processing_out.setStreamName("processing_out")
        processing_script.outputs['to_host'].link(processing_out.input)
        
        # Define neural network to recognize pose
        print("creating pose recognition neural network")
        pr_nn = pipeline.create(dai.node.NeuralNetwork)
        pose_blob_path = str(Path(SCRIPT_DIR / "models/pose_recognition.blob").resolve().absolute())
        pose_blob_path = blobconverter.from_openvino(
            xml=str(Path(POSE_RECOGNITION_MODEL_XML).resolve().absolute()),
            bin=str(Path(POSE_RECOGNITION_MODEL_BIN).resolve().absolute()),
            data_type="FP16",
            shaves=6,
        )
        pr_nn.setBlobPath(pose_blob_path)
        processing_script.outputs['to_pr_nn'].link(pr_nn.input) 
        
        # Define link to send pose to host
        pose_out = pipeline.create(dai.node.XLinkOut)
        pose_out.setStreamName("pose_out")
        pr_nn.out.link(pose_out.input)

        # Debug
        debug_out = pipeline.create(dai.node.XLinkOut)
        debug_out.setStreamName("debug_out")
        processing_script.outputs['to_pr_nn'].link(debug_out.input)


        print("Pipeline created.")

        return pipeline        

    def build_processing_script(self):
        '''
        The code of the scripting node 'processing_script' depends on :
            - the NN model (thunder or lightning),
            - the score threshold,
            - the video frame shape
        So we build this code from the content of the file template_processing_script.py which is a python template
        '''
        # Read the template
        with open('template_processing_script.py', 'r') as file:
            template = Template(file.read())
        
        # Perform the substitution
        code = template.substitute(
                    _init_crop_region = str(self.init_crop_region._asdict()).replace("OrderedDict", "dict"),
                    _pd_input_length = self.pd_input_length,
                    _score_thresh = self.score_thresh,
                    _img_w = self.img_w,
                    _img_h = self.img_h,
                    _smart_crop = self.smart_crop
        )
        # Remove comments and empty lines
        import re
        code = re.sub(r'"{3}.*?"{3}', '', code, flags=re.DOTALL)
        code = re.sub(r'#.*', '', code)
        code = re.sub('\n\s*\n', '\n', code)
        # For debuging
        with open("tmp_code.py", "w") as file:
            file.write(code)

        return code

    def pd_postprocess(self, inference):
        result = marshal.loads(inference.getData())
        scores = np.array(result["scores"])
        keypoints_norm = np.array(list(zip(result["xnorm"], result["ynorm"])))
        keypoints = np.array(list(zip(result["x"], result["y"])))
        next_crop_region = CropRegion(**result["next_crop_region"])
        body = Body(scores, keypoints_norm, keypoints, self.score_thresh, self.crop_region, next_crop_region)
        return body
    
    def pr_postprocess(self, detection):
        pred = detection.getFirstLayerFp16()
        print(pred)
        pred_label = CLASS_NAMES[np.argmax(pred)]
        return pred_label

    def next_frame(self):

        self.fps.update()

        # Get the device camera frame if wanted
        if self.laconic:
            frame = np.zeros((self.frame_size, self.frame_size, 3), dtype=np.uint8)
        else:
            in_video = self.q_video.get()
            frame = in_video.getCvFrame()

        # For debugging
        # manip = self.q_manip_out.get().getCvFrame()
        # cv2.imshow("manip", manip)

        # Get result from device
        inference = self.q_processing_out.get()
        body = self.pd_postprocess(inference)
        det = self.q_pose_out.get()
        detection = self.pr_postprocess(det)
        self.crop_region = body.next_crop_region

        debug = self.q_debug_out.get()
        #print(debug.getData())

        # Statistics
        if self.stats:
            self.nb_frames += 1
            self.nb_pd_inferences += 1

        return frame, body, detection


    def exit(self):
        # Print some stats
        if self.stats:
            print(f"FPS : {self.fps.global_duration():.1f} f/s (# frames = {self.fps.nb_frames()})")
           