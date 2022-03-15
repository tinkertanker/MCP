import cv2
from math import atan2, degrees
import sys

sys.path.append("")
from MovenetDepthaiEdge import MovenetDepthai, KEYPOINT_DICT
from MovenetRenderer import MovenetRenderer
import argparse
import numpy as np
import os
import csv
import threading


class EMADictSmoothing(object):
    """Smoothes pose classification."""

    def __init__(self, window_size=10, alpha=0.2):
        self._window_size = window_size
        self._alpha = alpha

        self._data_in_window = []

    def __call__(self, data):
        """Smoothes given pose classification.

        Smoothing is done by computing Exponential Moving Average for every pose
        class observed in the given time window. Missed pose classes arre replaced
        with 0.

        Args:
          data: Dictionary with pose classification. Sample:
              {
                'pushups_down': 8,
                'pushups_up': 2,
              }

        Result:
          Dictionary in the same format but with smoothed and float instead of
          integer values. Sample:
            {
              'pushups_down': 8.3,
              'pushups_up': 1.7,
            }
        """
        # Add new data to the beginning of the window for simpler code.
        self._data_in_window.insert(0, data)
        self._data_in_window = self._data_in_window[:self._window_size]

        # Get all keys.
        keys = set([key for data in self._data_in_window for key, _ in data.items()])

        # Get smoothed values.
        smoothed_data = dict()
        for key in keys:
            factor = 1.0
            top_sum = 0.0
            bottom_sum = 0.0
            for data in self._data_in_window:
                value = data[key] if key in data else 0.0

                top_sum += factor * value
                bottom_sum += factor

                # Update factor.
                factor *= (1.0 - self._alpha)

            smoothed_data[key] = top_sum / bottom_sum

        return smoothed_data


class PoseSample(object):

    def __init__(self, name, landmarks, class_name, embedding):
        self.name = name
        self.landmarks = landmarks
        self.class_name = class_name

        self.embedding = embedding


class PoseSampleOutlier(object):

    def __init__(self, sample, detected_class, all_classes):
        self.sample = sample
        self.detected_class = detected_class
        self.all_classes = all_classes


class FullBodyPoseEmbedder(object):
    """Converts 3D pose landmarks into 3D embedding."""

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

        # Names of the landmarks as they appear in the prediction.
        self._landmark_names = [
            'nose',
            'left_eye',
            'right_eye',
            'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
        ]

    def __call__(self, landmarks):
        """Normalizes pose landmarks and converts to embedding

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances defined in `_get_pose_distance_embedding`.
        """
        # print(landmarks.shape[0])
        # print(len(self._landmark_names))
        assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(
            landmarks.shape[0])

        # Get pose landmarks.
        landmarks = np.copy(landmarks)

        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)

        # Get embedding. HERE
        embedding = self._get_pose_distance_embedding(landmarks)

        return embedding

    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale."""
        landmarks = np.copy(landmarks)

        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it eaasier to debug.
        landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # This approach uses only 2D landmarks to compute pose size.
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding. All distances
        include X and Y components with sign. We differnt types of pairs to cover
        different pose classes. Feel free to remove some or add new.

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances.
        """
        embedding = np.array([
            # One joint.

            self._get_distance(
                self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
                self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

            self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_knee', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_knee', 'right_ankle'),

            # Two joints.

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_ankle'),

            # Four joints.

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Five joints.

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Cross body.

            self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
            self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
            self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),

            # Body bent direction.

            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
            #     landmarks[self._landmark_names.index('left_hip')]),
            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
            #     landmarks[self._landmark_names.index('right_hip')]),
        ])

        return embedding

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from


class PoseClassifier(object):
    """Classifies pose landmarks."""

    def __init__(self,
                 pose_samples_folder,
                 pose_embedder,
                 file_extension='csv',
                 file_separator=',',
                 n_landmarks=17,
                 n_dimensions=2,
                 top_n_by_max_distance=30,
                 top_n_by_mean_distance=10,
                 axes_weights=(1., 1.)):
        self._pose_embedder = pose_embedder
        self._n_landmarks = n_landmarks
        self._n_dimensions = n_dimensions
        self._top_n_by_max_distance = top_n_by_max_distance
        self._top_n_by_mean_distance = top_n_by_mean_distance
        self._axes_weights = axes_weights

        self._pose_samples = self._load_pose_samples(pose_samples_folder,
                                                     file_extension,
                                                     file_separator,
                                                     n_landmarks,
                                                     n_dimensions,
                                                     pose_embedder)

    def _load_pose_samples(self,
                           pose_samples_folder,
                           file_extension,
                           file_separator,
                           n_landmarks,
                           n_dimensions,
                           pose_embedder):
        """Loads pose samples from a given folder.

        Required folder structure:
          neutral_standing.csv
          pushups_down.csv
          pushups_up.csv
          squats_down.csv
          ...

        Required CSV structure:
          sample_00001,x1,y1,x2,y2,....
          sample_00002,x1,y1,x2,y2,....
          ...
        """
        # Each file in the folder represents one pose class.
        file_names = [name for name in os.listdir(pose_samples_folder) if name.endswith(file_extension)]

        pose_samples = []
        for file_name in file_names:
            # Use file name as pose class name.
            class_name = file_name[:-(len(file_extension) + 1)]

            # Parse CSV.
            with open(os.path.join(pose_samples_folder, file_name)) as csv_file:
                csv_reader = csv.reader(csv_file, delimiter=file_separator)
                for row in csv_reader:
                    # print(len(row))
                    # print(n_landmarks * n_dimensions + 1)
                    assert len(row) == n_landmarks * n_dimensions + 1, 'Wrong number of values: {}'.format(len(row))
                    landmarks = np.array(row[1:], np.float32).reshape([n_landmarks, n_dimensions])
                    pose_samples.append(PoseSample(
                        name=row[0],
                        landmarks=landmarks,
                        class_name=class_name,
                        embedding=pose_embedder(landmarks),
                    ))

        return pose_samples

    def find_pose_sample_outliers(self):
        """Classifies each sample against the entire database."""
        # Find outliers in target poses
        outliers = []
        for sample in self._pose_samples:
            # Find nearest poses for the target one.
            pose_landmarks = sample.landmarks.copy()
            pose_classification = self.__call__(pose_landmarks)
            class_names = [class_name for class_name, count in pose_classification.items() if
                           count == max(pose_classification.values())]

            # Sample is an outlier if nearest poses have different class or more than
            # one pose class is detected as nearest.
            if sample.class_name not in class_names or len(class_names) != 1:
                outliers.append(PoseSampleOutlier(sample, class_names, pose_classification))

        return outliers

    def __call__(self, pose_landmarks):
        """Classifies given pose.

        Classification is done in two stages:
          * First we pick top-N samples by MAX distance. It allows to remove samples
            that are almost the same as given pose, but has few joints bent in the
            other direction.
          * Then we pick top-N samples by MEAN distance. After outliers are removed
            on a previous step, we can pick samples that are closes on average.

        Args:
          pose_landmarks: NumPy array with 3D landmarks of shape (N, 3).

        Returns:
          Dictionary with count of nearest pose samples from the database. Sample:
            {
              'pushups_down': 8,
              'pushups_up': 2,
            }
        """
        # Check that provided and target poses have the same shape.
        assert pose_landmarks.shape == (self._n_landmarks, self._n_dimensions), 'Unexpected shape: {}'.format(
            pose_landmarks.shape)

        # Get given pose embedding.
        pose_embedding = self._pose_embedder(pose_landmarks)
        flipped_pose_embedding = self._pose_embedder(pose_landmarks * np.array([1, 1]))

        # Filter by max distance.
        #
        # That helps to remove outliers - poses that are almost the same as the
        # given one, but has one joint bent into another direction and actually
        # represnt a different pose class.
        max_dist_heap = []
        for sample_idx, sample in enumerate(self._pose_samples):
            max_dist = min(
                np.max(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.max(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights),
            )
            max_dist_heap.append([max_dist, sample_idx])

        max_dist_heap = sorted(max_dist_heap, key=lambda x: x[0])
        max_dist_heap = max_dist_heap[:self._top_n_by_max_distance]

        # Filter by mean distance.
        #
        # After removing outliers we can find the nearest pose by mean distance.
        mean_dist_heap = []
        for _, sample_idx in max_dist_heap:
            sample = self._pose_samples[sample_idx]
            mean_dist = min(
                np.mean(np.abs(sample.embedding - pose_embedding) * self._axes_weights),
                np.mean(np.abs(sample.embedding - flipped_pose_embedding) * self._axes_weights),
            )
            mean_dist_heap.append([mean_dist, sample_idx])

        mean_dist_heap = sorted(mean_dist_heap, key=lambda x: x[0])
        mean_dist_heap = mean_dist_heap[:self._top_n_by_mean_distance]

        # Collect results into map: (class_name -> n_samples)
        class_names = [self._pose_samples[sample_idx].class_name for _, sample_idx in mean_dist_heap]
        result = {class_name: class_names.count(class_name) for class_name in set(class_names)}

        # print(result)

        return result


def recognize_pose(b):
    pose_embedder = FullBodyPoseEmbedder()

    pose_classifier = PoseClassifier(
        pose_samples_folder='./fitness_poses_csvs_out_processed_f',
        pose_embedder=pose_embedder,
        top_n_by_max_distance=30,
        top_n_by_mean_distance=10)

    # assert b.keypoints.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(b.keypoints.shape)

    # print(b.keypoints)
    # print(type(b.keypoints))

    b.keypoints = b.keypoints.astype('float32')

    pose_classification = pose_classifier(b.keypoints)

    pose_classification_filter = EMADictSmoothing(
        window_size=10,
        alpha=0.2)

    # Smooth classification using EMA.
    pose_classification_filtered = pose_classification_filter(pose_classification)

    max_sample = 0
    pose = 0

    for i in pose_classification_filtered.keys():
        if pose_classification_filtered[i] > max_sample:
            pose = i
            max_sample = pose_classification_filtered[i]

    posef = pose
    return [posef, list(pose_classification_filtered.items())]


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, choices=['lightning', 'thunder'], default='thunder',
                    help="Model to use (default=%(default)s")
parser.add_argument('-i', '--input', type=str, default='rgb',
                    help="'rgb' or 'rgb_laconic' or path to video/image file to use as input (default: %(default)s)")
parser.add_argument("-o", "--output",
                    help="Path to output video file")
args = parser.parse_args()

pose = MovenetDepthai(input_src=args.input, model=args.model)
#renderer = MovenetRenderer(pose, output=args.output)

info_set = []



# !/usr/bin/env python3

from light_control import LightControl
import time
import numpy as np

# global vars
animStateCounter = 0
animMaxFrames = 51
globalCounter = 1  # pls start at 1 !
totalAnims = 3
playbackFramerate = 25 / 1000

# 1 is normal playback,
# 2 is frame stepping mode
#   , for prev frame
#   . for next frame
#   numbers for changing advanceByFrames
mode = 1
advanceByFrames = 1


def showPanelBoundaries(lc):
    lc.set_color(1, 0, 0, [255, 255, 255])
    lc.set_color(1, 8, 6, [255, 255, 255])
    lc.set_color(2, 0, 0, [255, 255, 255])
    lc.set_color(2, 9, 6, [255, 255, 255])
    lc.set_color(0, 0, 0, [255, 255, 255])
    lc.set_color(0, 8, 4, [255, 255, 255])


class AnimLoader:
    def __init__(self, filepath, framecount, playbackMode):
        self.filepath = filepath
        self.framecount = framecount
        self.playbackMode = playbackMode

        self.frameCounter = 0
        self.active = False

        colorData = np.loadtxt(self.filepath, delimiter=",", dtype="int")
        colorData = colorData.reshape(self.framecount, 185, 3)  # 179

        print("Modified shape = ", colorData.shape)
        print("data =", colorData[42, 2, :])

        self.colorData = colorData

    def getColorData(self, pos):
        return self.colorData[self.frameCounter, pos, :]

    def getCurrentFrame(self):
        return self.frameCounter

    def advance(self, dir=1, step=1):
        self.frameCounter += step * dir

        if (self.frameCounter >= self.framecount):
            self.frameCounter = 0

            if (self.playbackMode == 1):
                self.active = False

        elif (self.frameCounter < 0):
            self.frameCounter = self.framecount - 1

    def isActive(self):
        return self.active

    def setActiveState(self, s):
        self.active = s


class AnimController:
    def __init__(self, playbackMode):
        self.anims = [None, None, None]
        self.anims[0] = AnimLoader('./data/anim1_animData.txt', 51, playbackMode)
        self.anims[1] = AnimLoader('./data/anim2_animData.txt', 51, playbackMode)
        self.anims[2] = AnimLoader('./data/anim3_animData.txt', 51, playbackMode)

    def getTotalAnims(self):
        return len(self.anims)

    def getAnim(self, i):
        return self.anims[i]

    def getAnimState(self, i):
        return self.anims[i].isActive()

    def getCombinedColor(self, pos):
        cd = [None, None, None]
        final_cd = [0, 0, 0]
        activeAnims = 0

        for i in range(0, len(self.anims)):
            if (self.anims[i].isActive()):
                activeAnims += 1

        if (activeAnims == 0):
            return [0, 0, 0]

        else:
            p = 1 / activeAnims

            # print('p: ', p)

            for i in range(0, len(self.anims)):
                cd[i] = np.asarray([0, 0, 0])

                if (self.anims[i].isActive):
                    cd[i] = np.asarray(self.anims[i].getColorData(pos))
                    # print('cd[i]: ', cd[i])

                    final_cd += cd[i] * p
            def limiter(num):
                if num>255:
                    return 255
                elif num<0:
                    return 0
                else:
                    return int(num)

            final_cd = [limiter(final_cd[0]), limiter(final_cd[1]), limiter(final_cd[2])]

            return final_cd


# light count 63 : 71 : 45


lc = LightControl(simulate=False)
ac = AnimController(mode)
# ac.getAnim(0).setActiveState(True)

pose_state="default"

def recog(lock):
    global pose_state
    while True:
        #print( "recog")
        #print(threading.current_thread())
        # Run blazepose on next frame
        frame, body = pose.next_frame()
        if frame is None: break
        # Draw 2d skeleton
        #frame = renderer.draw(frame, body)
        # Gesture recognition
        pose_info = recognize_pose(body)
        pose1 = pose_info[0]
        info_set = list(pose.crop_region[1:5]) + pose_info
        print(info_set)
        lock.acquire()
        pose_state = info_set[4]
        lock.release()
        if pose1:
            cv2.putText(frame, pose1, (frame.shape[1] // 2, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 190, 255), 3)
        #key = renderer.waitKey(delay=1)
        #if key == 27 or key == ord('q'):
        #    break
    #renderer.exit()
    pose.exit()

def lightcon():
    global pose_state
    global animStateCounter
    while True:
        #print( "LC")
        #print(threading.current_thread())
        #lock.acquire()
        # here we make the different anims trigger at diff times...
        if (pose_state == "leftdab" or pose_state == "rightdab"):
            ac.getAnim(0).setActiveState(True)

        elif (pose_state == "squat"):
            ac.getAnim(1).setActiveState(True)

        else:
            ac.getAnim(2).setActiveState(True)
        #lock.release()
        lc.clear()

        # for seeing the top left and btm right boundaries of each panel
        # showPanelBoundaries(lc)

        # Printing some Diagnostic stuff
        #print('\nglobalcounter: ', globalCounter)

        animst = [0, 0, 0]
        animfr = [0, 0, 0]

        for i in range(0, ac.getTotalAnims()):
            animst[i] = str(ac.getAnim(i).isActive())
            animfr[i] = str(ac.getAnim(i).getCurrentFrame())

        print('anim states: ', animst[0], ':', animfr[0], ' ', animst[1], ':', animfr[1], ' ', animst[2], ':',
              animfr[2])

        offsets1 = 0

        for x in range(0, 9):
            for y in range(0, 7):
                # cd = ac.getAnim(0).getColorData(x + y*9)
                cd = ac.getCombinedColor(x + y * 9)
                lc.set_color(1, x, y, [cd[0], cd[1], cd[2]])

                offsets1 += 1

        # print('offsets1: ', offsets1)
        offsets2 = offsets1

        for x in range(0, 11):
            for y in range(0, 7):
                # cd2 = ac.getAnim(0).getColorData(x + (y*11) + offsets1)
                cd2 = ac.getCombinedColor(x + (y * 11) + offsets1)
                lc.set_color(2, x, y, [cd2[0], cd2[1], cd2[2]])

                offsets2 += 1

        # print('offsets2: ', offsets2)

        for x in range(0, 9):
            for y in range(0, 5):
                # cd3 = ac.getAnim(0).getColorData(x + (y*9) + offsets2)
                cd3 = ac.getCombinedColor(x + (y * 9) + offsets2)
                lc.set_color(0, x, y, [cd3[0], cd3[1], cd3[2]])

        lc.show()
        time.sleep(playbackFramerate)

        if (mode == 2):  # stepping mode
            ipt = input()
            if (ipt == ','):
                print('<- ', advanceByFrames)
                animStateCounter -= advanceByFrames

            elif (ipt == '.'):
                print(advanceByFrames, '->')
                animStateCounter += advanceByFrames

            elif (ipt.strip().isdigit()):
                advanceByFrames = int(ipt)
                print('advanceByFrames changed to:', advanceByFrames)

            if (animStateCounter >= animMaxFrames):
                animStateCounter = 0
            elif (animStateCounter < 0):
                animStateCounter = animMaxFrames - 1

        else:  # normal playback mode
            # ac.getAnim(0).advance()
            for i in range(0, ac.getTotalAnims()):
                if (ac.getAnim(i).isActive()):
                    ac.getAnim(i).advance()

            if (animStateCounter == animMaxFrames - 1):
                animStateCounter = 0
            else:
                animStateCounter += 1



if __name__ == '__main__':
    # print ID of current process
    print("ID of process running main program: {}".format(os.getpid()))

    # print name of main thread
    print("Main thread name: {}".format(threading.current_thread().name))

    lock = threading.Lock()

    # creating threads
    t1 = threading.Thread(target=recog, name='t1',args=(lock,))
    #t2 = threading.Thread(target=lightcon, name='t2',args=(lock,))
    # starting threads
    t1.start()
    lightcon()