import blobconverter
import cv2
import argparse
import numpy as np
import time
import depthai as dai
from light_control import LightControl
import colorsys
import math
import random
import time


# The HeartBeat animation consists of two animations
# - a slower animation that makes the background color cycle through the color wheel
#   (e.g. every 12 seconds)
# - an a faster subsequence that plays waves like the lub dub of a heartbeat
#   (e.g. every 3 seconds)
class HeartBeat:

    def __init__(self, lc, background_period=12.0, sequence_period=3.0, wave_speed=400, randomize_people_count=False,
                 wait_for_keyboard_input=False):
        self.lc = lc

        self.background_period = background_period
        self.sequence_period = sequence_period

        self.sequence_clock = time.time() % self.sequence_period
        self.waves_to_show = 1

        self.number_of_people = 0
        self.previous_number_of_people = 0
        self.randomize_people_count = randomize_people_count
        self.wait_for_keyboard_input = wait_for_keyboard_input

        self.background_color = (0, 0, 0)
        self.complementary_color = (0, 0, 0)

        self._set_wave_speed(wave_speed)

    def _set_wave_speed(self, wave_speed):
        self.wave_speed = wave_speed
        self.main_pulse_delay = 0 / wave_speed
        self.left_pulse_delay = 100 / wave_speed
        self.right_pulse_delay = 40 / wave_speed

        # Determine sequence period from speed
        # self.sequence_period = math.ceil(400/wave_speed)
        # self.waves_to_show = int(wave_speed * self.sequence_period / 400)

        # Determine waves to show from speed
        self.waves_to_show = int(self.wave_speed / 100)

        self.wave_delay = self.sequence_period / self.waves_to_show
        print(f"{self.number_of_people} people and {self.waves_to_show} waves in {self.sequence_period} seconds")

    def _add_pulse(self, invert_r, invert_g, invert_b, complementary_color, side, pulse_start_time, skew=1):
        pulse_clock = self.sequence_clock - pulse_start_time
        if pulse_clock < 0:
            return
        frame_clock = pulse_clock * self.wave_speed
        for x in range(0, 11):
            for y in range(0, 7):
                # some adjustments to coordinates for calculation depending on side
                x2 = x
                y2 = y
                if side == 0:
                    y2 = 5 - y
                elif side == 1:
                    y2 = 5 + 8 - y
                elif side == 2:
                    x2 = 12 - x
                    y2 = 5 + 10 - y

                # use this to "draw" circle using formula x^2 + y^2 = r
                # where we later compare the calculated distance(squared) with the desired radius
                distancesq = abs((x2 ** 2 + (skew * y2) ** 2) - frame_clock)

                # adjust color of pixel within a thickness "band" from the circle
                # and vary depending on distance from center of band
                thresholdsq = 20
                if distancesq < thresholdsq:
                    distance = distancesq ** 0.5
                    threshold = thresholdsq ** 0.5
                    adjustment = 255 - min(255, int(distance * 3.2))
                    destination_color = complementary_color
                    destination_color_factor = 1
                    if distancesq > thresholdsq / 3:
                        # fade near the edges
                        destination_color_factor = (thresholdsq - distancesq) / thresholdsq
                    current_color_factor = 1 - destination_color_factor
                    self.lc.set_color(side, x, y,
                                      (int(destination_color[0] * destination_color_factor + self.background_color[
                                          0] * current_color_factor),
                                       int(destination_color[1] * destination_color_factor + self.background_color[
                                           1] * current_color_factor),
                                       int(destination_color[2] * destination_color_factor + self.background_color[
                                           2] * current_color_factor)))
                    # self.lc.add_color(side, x, y, (invert_r * adjustment, invert_g * adjustment, invert_b * adjustment))
                    # self.lc.set_color(side, x, y, complementary_color)

    def _add_wave(self, invert_r, invert_g, invert_b, complementary_color, wave_start_time):
        self._add_pulse(invert_r, invert_g, invert_b, complementary_color, 0, wave_start_time + self.main_pulse_delay,
                        skew=1.36)
        self._add_pulse(invert_r, invert_g, invert_b, complementary_color, 1, wave_start_time + self.left_pulse_delay,
                        skew=0.6)
        self._add_pulse(invert_r, invert_g, invert_b, complementary_color, 2, wave_start_time + self.right_pulse_delay,
                        skew=0.6)

    def step_frame(self):
        # vary background color varies with global time
        hue = (200 + abs(((time.time() % self.background_period) / self.background_period) * 190 - 95)) / 360
        color = colorsys.hsv_to_rgb(hue, 1.0, 0.5)
        r = int(color[0] * 255.0)
        g = int(color[1] * 255.0)
        b = int(color[2] * 255.0)
        self.lc.fill_color((r, g, b))
        self.background_color = (r, g, b)

        complementary_hue = (hue + 0.75) % 1
        complementary_color = colorsys.hsv_to_rgb(complementary_hue, 1.0, 1.0)
        complementary_color = (
            int(complementary_color[0] * 255),
            int(complementary_color[1] * 255),
            int(complementary_color[2] * 255)
        )

        # whether to add or subtract our pulse color from the background color
        # depending on how bright the color components currently are
        invert_r = 1 if r < 128 else -1
        invert_g = 1 if g < 128 else -1
        invert_b = 1 if b < 128 else -1

        # a sequence is a repeating animation with 0-3 pulse waves
        # depending on the number of people detected

        # check if we are starting a new sequence, and determine number of waves accordingly
        new_sequence_clock = time.time() % self.sequence_period
        if new_sequence_clock < self.sequence_clock:
            # get number of people from various simulated means
            if self.randomize_people_count:
                self.number_of_people = random.randint(0, 10)
                print(f"{self.number_of_people} People")
            if self.wait_for_keyboard_input:
                self.number_of_people = int(input("Number of people: "))
            # set speed based on number of people
            # self._set_wave_speed(100 + 50 * min(5, self.number_of_people))
            self._set_wave_speed(100 * min(5, int(self.number_of_people / 2) + 1))
            self.previous_number_of_people = self.number_of_people
        self.sequence_clock = new_sequence_clock

        for i in range(0, self.waves_to_show):
            self._add_wave(invert_r, invert_g, invert_b, complementary_color, wave_start_time=self.wave_delay * i)

        self.lc.show()


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-m', '--model', type=str, help='File path of .blob file.')
parser.add_argument('-v', '--video_path', type=str, default='',
                    help='Path to video. If empty OAK-RGB camera is used. (default=\'\')')
parser.add_argument('-roi', '--roi_position', type=float,
                    default=0.5, help='ROI Position (0-1)')
parser.add_argument('-a', '--axis', default=True, action='store_false',
                    help='Axis for cumulative counting (default=x axis)')
parser.add_argument('-x', '--show', default=True,
                    action='store_false', help='Show output')
parser.add_argument('-sp', '--save_path', type=str, default='',
                    help='Path to save the output. If None output won\'t be saved')
parser.add_argument('-s', '--sync', action="store_true",
                    help="Sync RGB output with NN output", default=False)
argsv = parser.parse_args()
argsv.axis = True

if argsv.model is None:
    argsv.model = blobconverter.from_zoo(name="mobilenet-ssd", shaves=7)

# Create pipeline
pipeline = dai.Pipeline()

# Define a neural network that will make predictions based on the source frames
nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
nn.setConfidenceThreshold(0.5)
nn.setBlobPath(argsv.model)
nn.setNumInferenceThreads(2)
nn.input.setBlocking(False)

# Define a source for the neural network input
if argsv.video_path != '':
    # Create XLinkIn object as conduit for sending input video file frames
    # to the neural network
    xinFrame = pipeline.create(dai.node.XLinkIn)
    xinFrame.setStreamName("inFrame")
    # Connect (link) the video stream from the input queue to the
    # neural network input
    xinFrame.out.link(nn.input)
else:
    # Create color camera node.
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(300, 300)
    cam.setInterleaved(False)
    # Connect (link) the camera preview output to the neural network input
    cam.preview.link(nn.input)

    # Create XLinkOut object as conduit for passing camera frames to the host
    xoutFrame = pipeline.create(dai.node.XLinkOut)
    xoutFrame.setStreamName("outFrame")
    cam.preview.link(xoutFrame.input)

# Create neural network output (inference) stream
nnOut = pipeline.create(dai.node.XLinkOut)
nnOut.setStreamName("nn")
nn.out.link(nnOut.input)

# Create and configure the object tracker
objectTracker = pipeline.create(dai.node.ObjectTracker)
# objectTracker.setDetectionLabelsToTrack([0])  # track only person
# possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
objectTracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
# take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
objectTracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.SMALLEST_ID)

# Link detection networks outputs to the object tracker
nn.passthrough.link(objectTracker.inputTrackerFrame)
nn.passthrough.link(objectTracker.inputDetectionFrame)
nn.out.link(objectTracker.inputDetections)

# Send tracklets to the host
trackerOut = pipeline.create(dai.node.XLinkOut)
trackerOut.setStreamName("tracklets")
objectTracker.out.link(trackerOut.input)


# from https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/
class TrackableObject:
    def __init__(self, objectID, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]

        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False





if __name__ == '__main__':

    from tkinter import Tk
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--disable-simulator", help="Disable Light Simulator", action="store_true", default=False)
    parser.add_argument("-r", "--randomize-people-count", help="Randomize People Count", action="store_true",
                        default=False)
    parser.add_argument("-k", "--wait-for-keyboard", help="Wait for Keyboard Input", action="store_true", default=False)
    parser.add_argument("-a", "--alt-mapping", help="Alt LED Mapping", action="store_true", default=False)
    args = parser.parse_args()

    lc = LightControl(simulate=(not args.disable_simulator), alt_mapping=args.alt_mapping)

    hb = HeartBeat(
        lc,
        background_period=12.0,
        sequence_period=4.0,
        wave_speed=100,
        randomize_people_count=args.randomize_people_count,
        wait_for_keyboard_input=args.wait_for_keyboard
    )


    def onKeyPress(event):
        try:
            people = int(event.char)
            print(f"{people} people detected")
            hb.number_of_people = people
        except:
            pass


    if (not args.disable_simulator):
        lc.tk_root.bind('<KeyPress>', onKeyPress)

    # Pipeline defined, now the device is connected to
    with dai.Device(pipeline) as device:

        # Define queues for image frames
        if argsv.video_path != '':
            # Input queue for sending video frames to device
            qIn_Frame = device.getInputQueue(
                name="inFrame", maxSize=4, blocking=False)
        else:
            # Output queue for retrieving camera frames from device
            qOut_Frame = device.getOutputQueue(
                name="outFrame", maxSize=4, blocking=False)

        qDet = device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        tracklets = device.getOutputQueue("tracklets", 4, False)

        if argsv.video_path != '':
            cap = cv2.VideoCapture(argsv.video_path)

        if argsv.save_path:
            if argsv.video_path != '':
                width = int(cap.get(3))
                height = int(cap.get(4))
                fps = cap.get(cv2.CAP_PROP_FPS)
            else:
                width = 300
                height = 300
                fps = 30

            out = cv2.VideoWriter(argsv.save_path, cv2.VideoWriter_fourcc(
                'M', 'J', 'P', 'G'), fps, (width, height))


        def should_run():
            return cap.isOpened() if argsv.video_path != '' else True


        def get_frame():
            if argsv.video_path != '':
                return cap.read()
            else:
                in_Frame = qOut_Frame.get()
                frame = in_Frame.getCvFrame()
                return True, frame


        startTime = time.monotonic()
        detections = []
        frame_count = 0
        counter = [0, 0, 0, 0]  # left, right, up, down

        trackableObjects = {}


        def to_planar(arr: np.ndarray, shape: tuple) -> np.ndarray:
            return cv2.resize(arr, shape).transpose(2, 0, 1).flatten()


        while should_run():
            # Get image frames from camera or video file
            read_correctly, frame = get_frame()
            if not read_correctly:
                break

            if argsv.video_path != '':
                # Prepare image frame from video for sending to device
                img = dai.ImgFrame()
                img.setType(dai.ImgFrame.Type.BGR888p)
                img.setData(to_planar(frame, (300, 300)))
                img.setTimestamp(time.monotonic())
                img.setWidth(300)
                img.setHeight(300)
                # Use input queue to send video frame to device
                qIn_Frame.send(img)
            else:
                in_Frame = qOut_Frame.tryGet()

                if in_Frame is not None:
                    frame = in_Frame.getCvFrame()
                    cv2.putText(frame, "NN fps: {:.2f}".format(frame_count / (time.monotonic() - startTime)),
                                (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color=(255, 255, 255))

            inDet = qDet.tryGet()
            if inDet is not None:
                detections = inDet.detections
                frame_count += 1

            track = tracklets.tryGet()

            if frame is not None:
                height = frame.shape[0]
                width = frame.shape[1]

                if track:
                    trackletsData = track.tracklets
                    for t in trackletsData:
                        to = trackableObjects.get(t.id, None)

                        # calculate centroid
                        roi = t.roi.denormalize(width, height)
                        x1 = int(roi.topLeft().x)
                        y1 = int(roi.topLeft().y)
                        x2 = int(roi.bottomRight().x)
                        y2 = int(roi.bottomRight().y)
                        centroid = (int((x2 - x1) / 2 + x1), int((y2 - y1) / 2 + y1))

                        # If new tracklet, save its centroid
                        if t.status == dai.Tracklet.TrackingStatus.NEW:
                            to = TrackableObject(t.id, centroid)
                        elif to is not None:
                            if argsv.axis and not to.counted:
                                x = [c[0] for c in to.centroids]
                                direction = centroid[0] - np.mean(x)

                                if centroid[0] > argsv.roi_position * width and direction > 0 and np.mean(
                                        x) < argsv.roi_position * width:
                                    counter[1] += 1
                                    hb.number_of_people+=1
                                    to.counted = True
                                elif centroid[0] < argsv.roi_position * width and direction < 0 and np.mean(
                                        x) > argsv.roi_position * width:
                                    counter[0] += 1
                                    if hb.number_of_people>1:
                                        hb.number_of_people-=1
                                    else:
                                        pass
                                    to.counted = True

                            elif not argsv.axis and not to.counted:
                                y = [c[1] for c in to.centroids]
                                direction = centroid[1] - np.mean(y)

                                if centroid[1] > argsv.roi_position * height and direction > 0 and np.mean(
                                        y) < argsv.roi_position * height:
                                    counter[3] += 1
                                    hb.number_of_people += 1
                                    to.counted = True
                                elif centroid[1] < argsv.roi_position * height and direction < 0 and np.mean(
                                        y) > argsv.roi_position * height:
                                    counter[2] += 1
                                    if hb.number_of_people>1:
                                        hb.number_of_people-=1
                                    else:
                                        pass
                                    to.counted = True

                            to.centroids.append(centroid)

                        trackableObjects[t.id] = to

                        if t.status != dai.Tracklet.TrackingStatus.LOST and t.status != dai.Tracklet.TrackingStatus.REMOVED:
                            text = "ID {}".format(t.id)
                            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                            cv2.circle(
                                frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

                # Draw ROI line
                if argsv.axis:
                    cv2.line(frame, (int(argsv.roi_position * width), 0),
                             (int(argsv.roi_position * width), height), (0xFF, 0, 0), 5)
                else:
                    cv2.line(frame, (0, int(argsv.roi_position * height)),
                             (width, int(argsv.roi_position * height)), (0xFF, 0, 0), 5)

                # display count and status
                font = cv2.FONT_HERSHEY_SIMPLEX
                if argsv.axis:
                    cv2.putText(frame, f'Left: {counter[0]}; Right: {counter[1]}', (
                        10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
                else:
                    cv2.putText(frame, f'Up: {counter[2]}; Down: {counter[3]}', (
                        10, 35), font, 0.8, (0, 0xFF, 0xFF), 2, cv2.FONT_HERSHEY_SIMPLEX)
                hb.step_frame()
                if argsv.show:
                    cv2.imshow('cumulative_object_counting', frame)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break

                if argsv.save_path:
                    out.write(frame)

        cv2.destroyAllWindows()

        if argsv.video_path != '':
            cap.release()

        if argsv.save_path:
            out.release()
