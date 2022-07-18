#!/usr/bin/env python3

import argparse
import colorsys
from src.stream_analyzer import Stream_Analyzer
from light_control import LightControl
import socket
import time
import math

light_to_bin_map = [
        [ # side 0,
            [ 19, 00, 00, 00, 00, 16, 16, 19, 19 ],
            [ 19, 00, 00, 00, 00, 16, 16, 19, 19 ],
            [ 19,  9,  9, 00, 00,  5,  5, 19, 19 ],
            [ 19,  9,  9, 00, 00,  5,  5, 19, 19 ],
            [ 19, 19, 19, 19, 19, 19, 19, 19, 19 ],
        ],
        [ # side 1 (left)
            [ 19, 19, 19, 19, 19, 19, 19, 19, 19 ],
            [ 19, 19, 19,  1,  1, 12, 12, 19, 19 ],
            [ 19, 19, 19,  1,  1, 12, 12, 19, 19 ],
            [ 19,  2,  2,  8,  8,  4,  4,  4, 19 ],
            [ 19,  2,  2,  8,  8,  4,  4,  4, 19 ],
            [ 19, 15, 15, 19, 19,  4,  4,  4, 19 ],
            [ 19, 15, 15, 19, 19, 19, 19, 19, 19 ],
        ],
        [ # side 2 (right)
            [ 19, 19, 19, 19, 17, 17, 19, 19, 13, 13, 13],
            [ 19, 19, 19, 19, 17, 17, 19, 19, 13, 13, 13],
            [ 14, 14, 10, 10,  6,  6, 11, 11, 13, 13, 13],
            [ 14, 14, 10, 10,  6,  6, 11, 11, 19, 19, 19],
            [ 19, 19,  7,  7, 18, 18,  3,  3,  3, 19, 19],
            [ 19, 19,  7,  7, 18, 18,  3,  3,  3, 19, 19],
            [ 19, 19, 19, 19, 19, 19,  3,  3,  3, 19, 19],
        ]
    ]

bin_to_light_map = [[] for i in range (20)]

for side in range(len(light_to_bin_map)):
    for y in range(len(light_to_bin_map[side])):
        for x in range(len(light_to_bin_map[side][y])):
            bin_to_light_map[light_to_bin_map[side][y][x]].append((side,x,y))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=None, dest='device',
                        help='pyaudio (portaudio) device index')
    parser.add_argument('--height', type=int, default=450, dest='height',
                        help='height, in pixels, of the visualizer window')
    parser.add_argument('--n_frequency_bins', type=int, default=20, dest='frequency_bins',
                        help='The FFT features are grouped in bins')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--window_ratio', default='24/9', dest='window_ratio',
                        help='float ratio of the visualizer window. e.g. 24/9')
    parser.add_argument('--hostname', type=str, help="Override Hostname", default='')
    parser.add_argument("-a", "--alt-mapping", help="Alt LED Mapping", action="store_true",default=False)
    parser.add_argument("-s", "--simulate", help="Enable Light Simulator", action="store_true", default=False)
    return parser.parse_args()

def convert_window_ratio(window_ratio):
    if '/' in window_ratio:
        dividend, divisor = window_ratio.split('/')
        try:
            float_ratio = float(dividend) / float(divisor)
        except:
            raise ValueError('window_ratio should be in the format: float/float')
        return float_ratio
    raise ValueError('window_ratio should be in the format: float/float')

def map_energy_to_hue(hostname, energy):
    if hostname in ["marinapi3", "marinapi2"]:
        if energy < 3:
            return 220 - ((energy - 1) * (220 - 180) / 2.0) # 290 down to 180
        else:
            return 70 - ((energy - 3) * (70 - 40) / 2.0) # 70 down to 40
    else:
        if energy < 3:
            return 160 - ((energy - 1) * (160 - 70) / 2.0) # 160 down to 70
        else:
            return (380 - ((energy - 3) * (380 - 300) / 2.0)) % 360 # 20 down to 300

def run_FFT_analyzer():
    args = parse_args()
    hostname = args.hostname or socket.gethostname()

    lc = LightControl(simulate=args.simulate, alt_mapping=args.alt_mapping)

    for x in range(0,9):
        for y in range(0,7):
            lc.set_color(1, x, y, [x*25, y*25, 200])
            lc.show()
            time.sleep(0.01)

    for x in range(0,11):
        for y in range(0,7):
            lc.set_color(2, x, y, [x*25, 200, y*25])
            lc.show()
            time.sleep(0.01)

    for x in range(0,9):
        for y in range(0,5):
            lc.set_color(0, x, y, [200, x*25, y*25])
            lc.show()
            time.sleep(0.01)

    window_ratio = convert_window_ratio(args.window_ratio)

    ear = Stream_Analyzer(
                    device = args.device,        # Pyaudio (portaudio) device index, defaults to first mic input
                    rate   = None,               # Audio samplerate, None uses the default source settings
                    FFT_window_size_ms  = 60,    # Window size used for the FFT transform
                    updates_per_second  = 1000,  # How often to read the audio stream for new data
                    smoothing_length_ms = 50,    # Apply some temporal smoothing to reduce noisy features
                    n_frequency_bins = args.frequency_bins, # The FFT features are grouped in bins
                    visualize = 0,               # Visualize the FFT features with PyGame
                    verbose   = args.verbose,    # Print running statistics (latency, fps, ...)
                    height    = args.height,     # Height, in pixels, of the visualizer window,
                    window_ratio = window_ratio  # Float ratio of the visualizer window. e.g. 24/9
                    )

    fps = 15  #How often to update the FFT features + display
    last_update = time.time()
    energy_threshold_low = 1
    energy_threshold_high = 5
    while True:
        if (time.time() - last_update) > (1./fps):
            last_update = time.time()
            raw_fftx, raw_fft, binned_fftx, binned_fft = ear.get_audio_features()
            frequency_bin_energies = ear.frequency_bin_energies / ear.bin_mean_values
            for i in range(0, len(frequency_bin_energies)):
                energy = frequency_bin_energies[i]
                energy = max(energy, energy_threshold_low)
                energy = min(energy, energy_threshold_high)
                hue = map_energy_to_hue(hostname, energy) / 360.0
                if energy <= 1:
                    energy = energy * energy
                color = colorsys.hsv_to_rgb(hue, 0.8, 0.00 + 1 * energy/energy_threshold_high)
                r = int(color[0] * 255.0)
                g = int(color[1] * 255.0)
                b = int(color[2] * 255.0)
                coordinates = bin_to_light_map[i]
                for (side, x, y) in coordinates:
                    lc.set_color(side, x, y, (r,g,b))
            lc.show()

        else:
            time.sleep(max(0,((1./fps)-(time.time()-last_update)) * 0.99))

if __name__ == '__main__':

    run_FFT_analyzer()
