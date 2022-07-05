import argparse
import colorsys
from src.stream_analyzer import Stream_Analyzer
from light_control import LightControl
import time
import math

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=None, dest='device',
                        help='pyaudio (portaudio) device index')
    parser.add_argument('--height', type=int, default=450, dest='height',
                        help='height, in pixels, of the visualizer window')
    parser.add_argument('--n_frequency_bins', type=int, default=400, dest='frequency_bins',
                        help='The FFT features are grouped in bins')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--window_ratio', default='24/9', dest='window_ratio',
                        help='float ratio of the visualizer window. e.g. 24/9')
    parser.add_argument('--sleep_between_frames', dest='sleep_between_frames', action='store_true',
                        help='when true process sleeps between frames to reduce CPU usage (recommended for low update rates)')
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

def run_FFT_analyzer():
    args = parse_args()
    window_ratio = convert_window_ratio(args.window_ratio)

    ear = Stream_Analyzer(
                    device = args.device,        # Pyaudio (portaudio) device index, defaults to first mic input
                    rate   = None,               # Audio samplerate, None uses the default source settings
                    FFT_window_size_ms  = 60,    # Window size used for the FFT transform
                    updates_per_second  = 1000,  # How often to read the audio stream for new data
                    smoothing_length_ms = 50,    # Apply some temporal smoothing to reduce noisy features
                    n_frequency_bins = args.frequency_bins, # The FFT features are grouped in bins
                    visualize = 1,               # Visualize the FFT features with PyGame
                    verbose   = args.verbose,    # Print running statistics (latency, fps, ...)
                    height    = args.height,     # Height, in pixels, of the visualizer window,
                    window_ratio = window_ratio  # Float ratio of the visualizer window. e.g. 24/9
                    )

    fps = 15  #How often to update the FFT features + display
    last_update = time.time()
    energy_threshold_low = 2
    energy_threshold_high = 10
    while True:
        if (time.time() - last_update) > (1./fps):
            last_update = time.time()
            raw_fftx, raw_fft, binned_fftx, binned_fft = ear.get_audio_features()
            frequency_bin_energies = ear.frequency_bin_energies / ear.bin_mean_values
            print(frequency_bin_energies)
            for i in range(0, len(frequency_bin_energies)):
                energy = frequency_bin_energies[i]
                energy = max(energy, energy_threshold_low)
                energy = min(energy, energy_threshold_high)
                hue = 10 * energy / 360
                color = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
                r = int(color[0] * 255.0)
                g = int(color[1] * 255.0)
                b = int(color[2] * 255.0)
                lc.set_color(2, math.floor(i/5), i%5, (r,g,b))
            lc.show()

        elif args.sleep_between_frames:
            time.sleep(max(0,((1./fps)-(time.time()-last_update)) * 0.99))

if __name__ == '__main__':
    lc = LightControl(simulate=True, alt_mapping=False)
    for x in range(0,9):
        for y in range(0,7):
            lc.set_color(1, x, y, [x*25, y*25, 200])
            lc.show()
            time.sleep(0.01)
    run_FFT_analyzer()
