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

    def __init__(self, lc, background_period=12.0, sequence_period=3.0, wave_speed=400, randomize_people_count=False, wait_for_keyboard_input=False):
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
        self.main_pulse_delay = 0/wave_speed
        self.left_pulse_delay = 100/wave_speed
        self.right_pulse_delay = 40/wave_speed

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
                distancesq = abs((x2**2 + (skew*y2)**2) - frame_clock)

                # adjust color of pixel within a thickness "band" from the circle
                # and vary depending on distance from center of band
                thresholdsq = 20
                if distancesq < thresholdsq:
                    distance = distancesq**0.5
                    threshold = thresholdsq**0.5
                    adjustment = 255-min(255, int(distance*3.2))
                    destination_color = complementary_color
                    destination_color_factor = 1
                    if distancesq > thresholdsq / 3:
                        # fade near the edges
                        destination_color_factor = (thresholdsq - distancesq) / thresholdsq
                    current_color_factor = 1 - destination_color_factor
                    self.lc.set_color(side, x, y,
                        (int(destination_color[0] * destination_color_factor + self.background_color[0] * current_color_factor),
                        int(destination_color[1] * destination_color_factor + self.background_color[1] * current_color_factor),
                        int(destination_color[2] * destination_color_factor + self.background_color[2] * current_color_factor)))
                    #self.lc.add_color(side, x, y, (invert_r * adjustment, invert_g * adjustment, invert_b * adjustment))
                    #self.lc.set_color(side, x, y, complementary_color)

    def _add_wave(self, invert_r, invert_g, invert_b, complementary_color, wave_start_time):
        self._add_pulse(invert_r, invert_g, invert_b, complementary_color, 0, wave_start_time + self.main_pulse_delay, skew=1.36)
        self._add_pulse(invert_r, invert_g, invert_b, complementary_color, 1, wave_start_time + self.left_pulse_delay, skew=0.6)
        self._add_pulse(invert_r, invert_g, invert_b, complementary_color, 2, wave_start_time + self.right_pulse_delay, skew=0.6)


    def step_frame(self):
        # vary background color varies with global time
        hue = (200 + abs(((time.time()%self.background_period)/self.background_period) * 190 - 95)) / 360
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
                self.number_of_people = random.randint(0,10)
                print(f"{self.number_of_people} People")
            if self.wait_for_keyboard_input:
                self.number_of_people = int(input("Number of people: "))
            # set speed based on number of people
            #self._set_wave_speed(100 + 50 * min(5, self.number_of_people))
            self._set_wave_speed(100 * min(5, int(self.number_of_people/2)+1))
            self.previous_number_of_people = self.number_of_people
        self.sequence_clock = new_sequence_clock

        for i in range(0, self.waves_to_show):
            self._add_wave(invert_r, invert_g, invert_b, complementary_color, wave_start_time=self.wave_delay * i)

        self.lc.show()

if __name__ == '__main__':

    from tkinter import Tk
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--enable-simulator", help="Enable Light Simulator", action="store_true", default=False)
    parser.add_argument("-r", "--randomize-people-count", help="Randomize People Count", action="store_true", default=False)
    parser.add_argument("-k", "--wait-for-keyboard", help="Wait for Keyboard Input", action="store_true", default=False)
    parser.add_argument("-a", "--alt-mapping", help="Alt LED Mapping", action="store_true",default=False)
    args = parser.parse_args()

    lc = LightControl(simulate=args.enable_simulator, alt_mapping=args.alt_mapping)

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

    while True:
        hb.step_frame()
        time.sleep(0.1)
