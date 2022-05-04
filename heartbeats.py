from light_control import LightControl
import colorsys
import time

# The HeartBeat animation consists of two animations
# - a slower animation that makes the background color cycle through the color wheel
#   (e.g. every 12 seconds)
# - an a faster subsequence that plays waves like the lub dub of a heartbeat
#   (e.g. every 3 seconds)
class HeartBeat:

    def __init__(self, lc, background_period=12.0, sequence_period=3.0, wave_speed=400, wait_for_keyboard_input=False):
        self.lc = lc

        self.background_period = background_period
        self.sequence_period = sequence_period
        self.wave_speed = wave_speed

        self.left_pulse_delay = 160/wave_speed
        self.right_pulse_delay = 100/wave_speed
        self.wave_delay = 260/wave_speed

        self.sequence_clock = time.time() % self.sequence_period
        self.waves_to_show = 3

        self.number_of_people = 0
        self.previous_number_of_people = 0
        self.wait_for_keyboard_input = wait_for_keyboard_input

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
                    x2 = 8 - x
                    y2 = 5 + 8 - y
                elif side == 2:
                    y2 = 5 + 10 - y

                # use this to "draw" circle using formula x^2 + y^2 = r
                # where we later compare the calculated distance(squared) with the desired radius
                distance = abs((x2**2 + (skew*y2)**2) - frame_clock)

                # adjust color of pixel within a thickness "band" from the circle
                # and vary depending on distance from center of band
                adjustment = 255-min(255, int(distance*8.5))
                if distance < 30:
                    #self.lc.add_color(side, x, y, (invert_r * adjustment, invert_g * adjustment, invert_b * adjustment))
                    self.lc.set_color(side, x, y, complementary_color)

    def _add_wave(self, invert_r, invert_g, invert_b, complementary_color, wave_start_time):
        self._add_pulse(invert_r, invert_g, invert_b, complementary_color, 0, wave_start_time, skew=1.36)
        self._add_pulse(invert_r, invert_g, invert_b, complementary_color, 1, wave_start_time + self.left_pulse_delay, skew=1)
        self._add_pulse(invert_r, invert_g, invert_b, complementary_color, 2, wave_start_time + self.right_pulse_delay, skew=1)


    def step_frame(self):
        # vary background color varies with global time
        hue = (time.time()%self.background_period)/self.background_period
        color = colorsys.hsv_to_rgb(hue, 0.6, 1.0)
        r = int(color[0] * 255.0)
        g = int(color[1] * 255.0)
        b = int(color[2] * 255.0)
        self.lc.fill_color((r, g, b))

        complementary_hue = hue + 0.5 if hue < 0.5 else hue - 0.5
        complementary_color = colorsys.hsv_to_rgb(complementary_hue, 0.6, 1.0)
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
            if self.wait_for_keyboard_input:
                self.number_of_people = int(input("Number of people: "))
            if self.number_of_people > self.previous_number_of_people:
                self.waves_to_show = 3
            else:
                self.waves_to_show = max(0, self.waves_to_show - 1)
            self.previous_number_of_people = self.number_of_people
        self.sequence_clock = new_sequence_clock

        for i in range(0, self.waves_to_show):
            self._add_wave(invert_r, invert_g, invert_b, complementary_color, wave_start_time=self.wave_delay * i)

        self.lc.show()

if __name__ == '__main__':

    hb = HeartBeat(
        LightControl(simulate=True),
        background_period=12.0,
        sequence_period=3.0,
        wave_speed=400,
        wait_for_keyboard_input=True
    )

    while True:
        hb.step_frame()
        time.sleep(0.1)
