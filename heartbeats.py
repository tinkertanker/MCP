from light_control import LightControl
import colorsys
import time

def add_pulse(lc, invert_r, invert_g, invert_b, complementary_color, side, sequence_clock, pulse_start_time, skew=1):
    pulse_clock = sequence_clock - pulse_start_time
    if pulse_clock < 0:
        return
    frame_clock = pulse_clock * 350
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
                #lc.add_color(side, x, y, (invert_r * adjustment, invert_g * adjustment, invert_b * adjustment))
                lc.set_color(side, x, y, complementary_color)

def add_wave(lc, invert_r, invert_g, invert_b, complementary_color, sequence_clock, wave_start_time, ):
    add_pulse(lc, invert_r, invert_g, invert_b, complementary_color, 0, sequence_clock, wave_start_time + 0, skew=1.36)
    add_pulse(lc, invert_r, invert_g, invert_b, complementary_color, 1, sequence_clock, wave_start_time + 0.4, 1)
    add_pulse(lc, invert_r, invert_g, invert_b, complementary_color, 2, sequence_clock, wave_start_time + 0.25, 1)


lc = LightControl(simulate=True)

# side 0 (bottom) is 9x5
# side 1 (left) is 9x7
# side 2 (right) is 11x7
# orgin of each panel is top left

animation_start_time = time.time()
pulse_start_time = time.time()
pulse_time = time.time() - pulse_start_time

while time.time() - animation_start_time < 12:
    animation_time = time.time() - animation_start_time

    # background color varies with global time
    # colorchange_period affects how quickly it will cycle through color wheel
    colorchange_period = 12.0
    hue = (time.time()%colorchange_period)/colorchange_period
    color = colorsys.hsv_to_rgb(hue, 0.6, 1.0)
    r = int(color[0] * 255.0)
    g = int(color[1] * 255.0)
    b = int(color[2] * 255.0)
    lc.fill_color((r, g, b))

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
    sequence_time = (time.time() - animation_start_time) % 2
    add_wave(lc, invert_r, invert_g, invert_b, complementary_color, sequence_time, 0, )
    add_wave(lc, invert_r, invert_g, invert_b, complementary_color, sequence_time, 1, )

    lc.show()
    time.sleep(0.1)
