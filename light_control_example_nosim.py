from light_control import LightControl
import time

lc = LightControl(simulate=False)

while True:
    for i in range(1, 10):
        lc.clear()
        lc.show()
        time.sleep(0.1)

        lc.rainbow_grid()
        lc.show()
        time.sleep(0.1)

    lc.clear()

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

    time.sleep(1.5)
