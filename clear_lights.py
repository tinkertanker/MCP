#!/usr/bin/env python3

from light_control import LightControl
import os

lc = LightControl(simulate=False)
lc.clear()
lc.show()
os.system('uhubctl -a off -l 2 -r 100')