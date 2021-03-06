#!/usr/bin/env python3

"""
Light Control Module for Marina Central Project
"""

from tkinter import Tk, Canvas, Frame, BOTH
import time

pixels = None
try:
    import board
    import neopixel
    pixels = neopixel.NeoPixel(board.D18, 185, brightness=1.0, auto_write=False)
except:
    print("Couldn't initialize physical lights.")

class Lights:
    # mapping is an array of side, y, x coordinates to LED number
    mapping_1 = [
        [ # side 0,
            [ 44, 43, 42, 41, 40, 39, 38, 37, 36 ],
            [ 27, 28, 29, 30, 31, 32, 33, 34, 35 ],
            [ 26, 25, 24, 23, 22, 21, 20, 19, 18 ],
            [  9, 10, 11, 12, 13, 14, 15, 16, 17 ],
            [  8,  7,  6,  5,  4,  3,  2,  1,  0 ]
        ],
        [ # side 1 (left)
            [ 24, 25, 26, 27, 28, 29, 30, 31, 32],
            [ 23, 22, 21, 20, 37, 36, 35, 34, 33],
            [ 16, 17, 18, 19, 38, 39, 40, 41, 42],
            [ 15, 14, 13, 12, 47, 46, 45, 44, 43],
            [  8,  9, 10, 11, 48, 49, 50, 51, 52],
            [  7,  6,  5,  4, 57, 56, 55, 54, 53],
            [  0,  1,  2,  3, 58, 59, 60, 61, 62]
        ],
        [ # side 2 (right)
            [38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48],
            [37, 36, 34, 34, 33, 32, 31, 52, 51, 50, 49],
            [24, 25, 26, 27, 28, 29, 30, 53, 54, 55, 56],
            [23, 22, 21, 20, 19, 18, 17, 60, 59, 58, 57],
            [10, 11, 12, 13, 14, 15, 16, 61, 62, 63, 64],
            [ 9,  8,  7,  6,  5,  4,  3, 68, 67, 66, 65],
            [-1,  0, -1,  1, -1,  2, -1, 69, -1, 70, -1],
        ]
    ]

    # alt mapping for tiny mockup
    mapping_2 = [
        [ # side 0,
            [ 44, 43, 42, 41, 40, 39, 38, 37, 36 ],
            [ 27, 28, 29, 30, 31, 32, 33, 34, 35 ],
            [ 26, 25, 24, 23, 22, 21, 20, 19, 18 ],
            [  9, 10, 11, 12, 13, 14, 15, 16, 17 ],
            [  8,  7,  6,  5,  4,  3,  2,  1,  0 ]
        ],
        [ # side 1 (left)
            [ 54, 55, 56, 57, 58, 59, 60, 61, 62],
            [ 53, 52, 51, 50, 49, 48, 47, 46, 45],
            [ 36, 37, 38, 39, 40, 41, 42, 43, 44],
            [ 35, 34, 33, 32, 31, 30, 29, 28, 27],
            [ 18, 19, 20, 21, 22, 23, 24, 25, 26],
            [ 17, 16, 15, 14, 13, 12, 11, 10,  9],
            [  0,  1,  2,  3,  4,  5,  6,  7,  8]
        ],
        [ # side 2 (right)
            [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10],
            [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11],
            [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
            [43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33],
            [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
            [65, 64, 63, 62, 61, 60, 59, 58, 57, 56, 55],
            [-1, 67, -1, 69, -1, 71, -1, 73, -1, 75, -1],
        ]
    ]

    def __init__(self, alt_mapping=False):
        self.alt_mapping=alt_mapping
        self.offset = 1
        self.side_2_offset = 63
        self.side_0_offset = 63 + 71
        self.mapping = self.mapping_1
        if alt_mapping:
            self.mapping = self.mapping_2
            self.offset = 0
            self.side_0_offset = 63 + 71 + 6


    def set_color(self, side, x, y, color=(255,255,255)):
        index = self.mapping[side][y][x]
        if index >= 0 and index != -1:
            if(side == 2):
                index += self.side_2_offset
            if(side == 0):
                index += self.side_0_offset
            
            pixels[index + self.offset] = tuple(color)

    def clear(self):
        pixels.fill((0,0,0))

    def show(self):
        pixels.show()

class Simulator(Frame):

    def __init__(self):
        super().__init__()

        self.init_ui()


    def init_ui(self):

        self.master.title("YUME + Tinkertanker Marina Central Project")
        self.config(bg='black')
        self.pack(fill=BOTH, expand=1)

        self.canvas = Canvas(self, bg='black')
        self.canvas.configure(bg='black')
        self.canvas.create_line(195, 20, 195, 220, fill='white')
        self.canvas.create_line(20, 320, 190, 235, fill='white')
        self.canvas.create_line(200, 230, 420, 340, fill='white')
        self.init_dots()

        self.canvas.pack(fill=BOTH, expand=1)

    def init_dots(self):
        dots = [[],[],[]]

        for x in range(0,9):
            dots[1].append([])
            for y in range(0,7):
                dots[1][x].append(self.create_dot(1, x, y))

        for x in range(0,11):
            dots[2].append([])
            for y in range(0,7):
                dots[2][x].append(self.create_dot(2, x, y))

        for x in range(0,9):
            dots[0].append([])
            for y in range(0,5):
                dots[0][x].append(self.create_dot(0, x, y))
        
        self.dots = dots

    def create_dot(self, side, x, y):
        origin_x, origin_y, relative_x, relative_y, width, height = 0, 0, 0, 0, 2, 2
        x_spacing = 20
        y_spacing = 30
        if (side==1): # Left
            origin_x = 20
            origin_y = 100
            relative_x = x_spacing * x
            relative_y = y_spacing * y - 10 * x
            if (x + y) % 2 == 0:
                height = 5
        elif (side==2): # Right
            if y == 6 and x % 2 == 0:
                return None
            origin_x = 10 * x_spacing + 10
            origin_y = 35
            relative_x = x_spacing * x
            relative_y = y_spacing * y + 10 * x
            if (x + y) % 2 == 1:
                height = 5
        elif (side==0): # Bottom
            origin_x = 40
            origin_y = 130 + y_spacing * 7
            relative_x = x_spacing * x + 40 * y
            relative_y = y_spacing/3*2 * y - 10 * x

        return self.canvas.create_rectangle(
            origin_x + relative_x,
            origin_y + relative_y,
            origin_x + width + relative_x,
            origin_y + height + relative_y,
            outline='black', fill='black')

    def set_color(self, side, x, y, color=(255,255,255)):
        rgbstring = "#{0}".format(''.join('%02x'%c for c in color))
        self.canvas.itemconfig(self.dots[side][x][y], outline=rgbstring, fill=rgbstring)
    
    def clear(self):
        for x in range(0,9):
            for y in range(0,7):
                self.set_color(1, x, y, [0, 0, 0])

        for x in range(0,11):
            for y in range(0,7):
                self.set_color(2, x, y, [0, 0, 0])

        for x in range(0,9):
            for y in range(0,5):
                self.set_color(0, x, y, [0, 0, 0]) 

class LightControl:

    def __init__(self, simulate=True, alt_mapping=False):
        self.simulate = simulate
        self.sim_frame, self.tk_root = None, None
        self.lights = None

        self.buffer = [[],[],[]]
        for x in range(0,9):
            self.buffer[1].append([])
            for y in range(0,7):
                self.buffer[1][x].append((0,0,0))
        for x in range(0,11):
            self.buffer[2].append([])
            for y in range(0,7):
                self.buffer[2][x].append((0,0,0))
        for x in range(0,9):
            self.buffer[0].append([])
            for y in range(0,5):
                self.buffer[0][x].append((0,0,0))

        if pixels is not None:
            self.lights = Lights(alt_mapping=alt_mapping)

        if simulate:
            self.tk_root = Tk()
            self.sim_frame = Simulator()
            self.tk_root.geometry("430x450+300+300")
            self.tk_root.update()

    # sides:
    # 0 - bottom
    # 1 - left
    # 2 - right
    def set_color(self, side, x, y, color=(255,255,255)):
        # ignore values that are out of range
        if side >= len(self.buffer) or x >= len(self.buffer[side]) or y >= len(self.buffer[side][y]):
            return

        r = max(0, min(color[0], 255))
        g = max(0, min(color[1], 255))
        b = max(0, min(color[2], 255))

        if self.simulate:
            self.sim_frame.set_color(side, x, y, (r,g,b))
        if self.lights:
            self.lights.set_color(side, x, y, (r,g,b))
        self.buffer[side][x][y] = (r,g,b)
    
    def get_color(self, side, x, y):
        # ignore values that are out of range
        if side >= len(self.buffer) or x >= len(self.buffer[side]) or y >= len(self.buffer[side][x]):
            return

        return self.buffer[side][x][y]

    def add_color(self, side, x, y, color=(255,255,255)):
        # ignore values that are out of range
        if side >= len(self.buffer) or x >= len(self.buffer[side]) or y >= len(self.buffer[side][x]):
            return

        original_color = self.get_color(side, x, y)
        new_color_unclipped = (
            original_color[0] + color[0],
            original_color[1] + color[1],
            original_color[2] + color[2],
        )
        self.set_color(side, x, y, new_color_unclipped)

    def quick_fade(self):
        for i in range(0,16):
            for x in range(0,9):
                for y in range(0,7):
                    self.set_color(1, x, y, tuple([int(c/1.3) for c in self.buffer[1][x][y]]))

            for x in range(0,11):
                for y in range(0,7):
                    self.set_color(2, x, y, tuple([int(c/1.3) for c in self.buffer[2][x][y]]))

            for x in range(0,9):
                for y in range(0,5):
                    self.set_color(0, x, y, tuple([int(c/1.3) for c in self.buffer[0][x][y]]))
            self.show()
            time.sleep(0.01)

    def clear(self):
        if self.simulate:
            self.sim_frame.clear()
        if self.lights:
            self.lights.clear()

    def fill_color(self, color):
        for x in range(0,9):
            for y in range(0,7):
                self.set_color(1, x, y, color)

        for x in range(0,11):
            for y in range(0,7):
                self.set_color(2, x, y, color)

        for x in range(0,9):
            for y in range(0,5):
                self.set_color(0, x, y, color)

    def rainbow_grid(self):
        for x in range(0,9):
            for y in range(0,7):
                self.set_color(1, x, y, [x*25, y*25, 200])

        for x in range(0,11):
            for y in range(0,7):
                self.set_color(2, x, y, [x*25, 200, y*25])

        for x in range(0,9):
            for y in range(0,5):
                self.set_color(0, x, y, [200, x*25, y*25])

    def show(self):
        if self.simulate:
            self.tk_root.update()
        if self.lights:
            self.lights.show()


if __name__ == '__main__':
    lc = LightControl(simulate=True, alt_mapping=False)
    lc.rainbow_grid()
    lc.show()
    input('Press Enter to quit')
