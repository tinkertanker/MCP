#!/usr/bin/env python3

"""
Light Control Module for Marina Central Project
"""

from tkinter import Tk, Canvas, Frame, BOTH

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

    def __init__(self, simulate=True):
        self.simulate = simulate
        self.sim_frame, self.tk_root = None, None

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
        if self.simulate:
            self.sim_frame.set_color(side, x, y, color)
    
    def clear(self):
        if self.simulate:
            self.sim_frame.clear()

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


if __name__ == '__main__':
    lc = LightControl(simulate=True)
    lc.rainbow_grid()
    lc.show()
    input('Press Enter to quit')