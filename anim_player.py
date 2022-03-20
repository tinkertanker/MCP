from light_control import LightControl
import time
import numpy as np
from anim_control import AnimController

class AnimPlayer():

    def __init__(self, simulate = True, frame_rate = 25, idle_anim_index = 7, idle_intro_index = -1):
        self.frame_period = 1.0/frame_rate
        self.light_controller = LightControl(simulate = simulate)
        self.anim_controller = AnimController()
        self.playing_idle = False
        self.idle_anim_index = idle_anim_index
        self.idle_intro_index = idle_intro_index
    
    def play_once(self, anim_index, delay_after=1.0):

        if self.playing_idle:
            self.light_controller.quick_fade()
        
        if anim_index < self.anim_controller.getTotalAnims():
            anim = self.anim_controller.getAnim(anim_index)
            self.stop_all_anims()  # implicitly resets frames to 0 too
            anim.setActiveState(True)
            while True:
                self._render_frame()
                self.anim_controller.advanceFrame()
                time.sleep(self.frame_period)
                if anim.getCurrentFrame() == 0:
                    anim.setActiveState(False)
                    if (delay_after > 0):
                        self.light_controller.clear()
                        self.light_controller.show()
                        time.sleep(delay_after)
                    else:
                        time.sleep(self.frame_period)
                    return
       
    def step_idle(self):
        if not self.playing_idle:
            if self.idle_intro_index:
                self.play_once(self.idle_intro_index, delay_after=0)
            self.stop_all_anims() # implicitly resets frames to 0 too, also resets playing_idle
            self.playing_idle = True
        self.anim_controller.getAnim(self.idle_anim_index).setActiveState(True)
        self._render_frame()
        self.anim_controller.advanceFrame()

    def stop_all_anims(self):
        self.playing_idle = False
        for anim_index in range(0, self.anim_controller.getTotalAnims()):
            anim = self.anim_controller.getAnim(anim_index)
            anim.resetFrame()
            anim.setActiveState(False)
    
    def _render_frame(self):
        self.light_controller.clear()
        self._render_panel(x_range=9, y_range=7, offset=0, panelNum=1)
        self._render_panel(x_range=11, y_range=7, offset=63, panelNum=2)
        self._render_panel(x_range=9, y_range=5, offset=140, panelNum=0)
        self.light_controller.show()

    def _render_panel(self, x_range, y_range, offset, panelNum):
        for x in range(0, x_range):
            for y in range(0, y_range):
                color_data = self.anim_controller.getCombinedColor(x + (y * x_range) + offset)
                self.light_controller.set_color(panelNum, x, y, [ color_data[0], color_data[1], color_data[2] ])


if __name__ == '__main__':

    anim_player = AnimPlayer(simulate=True, frame_rate=25, idle_anim_index=7, idle_intro_index=10)
    
    # for x in range(0, 13):
    #     anim_player.play_once(x)
    # anim_player.play_once(13)

    while True:
        anim_player.step_idle()
        time.sleep(1.0/25)
        # anim_player.play_once(13)

