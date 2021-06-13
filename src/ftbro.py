import numpy as np
import sys
if 1:
    sys.path.insert(0,
                    'C:/Users/emilm/Documents/fightertwister/src')
from ftbrobackend import FtBroBackend
from fightertwister import to_range, Encoder, ft_colors


def toggle_note_shift_mode(self: Encoder, ts):
    self.set_property('mode',
                      self.get_property('mode') ^ 1)
    if self.get_property('mode'):
        self.set_color(ft_colors.red)
    else:
        self.set_color(ft_colors.green)


def toggle_response_mode(self: Encoder, ts):
    self.set_property('mode',
                      self.get_property('mode') ^ 1)
    if self.get_property('mode'):
        self.set_color(ft_colors.red)
    else:
        self.set_color(ft_colors.green)


def toggle_signal_shape(self: Encoder, ts):
    self.set_property('sigshape',
                      (self.get_property('sigshape') + 1) % 3)
    print(self.get_property('sigshape'))


class FtBro(FtBroBackend):
    def __init__(self):
        super().__init__()
        for node in self.nodes:
            params = node.get_property('params')

            params[0, 3].register_cb_press(toggle_note_shift_mode)
            params[0, 3].set_property('mode', 0)

            params[1, 3].register_cb_hold(toggle_note_shift_mode)
            params[1, 3].register_cb_click(toggle_signal_shape)
            params[1, 3].set_property('mode', 0)
            params[1, 3].set_property('sigshape', 0)

    def get_shift_rate(self, node_idx):
        enc = self.nodes.ravel()[node_idx].get_property('params')[0, 3]
        return [enc.get_property('mode'), enc.value]

    def get_sine(self, node_idx):
        enc = self.nodes.ravel()[node_idx].get_property('params')[1, 3]
        mode = enc.get_property('mode') * (enc.get_property('sigshape') + 1)
        return [mode, enc.value]
