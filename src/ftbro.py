import numpy as np
import sys
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

    def __enter__(self):
        super().__enter__()
        self.do_task_delay(100, self.foo)

    def get_shift_rate(self, node_idx):
        enc = self.nodes.ravel()[node_idx].get_property('params')[0, 3]
        return [enc.get_property('mode'), enc.value]

    def get_sine(self, node_idx):
        enc = self.nodes.ravel()[node_idx].get_property('params')[1, 3]
        mode = enc.get_property('mode') * (enc.get_property('sigshape') + 1)
        return [mode, enc.value]

    def get_node_values(self, i):
        node = self.nodes[np.unravel_index(i, self.nodes.shape)]
        return node.value, node.get_property('params').value

    def foo(self):
        self.get_node_values(0)
        self.do_task_delay(100, self.foo)


if __name__ == '__main__':
    ft = FtBro()
    with ft:
        input('press enter to quit')
